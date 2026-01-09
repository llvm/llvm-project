//===--- IndexerMain.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// clangd-indexer is a tool to gather index data (symbols, xrefs) from source.
//
//===----------------------------------------------------------------------===//

#include "CompileCommands.h"
#include "Compiler.h"
#include "GlobalCompilationDatabase.h"
#include "index/Background.h"
#include "index/IndexAction.h"
#include "index/Merge.h"
#include "index/Ref.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolCollector.h"
#include "support/Logger.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include <utility>

namespace clang {
namespace clangd {
namespace {

static llvm::cl::opt<IndexFileFormat> Format(
    "format", llvm::cl::desc("Format of the index to be written"),
    llvm::cl::values(
        clEnumValN(IndexFileFormat::YAML, "yaml", "human-readable YAML format"),
        clEnumValN(IndexFileFormat::RIFF, "binary", "binary RIFF format"),
        clEnumValN(IndexFileFormat::BACKGROUND, "background",
                   "background index format for language servers")),
    llvm::cl::init(IndexFileFormat::RIFF));

static llvm::cl::list<std::string> QueryDriverGlobs{
    "query-driver",
    llvm::cl::desc(
        "Comma separated list of globs for white-listing gcc-compatible "
        "drivers that are safe to execute. Drivers matching any of these globs "
        "will be used to extract system includes. e.g. "
        "/usr/bin/**/clang-*,/path/to/repo/**/g++-*"),
    llvm::cl::CommaSeparated,
};

static llvm::cl::opt<std::string> ProjectRoot{
    "project-root",
    llvm::cl::desc(
        "Path to the project root for --format=background. "
        "Determines where to store index shards. Shards are stored in "
        "<project-root>/.cache/clangd/index/. "
        "Defaults to current directory if not specified."),
};

// Action factory that merges all symbols into a single index (for YAML/RIFF).
class IndexActionFactory : public tooling::FrontendActionFactory {
public:
  IndexActionFactory(IndexFileIn &Result) : Result(Result) {}

  std::unique_ptr<FrontendAction> create() override {
    SymbolCollector::Options Opts;
    Opts.CountReferences = true;
    Opts.FileFilter = [&](const SourceManager &SM, FileID FID) {
      const auto F = SM.getFileEntryRefForID(FID);
      if (!F)
        return false; // Skip invalid files.
      auto AbsPath = getCanonicalPath(*F, SM.getFileManager());
      if (!AbsPath)
        return false; // Skip files without absolute path.
      std::lock_guard<std::mutex> Lock(FilesMu);
      return Files.insert(*AbsPath).second; // Skip already processed files.
    };
    return createStaticIndexingAction(
        Opts,
        [&](SymbolSlab S) {
          // Merge as we go.
          std::lock_guard<std::mutex> Lock(SymbolsMu);
          for (const auto &Sym : S) {
            if (const auto *Existing = Symbols.find(Sym.ID))
              Symbols.insert(mergeSymbol(*Existing, Sym));
            else
              Symbols.insert(Sym);
          }
        },
        [&](RefSlab S) {
          std::lock_guard<std::mutex> Lock(RefsMu);
          for (const auto &Sym : S) {
            // Deduplication happens during insertion.
            for (const auto &Ref : Sym.second)
              Refs.insert(Sym.first, Ref);
          }
        },
        [&](RelationSlab S) {
          std::lock_guard<std::mutex> Lock(RelsMu);
          for (const auto &R : S) {
            Relations.insert(R);
          }
        },
        /*IncludeGraphCallback=*/nullptr);
  }

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    disableUnsupportedOptions(*Invocation);
    return tooling::FrontendActionFactory::runInvocation(
        std::move(Invocation), Files, std::move(PCHContainerOps), DiagConsumer);
  }

  // Awkward: we write the result in the destructor, because the executor
  // takes ownership so it's the easiest way to get our data back out.
  ~IndexActionFactory() {
    Result.Symbols = std::move(Symbols).build();
    Result.Refs = std::move(Refs).build();
    Result.Relations = std::move(Relations).build();
  }

private:
  IndexFileIn &Result;
  std::mutex FilesMu;
  llvm::StringSet<> Files;
  std::mutex SymbolsMu;
  SymbolSlab::Builder Symbols;
  std::mutex RefsMu;
  RefSlab::Builder Refs;
  std::mutex RelsMu;
  RelationSlab::Builder Relations;
};

// Action factory that writes per-file shards (for background index format).
class BackgroundIndexActionFactory : public tooling::FrontendActionFactory {
public:
  BackgroundIndexActionFactory(BackgroundIndexStorage &Storage)
      : Storage(Storage), Symbols(std::make_unique<SymbolSlab::Builder>()),
        Refs(std::make_unique<RefSlab::Builder>()),
        Relations(std::make_unique<RelationSlab::Builder>()) {}

  std::unique_ptr<FrontendAction> create() override {
    SymbolCollector::Options Opts;
    Opts.CountReferences = true;
    Opts.FileFilter = [&](const SourceManager &SM, FileID FID) {
      const auto F = SM.getFileEntryRefForID(FID);
      if (!F)
        return false;
      auto AbsPath = getCanonicalPath(*F, SM.getFileManager());
      if (!AbsPath)
        return false;
      std::lock_guard<std::mutex> Lock(FilesMu);
      return Files.insert(*AbsPath).second;
    };
    return createStaticIndexingAction(
        Opts,
        [&](SymbolSlab S) {
          std::lock_guard<std::mutex> Lock(SymbolsMu);
          for (const auto &Sym : S) {
            if (const auto *Existing = Symbols->find(Sym.ID))
              Symbols->insert(mergeSymbol(*Existing, Sym));
            else
              Symbols->insert(Sym);
          }
        },
        [&](RefSlab S) {
          std::lock_guard<std::mutex> Lock(RefsMu);
          for (const auto &Sym : S) {
            for (const auto &Ref : Sym.second)
              Refs->insert(Sym.first, Ref);
          }
        },
        [&](RelationSlab S) {
          std::lock_guard<std::mutex> Lock(RelsMu);
          for (const auto &R : S)
            Relations->insert(R);
        },
        /*IncludeGraphCallback=*/nullptr);
  }

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    disableUnsupportedOptions(*Invocation);

    // Get the main file path before running.
    std::string MainFile;
    if (!Invocation->getFrontendOpts().Inputs.empty())
      MainFile = Invocation->getFrontendOpts().Inputs[0].getFile().str();

    bool Success = tooling::FrontendActionFactory::runInvocation(
        std::move(Invocation), Files, std::move(PCHContainerOps), DiagConsumer);

    // After processing, write a shard for this file.
    if (Success && !MainFile.empty())
      writeShardForFile(MainFile);

    return Success;
  }

private:
  void writeShardForFile(llvm::StringRef MainFile) {
    IndexFileIn Data;
    {
      std::lock_guard<std::mutex> Lock(SymbolsMu);
      Data.Symbols = std::move(*Symbols).build();
      Symbols = std::make_unique<SymbolSlab::Builder>();
    }
    {
      std::lock_guard<std::mutex> Lock(RefsMu);
      Data.Refs = std::move(*Refs).build();
      Refs = std::make_unique<RefSlab::Builder>();
    }
    {
      std::lock_guard<std::mutex> Lock(RelsMu);
      Data.Relations = std::move(*Relations).build();
      Relations = std::make_unique<RelationSlab::Builder>();
    }

    IndexFileOut Out(Data);
    Out.Format = IndexFileFormat::RIFF; // Shards use RIFF format.

    if (auto Err = Storage.storeShard(MainFile, Out)) {
      elog("Failed to write shard for {0}: {1}", MainFile, std::move(Err));
    } else {
      std::lock_guard<std::mutex> Lock(FilesMu);
      ++ShardsWritten;
      log("Wrote shard for {0} ({1} total)", MainFile, ShardsWritten);
    }
  }

  BackgroundIndexStorage &Storage;
  std::mutex FilesMu;
  llvm::StringSet<> Files;
  unsigned ShardsWritten = 0;
  std::mutex SymbolsMu;
  std::unique_ptr<SymbolSlab::Builder> Symbols;
  std::mutex RefsMu;
  std::unique_ptr<RefSlab::Builder> Refs;
  std::mutex RelsMu;
  std::unique_ptr<RelationSlab::Builder> Relations;
};

} // namespace
} // namespace clangd
} // namespace clang

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  const char *Overview = R"(
  Creates an index of symbol information etc in a whole project.

  Example usage for a project using CMake compile commands:

  $ clangd-indexer --executor=all-TUs compile_commands.json > clangd.dex

  Example usage for file sequence index without flags:

  $ clangd-indexer File1.cpp File2.cpp ... FileN.cpp > clangd.dex

  Example usage for background index format (writes shards to disk):

  $ clangd-indexer --format=background --executor=all-TUs build/

  This writes index shards to .cache/clangd/index/ in the current directory.
  Use --project-root to specify a different location for the shards.

  Note: only symbols from header files will be indexed.
  )";

  auto Executor = clang::tooling::createExecutorFromCommandLineArgs(
      argc, argv, llvm::cl::getGeneralCategory(), Overview);

  if (!Executor) {
    llvm::errs() << llvm::toString(Executor.takeError()) << "\n";
    return 1;
  }

  auto Mangler = std::make_shared<clang::clangd::CommandMangler>(
      clang::clangd::CommandMangler::detect());
  Mangler->SystemIncludeExtractor = clang::clangd::getSystemIncludeExtractor(
      static_cast<llvm::ArrayRef<std::string>>(
          clang::clangd::QueryDriverGlobs));

  auto Adjuster = clang::tooling::ArgumentsAdjuster(
      [Mangler = std::move(Mangler)](const std::vector<std::string> &Args,
                                     llvm::StringRef File) {
        clang::tooling::CompileCommand Cmd;
        Cmd.CommandLine = Args;
        Mangler->operator()(Cmd, File);
        return Cmd.CommandLine;
      });

  // Handle background index format separately - writes per-file shards.
  if (clang::clangd::Format == clang::clangd::IndexFileFormat::BACKGROUND) {
    // Default to current directory if --project-root not specified.
    std::string Root = clang::clangd::ProjectRoot;
    if (Root.empty()) {
      llvm::SmallString<256> CurrentDir;
      if (auto EC = llvm::sys::fs::current_path(CurrentDir)) {
        llvm::errs() << "Error: Failed to get current directory: "
                     << EC.message() << "\n";
        return 1;
      }
      Root = std::string(CurrentDir);
    }

    // Create storage factory for disk-backed index shards.
    auto IndexStorageFactory =
        clang::clangd::BackgroundIndexStorage::createDiskBackedStorageFactory(
            [Root](clang::clangd::PathRef) {
              return clang::clangd::ProjectInfo{Root};
            });

    // Get storage for the project root.
    clang::clangd::BackgroundIndexStorage *Storage = IndexStorageFactory(Root);

    auto Err = Executor->get()->execute(
        std::make_unique<clang::clangd::BackgroundIndexActionFactory>(*Storage),
        std::move(Adjuster));
    if (Err) {
      clang::clangd::elog("{0}", std::move(Err));
      return 1;
    }

    llvm::errs() << "Background index shards written to " << Root
                 << "/.cache/clangd/index/\n";
    return 0;
  }

  // Standard mode: collect and merge symbols, then emit to stdout.
  clang::clangd::IndexFileIn Data;
  auto Err = Executor->get()->execute(
      std::make_unique<clang::clangd::IndexActionFactory>(Data),
      std::move(Adjuster));
  if (Err) {
    clang::clangd::elog("{0}", std::move(Err));
  }

  // Emit collected data.
  clang::clangd::IndexFileOut Out(Data);
  Out.Format = clang::clangd::Format;
  llvm::outs() << Out;
  return 0;
}
