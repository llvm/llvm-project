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
#include "URI.h"
#include "index/Background.h"
#include "index/FileIndex.h"
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

enum class IndexOutputMode { Monolithic, Sharded };

namespace {

static llvm::cl::opt<IndexFileFormat>
    Format("format", llvm::cl::desc("Format of the index to be written"),
           llvm::cl::values(clEnumValN(IndexFileFormat::YAML, "yaml",
                                       "human-readable YAML format"),
                            clEnumValN(IndexFileFormat::RIFF, "binary",
                                       "binary RIFF format")),
           llvm::cl::init(IndexFileFormat::RIFF));

static llvm::cl::opt<IndexOutputMode> OutputMode(
    "index-type", llvm::cl::desc("Type of index output"),
    llvm::cl::values(
        clEnumValN(IndexOutputMode::Monolithic, "monolithic",
                   "Single merged index file written to stdout (default)"),
        clEnumValN(IndexOutputMode::Sharded, "sharded",
                   "Per-file shards written to disk")),
    llvm::cl::init(IndexOutputMode::Monolithic));

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
        "Path to the project root for --index-type=sharded. "
        "Determines where to store index shards. Shards are stored in "
        "<project-root>/.cache/clangd/index/. "
        "Defaults to current directory if not specified."),
};

// Tracks the content digest of the last shard written for a given file.
// Used to avoid writing duplicate shards when multiple TUs include the same
// header.
struct ShardVersion {
  FileDigest Digest{{0}};
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
    return createStaticIndexingAction(Opts, [&](IndexFileIn TUResult) {
      {
        // Merge as we go.
        std::lock_guard<std::mutex> Lock(SymbolsMu);
        for (const auto &Sym : *TUResult.Symbols) {
          if (const auto *Existing = Symbols.find(Sym.ID))
            Symbols.insert(mergeSymbol(*Existing, Sym));
          else
            Symbols.insert(Sym);
        }
      }
      {
        std::lock_guard<std::mutex> Lock(RefsMu);
        for (const auto &Sym : *TUResult.Refs) {
          // Deduplication happens during insertion.
          for (const auto &Ref : Sym.second)
            Refs.insert(Sym.first, Ref);
        }
      }
      {
        std::lock_guard<std::mutex> Lock(RelsMu);
        for (const auto &R : *TUResult.Relations) {
          Relations.insert(R);
        }
      }
      // FIXME: Handle TUResult.Sources?
    });
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

// Action factory that writes per-file shards (for sharded index format).
// Each TU's index data is sharded independently — no merging across TUs.
// Header shards are deduplicated: if a header's content hasn't changed since
// the last time we wrote its shard, we skip writing it again.
class ShardedIndexActionFactory : public tooling::FrontendActionFactory {
public:
  ShardedIndexActionFactory(BackgroundIndexStorage &Storage)
      : Storage(Storage) {}

  std::unique_ptr<FrontendAction> create() override {
    // Snapshot the current shard versions so the callback can check staleness
    // without holding the lock during indexing.
    llvm::StringMap<ShardVersion> Snapshot;
    {
      std::lock_guard<std::mutex> Lock(ShardVersionsMu);
      Snapshot = ShardVersions;
    }

    SymbolCollector::Options Opts;
    Opts.CountReferences = true;

    return createStaticIndexingAction(
        Opts, [this, Snapshot](IndexFileIn Index) {
          writeShards(std::move(Index), Snapshot);
        });
  }

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    disableUnsupportedOptions(*Invocation);
    return tooling::FrontendActionFactory::runInvocation(
        std::move(Invocation), Files, std::move(PCHContainerOps), DiagConsumer);
  }

private:
  void writeShards(IndexFileIn Index,
                   const llvm::StringMap<ShardVersion> &Snapshot) {
    if (!Index.Sources)
      return;

    // Find the URI of the main file (the TU root). Store as owned string
    // because we move Index below, which would invalidate any StringRef into it.
    std::string MainUri;
    for (const auto &[Uri, Node] : *Index.Sources) {
      if (Node.Flags & IncludeGraphNode::SourceFlag::IsTU) {
        MainUri = Uri.str();
        break;
      }
    }

    // Collect files that need updated shards, based on content digest.
    // Files whose shard was already written with the same digest are skipped.
    llvm::StringMap<std::pair<Path, FileDigest>> FilesToUpdate;
    for (const auto &[Uri, Node] : *Index.Sources) {
      auto AbsPath = URI::resolve(Uri, MainUri);
      if (!AbsPath) {
        elog("Failed to resolve URI {0}: {1}", Uri, AbsPath.takeError());
        continue;
      }
      auto DigestIt = Snapshot.find(*AbsPath);
      if (DigestIt == Snapshot.end() || DigestIt->second.Digest != Node.Digest)
        FilesToUpdate[Uri] = {std::move(*AbsPath), Node.Digest};
    }

    // Shard the index data by file.
    FileShardedIndex ShardedIndex(std::move(Index));

    unsigned Written = 0;
    for (const auto &[Uri, PathAndDigest] : FilesToUpdate) {
      auto Shard = ShardedIndex.getShard(Uri);
      if (!Shard) {
        elog("Failed to get shard for {0}", Uri);
        continue;
      }
      PathRef Path = PathAndDigest.first;

      // Command line is only meaningful for the TU's main file.
      if (Uri != MainUri)
        Shard->Cmd.reset();

      IndexFileOut Out(*Shard);
      Out.Format = IndexFileFormat::RIFF;
      if (auto Err = Storage.storeShard(Path, Out)) {
        elog("Failed to write shard for {0}: {1}", Path, std::move(Err));
      } else {
        std::lock_guard<std::mutex> Lock(ShardVersionsMu);
        ShardVersions[Path].Digest = PathAndDigest.second;
        ++Written;
      }
    }

    log("Wrote {0} shards for TU {1}", Written, MainUri);
  }

  BackgroundIndexStorage &Storage;
  std::mutex ShardVersionsMu;
  llvm::StringMap<ShardVersion> ShardVersions;
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

  Example usage for sharded index (writes shards to disk):

  $ clangd-indexer --index-type=sharded --executor=all-TUs build/

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

  // Handle sharded index type separately - writes per-file shards.
  if (clang::clangd::OutputMode == clang::clangd::IndexOutputMode::Sharded) {
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
        std::make_unique<clang::clangd::ShardedIndexActionFactory>(*Storage),
        std::move(Adjuster));
    if (Err) {
      clang::clangd::elog("{0}", std::move(Err));
      return 1;
    }

    llvm::errs() << "Index shards written to " << Root
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
