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

static llvm::cl::opt<IndexFileFormat>
    Format("format", llvm::cl::desc("Format of the index to be written"),
           llvm::cl::values(clEnumValN(IndexFileFormat::YAML, "yaml",
                                       "human-readable YAML format"),
                            clEnumValN(IndexFileFormat::RIFF, "binary",
                                       "binary RIFF format")),
           llvm::cl::init(IndexFileFormat::RIFF));

class IndexActionFactory : public tooling::FrontendActionFactory {
public:
  IndexActionFactory(IndexFileIn &Result) : Result(Result) {}

  std::unique_ptr<FrontendAction> create() override {
    SymbolCollector::Options Opts;
    Opts.CountReferences = true;
    Opts.FileFilter = [&](const SourceManager &SM, FileID FID) {
      const auto *F = SM.getFileEntryForID(FID);
      if (!F)
        return false; // Skip invalid files.
      auto AbsPath = getCanonicalPath(F, SM);
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

  Note: only symbols from header files will be indexed.
  )";

  auto Executor = clang::tooling::createExecutorFromCommandLineArgs(
      argc, argv, llvm::cl::getGeneralCategory(), Overview);

  if (!Executor) {
    llvm::errs() << llvm::toString(Executor.takeError()) << "\n";
    return 1;
  }

  // Collect symbols found in each translation unit, merging as we go.
  clang::clangd::IndexFileIn Data;
  auto Err = Executor->get()->execute(
      std::make_unique<clang::clangd::IndexActionFactory>(Data),
      clang::tooling::ArgumentsAdjuster(
          [Mangler = std::make_shared<clang::clangd::CommandMangler>(
               clang::clangd::CommandMangler::detect())](
              const std::vector<std::string> &Args, llvm::StringRef File) {
            clang::tooling::CompileCommand Cmd;
            Cmd.CommandLine = Args;
            Mangler->operator()(Cmd, File);
            return Cmd.CommandLine;
          }));
  if (Err) {
    clang::clangd::elog("{0}", std::move(Err));
  }

  // Emit collected data.
  clang::clangd::IndexFileOut Out(Data);
  Out.Format = clang::clangd::Format;
  llvm::outs() << Out;
  return 0;
}
