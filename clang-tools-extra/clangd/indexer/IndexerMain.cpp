//===--- IndexerMain.cpp -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// clangd-indexer is a tool to gather index data (symbols, xrefs) from source.
//
//===----------------------------------------------------------------------===//

#include "index/Index.h"
#include "index/IndexAction.h"
#include "index/Merge.h"
#include "index/Serialization.h"
#include "index/SymbolCollector.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

using namespace llvm;
using namespace clang::tooling;
using clang::clangd::SymbolSlab;

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

  clang::FrontendAction *create() override {
    SymbolCollector::Options Opts;
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
                 std::lock_guard<std::mutex> Lock(SymbolsMu);
                 for (const auto &Sym : S) {
                   // No need to merge as currently all Refs are from main file.
                   for (const auto &Ref : Sym.second)
                     Refs.insert(Sym.first, Ref);
                 }
               })
        .release();
  }

  // Awkward: we write the result in the destructor, because the executor
  // takes ownership so it's the easiest way to get our data back out.
  ~IndexActionFactory() {
    Result.Symbols = std::move(Symbols).build();
    Result.Refs = std::move(Refs).build();
  }

private:
  IndexFileIn &Result;
  std::mutex SymbolsMu;
  SymbolSlab::Builder Symbols;
  RefSlab::Builder Refs;
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
      argc, argv, cl::GeneralCategory, Overview);

  if (!Executor) {
    llvm::errs() << llvm::toString(Executor.takeError()) << "\n";
    return 1;
  }

  // Collect symbols found in each translation unit, merging as we go.
  clang::clangd::IndexFileIn Data;
  auto Err = Executor->get()->execute(
      llvm::make_unique<clang::clangd::IndexActionFactory>(Data));
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
  }

  // Emit collected data.
  clang::clangd::IndexFileOut Out(Data);
  Out.Format = clang::clangd::Format;
  llvm::outs() << Out;
  return 0;
}
