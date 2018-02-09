//===--- GlobalSymbolBuilderMain.cpp -----------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// GlobalSymbolBuilder is a tool to generate YAML-format symbols across the
// whole project. This tools is for **experimental** only. Don't use it in
// production code.
//
//===---------------------------------------------------------------------===//

#include "index/Index.h"
#include "index/Merge.h"
#include "index/SymbolCollector.h"
#include "index/SymbolYAML.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;
using namespace clang::tooling;
using clang::clangd::SymbolSlab;

namespace clang {
namespace clangd {
namespace {

static llvm::cl::opt<std::string> AssumedHeaderDir(
    "assume-header-dir",
    llvm::cl::desc("The index includes header that a symbol is defined in. "
                   "If the absolute path cannot be determined (e.g. an "
                   "in-memory VFS) then the relative path is resolved against "
                   "this directory, which must be absolute. If this flag is "
                   "not given, such headers will have relative paths."),
    llvm::cl::init(""));

class SymbolIndexActionFactory : public tooling::FrontendActionFactory {
public:
  SymbolIndexActionFactory(tooling::ExecutionContext *Ctx) : Ctx(Ctx) {}

  clang::FrontendAction *create() override {
    // Wraps the index action and reports collected symbols to the execution
    // context at the end of each translation unit.
    class WrappedIndexAction : public WrapperFrontendAction {
    public:
      WrappedIndexAction(std::shared_ptr<SymbolCollector> C,
                         const index::IndexingOptions &Opts,
                         tooling::ExecutionContext *Ctx)
          : WrapperFrontendAction(
                index::createIndexingAction(C, Opts, nullptr)),
            Ctx(Ctx), Collector(C) {}

      void EndSourceFileAction() override {
        WrapperFrontendAction::EndSourceFileAction();

        auto Symbols = Collector->takeSymbols();
        for (const auto &Sym : Symbols) {
          std::string IDStr;
          llvm::raw_string_ostream OS(IDStr);
          OS << Sym.ID;
          Ctx->reportResult(OS.str(), SymbolToYAML(Sym));
        }
      }

    private:
      tooling::ExecutionContext *Ctx;
      std::shared_ptr<SymbolCollector> Collector;
    };

    index::IndexingOptions IndexOpts;
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::All;
    IndexOpts.IndexFunctionLocals = false;
    auto CollectorOpts = SymbolCollector::Options();
    CollectorOpts.FallbackDir = AssumedHeaderDir;
    return new WrappedIndexAction(
        std::make_shared<SymbolCollector>(std::move(CollectorOpts)), IndexOpts,
        Ctx);
  }

  tooling::ExecutionContext *Ctx;
};

// Combine occurrences of the same symbol across translation units.
SymbolSlab mergeSymbols(tooling::ToolResults *Results) {
  SymbolSlab::Builder UniqueSymbols;
  llvm::BumpPtrAllocator Arena;
  Symbol::Details Scratch;
  Results->forEachResult([&](llvm::StringRef Key, llvm::StringRef Value) {
    Arena.Reset();
    auto Sym = clang::clangd::SymbolFromYAML(Value, Arena);
    clang::clangd::SymbolID ID;
    Key >> ID;
    if (const auto *Existing = UniqueSymbols.find(ID))
      UniqueSymbols.insert(mergeSymbol(*Existing, Sym, &Scratch));
    else
      UniqueSymbols.insert(Sym);
  });
  return std::move(UniqueSymbols).build();
}

} // namespace
} // namespace clangd
} // namespace clang

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  const char* Overview =
      "This is an **experimental** tool to generate YAML-format "
      "project-wide symbols for clangd (global code completion). It would be "
      "changed and deprecated eventually. Don't use it in production code!";
  auto Executor = clang::tooling::createExecutorFromCommandLineArgs(
      argc, argv, cl::GeneralCategory, Overview);

  if (!Executor) {
    llvm::errs() << llvm::toString(Executor.takeError()) << "\n";
    return 1;
  }

  if (!clang::clangd::AssumedHeaderDir.empty() &&
      !llvm::sys::path::is_absolute(clang::clangd::AssumedHeaderDir)) {
    llvm::errs() << "--assume-header-dir must be an absolute path.\n";
    return 1;
  }

  // Map phase: emit symbols found in each translation unit.
  auto Err = Executor->get()->execute(
      llvm::make_unique<clang::clangd::SymbolIndexActionFactory>(
          Executor->get()->getExecutionContext()));
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
  }

  // Reduce phase: combine symbols using the ID as a key.
  auto UniqueSymbols =
      clang::clangd::mergeSymbols(Executor->get()->getToolResults());

  // Output phase: emit YAML for result symbols.
  SymbolsToYAML(UniqueSymbols, llvm::outs());
  return 0;
}
