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
#include "index/SymbolCollector.h"
#include "index/SymbolYAML.h"
#include "clang/Frontend/FrontendActions.h"
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
    return new WrappedIndexAction(
        std::make_shared<SymbolCollector>(SymbolCollector::Options()),
        IndexOpts, Ctx);
  }

  tooling::ExecutionContext *Ctx;
};

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

  auto Err = Executor->get()->execute(
      llvm::make_unique<clang::clangd::SymbolIndexActionFactory>(
          Executor->get()->getExecutionContext()));
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
  }

  Executor->get()->getToolResults()->forEachResult(
      [](llvm::StringRef, llvm::StringRef Value) { llvm::outs() << Value; });
  return 0;
}
