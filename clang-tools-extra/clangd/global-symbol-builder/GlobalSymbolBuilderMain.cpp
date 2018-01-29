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

static llvm::cl::opt<std::string> AssumedHeaderDir(
    "assume-header-dir",
    llvm::cl::desc("The index includes header that a symbol is defined in. "
                   "If the absolute path cannot be determined (e.g. an "
                   "in-memory VFS) then the relative path is resolved against "
                   "this directory, which must be absolute. If this flag is "
                   "not given, such headers will have relative paths."));

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

  auto Err = Executor->get()->execute(
      llvm::make_unique<clang::clangd::SymbolIndexActionFactory>(
          Executor->get()->getExecutionContext()));
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
  }

  // Deduplicate the result by key and keep the longest value.
  // FIXME(ioeric): Merge occurrences, rather than just dropping all but one.
  // Definitions and forward declarations have the same key and may both have
  // information. Usage count will need to be aggregated across occurrences,
  // too.
  llvm::StringMap<llvm::StringRef> UniqueSymbols;
  Executor->get()->getToolResults()->forEachResult(
      [&UniqueSymbols](llvm::StringRef Key, llvm::StringRef Value) {
        auto Ret = UniqueSymbols.try_emplace(Key, Value);
        if (!Ret.second) {
          // If key already exists, keep the longest value.
          llvm::StringRef &V = Ret.first->second;
          V = V.size() < Value.size() ? Value : V;
        }
      });
  for (const auto &Sym : UniqueSymbols)
    llvm::outs() << Sym.second;
  return 0;
}
