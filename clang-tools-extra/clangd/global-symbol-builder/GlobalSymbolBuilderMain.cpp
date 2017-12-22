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
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;
using namespace clang::tooling;
using clang::clangd::SymbolSlab;

namespace clang {
namespace clangd {

class SymbolIndexActionFactory : public tooling::FrontendActionFactory {
public:
  SymbolIndexActionFactory() = default;

  clang::FrontendAction *create() override {
    index::IndexingOptions IndexOpts;
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::All;
    IndexOpts.IndexFunctionLocals = false;
    Collector = std::make_shared<SymbolCollector>();
    return index::createIndexingAction(Collector, IndexOpts, nullptr).release();
  }

  std::shared_ptr<SymbolCollector> Collector;
};

} // namespace clangd
} // namespace clang

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  const char* Overview =
      "This is an **experimental** tool to generate YAML-format "
      "project-wide symbols for clangd (global code completion). It would be "
      "changed and deprecated eventually. Don't use it in production code!";
  CommonOptionsParser OptionsParser(argc, argv, cl::GeneralCategory,
                                    /*Overview=*/Overview);

  // No compilation database found, fallback to single TU analysis, this is
  // mainly for debugging purpose:
  //   global-symbol-buidler /tmp/t.cc -- -std=c++11.
  if (OptionsParser.getCompilations().getAllFiles().empty()) {
    llvm::errs() << "No compilation database found, processing individual "
                    "files with flags from command-line\n.";
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());
    clang::clangd::SymbolIndexActionFactory IndexAction;
    Tool.run(&IndexAction);
    llvm::outs() << SymbolToYAML(IndexAction.Collector->takeSymbols());
    return 0;
  }

  // Found compilation database, we iterate all TUs from database to get all
  // symbols, and then merge them into a single SymbolSlab.
  SymbolSlab GlobalSymbols;
  std::mutex SymbolMutex;
  auto AddSymbols = [&](const SymbolSlab& NewSymbols) {
    // Synchronize set accesses.
    std::unique_lock<std::mutex> LockGuard(SymbolMutex);
    for (auto It : NewSymbols) {
      // FIXME: Better handling the overlap symbols, currently we overwrite it
      // with the latest one, but we always want to good declarations (class
      // definitions, instead of forward declarations).
      GlobalSymbols.insert(It.second);
    }
  };

  {
    llvm::ThreadPool Pool;
    for (auto& file : OptionsParser.getCompilations().getAllFiles()) {
      Pool.async([&OptionsParser, &AddSymbols](llvm::StringRef Path) {
        ClangTool Tool(OptionsParser.getCompilations(), {Path});
        clang::clangd::SymbolIndexActionFactory IndexAction;
        Tool.run(&IndexAction);
        AddSymbols(IndexAction.Collector->takeSymbols());
      }, file);
    }
  }

  llvm::outs() << SymbolToYAML(GlobalSymbols);
  return 0;
}
