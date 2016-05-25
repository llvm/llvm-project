//===-- FindAllSymbolsAction.h - find all symbols action --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_SYMBOLS_ACTION_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_SYMBOLS_ACTION_H

#include "FindAllMacros.h"
#include "FindAllSymbols.h"
#include "HeaderMapCollector.h"
#include "PragmaCommentHandler.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace find_all_symbols {

class FindAllSymbolsAction : public clang::ASTFrontendAction {
public:
  explicit FindAllSymbolsAction(
      SymbolReporter *Reporter,
      const HeaderMapCollector::HeaderMap *PostfixMap = nullptr);

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    StringRef InFile) override;

private:
  SymbolReporter *const Reporter;
  clang::ast_matchers::MatchFinder MatchFinder;
  HeaderMapCollector Collector;
  PragmaCommentHandler Handler;
  FindAllSymbols Matcher;
};

class FindAllSymbolsActionFactory : public tooling::FrontendActionFactory {
public:
  FindAllSymbolsActionFactory(
      SymbolReporter *Reporter,
      const HeaderMapCollector::HeaderMap *PostfixMap = nullptr)
      : Reporter(Reporter), PostfixMap(PostfixMap) {}

  virtual clang::FrontendAction *create() override {
    return new FindAllSymbolsAction(Reporter, PostfixMap);
  }

private:
  SymbolReporter *const Reporter;
  const HeaderMapCollector::HeaderMap *const PostfixMap;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_SYMBOLS_ACTION_H
