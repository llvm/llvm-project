//===-- FindAllSymbolsAction.h - find all symbols action --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_SYMBOLS_ACTION_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_SYMBOLS_ACTION_H

#include "FindAllSymbols.h"
#include "HeaderMapCollector.h"
#include "PragmaCommentHandler.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {
namespace find_all_symbols {

class FindAllSymbolsAction : public clang::ASTFrontendAction {
public:
  explicit FindAllSymbolsAction(
      SymbolReporter *Reporter,
      const HeaderMapCollector::RegexHeaderMap *RegexHeaderMap = nullptr);

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
      const HeaderMapCollector::RegexHeaderMap *RegexHeaderMap = nullptr)
      : Reporter(Reporter), RegexHeaderMap(RegexHeaderMap) {}

  clang::FrontendAction *create() override {
    return new FindAllSymbolsAction(Reporter, RegexHeaderMap);
  }

private:
  SymbolReporter *const Reporter;
  const HeaderMapCollector::RegexHeaderMap *const RegexHeaderMap;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_SYMBOLS_ACTION_H
