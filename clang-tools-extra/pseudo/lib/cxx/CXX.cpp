//===--- CXX.cpp - Define public interfaces for C++ grammar ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/cxx/CXX.h"
#include "clang-pseudo/Forest.h"
#include "clang-pseudo/Language.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"
#include <utility>

namespace clang {
namespace pseudo {
namespace cxx {
namespace {
static const char *CXXBNF =
#include "CXXBNF.inc"
    ;

bool guardOverride(llvm::ArrayRef<const ForestNode *> RHS,
                   const TokenStream &Tokens) {
  assert(RHS.size() == 1 &&
         RHS.front()->symbol() == tokenSymbol(clang::tok::identifier));
  return Tokens.tokens()[RHS.front()->startTokenIndex()].text() == "override";
}
bool guardFinal(llvm::ArrayRef<const ForestNode *> RHS,
                const TokenStream &Tokens) {
  assert(RHS.size() == 1 &&
         RHS.front()->symbol() == tokenSymbol(clang::tok::identifier));
  return Tokens.tokens()[RHS.front()->startTokenIndex()].text() == "final";
}

llvm::DenseMap<ExtensionID, RuleGuard> buildGuards() {
  return {
      {(ExtensionID)Extension::Override, guardOverride},
      {(ExtensionID)Extension::Final, guardFinal},
  };
}

Token::Index recoverBrackets(Token::Index Begin, const TokenStream &Tokens) {
  assert(Begin > 0);
  const Token &Left = Tokens.tokens()[Begin - 1];
  assert(Left.Kind == tok::l_brace || Left.Kind == tok::l_paren ||
         Left.Kind == tok::l_square);
  if (const Token *Right = Left.pair()) {
    assert(Tokens.index(*Right) > Begin);
    return Tokens.index(*Right);
  }
  return Token::Invalid;
}

llvm::DenseMap<ExtensionID, RecoveryStrategy> buildRecoveryStrategies() {
  return {
      {(ExtensionID)Extension::Brackets, recoverBrackets},
  };
}

} // namespace

const Language &getLanguage() {
  static const auto &CXXLanguage = []() -> const Language & {
    std::vector<std::string> Diags;
    auto G = Grammar::parseBNF(CXXBNF, Diags);
    assert(Diags.empty());
    LRTable Table = LRTable::buildSLR(G);
    const Language *PL = new Language{
        std::move(G),
        std::move(Table),
        buildGuards(),
        buildRecoveryStrategies(),
    };
    return *PL;
  }();
  return CXXLanguage;
}

} // namespace cxx
} // namespace pseudo
} // namespace clang
