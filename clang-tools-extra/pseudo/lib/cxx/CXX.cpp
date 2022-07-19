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
bool guardModule(llvm::ArrayRef<const ForestNode *> RHS,
                 const TokenStream &Tokens) {
  return Tokens.tokens()[RHS.front()->startTokenIndex()].text() == "module";
}
bool guardImport(llvm::ArrayRef<const ForestNode *> RHS,
                 const TokenStream &Tokens) {
  return Tokens.tokens()[RHS.front()->startTokenIndex()].text() == "import";
}
bool guardExport(llvm::ArrayRef<const ForestNode *> RHS,
                 const TokenStream &Tokens) {
  return Tokens.tokens()[RHS.front()->startTokenIndex()].text() == "export";
}

bool isFunctionDeclarator(const ForestNode *Declarator) {
  assert(Declarator->symbol() == (SymbolID)(cxx::Symbol::declarator));
  bool IsFunction = false;
  using cxx::Rule;
  while (true) {
    // not well-formed code, return the best guess.
    if (Declarator->kind() != ForestNode::Sequence)
      return IsFunction;

    switch ((cxx::Rule)Declarator->rule()) {
    case Rule::noptr_declarator_0declarator_id: // reached the bottom
      return IsFunction;
    // *X is a nonfunction (unless X is a function).
    case Rule::ptr_declarator_0ptr_operator_1ptr_declarator:
      Declarator = Declarator->elements()[1];
      IsFunction = false;
      continue;
    // X() is a function (unless X is a pointer or similar).
    case Rule::
        declarator_0noptr_declarator_1parameters_and_qualifiers_2trailing_return_type:
    case Rule::noptr_declarator_0noptr_declarator_1parameters_and_qualifiers:
      Declarator = Declarator->elements()[0];
      IsFunction = true;
      continue;
    // X[] is an array (unless X is a pointer or function).
    case Rule::
        noptr_declarator_0noptr_declarator_1l_square_2constant_expression_3r_square:
    case Rule::noptr_declarator_0noptr_declarator_1l_square_2r_square:
      Declarator = Declarator->elements()[0];
      IsFunction = false;
      continue;
    // (X) is whatever X is.
    case Rule::noptr_declarator_0l_paren_1ptr_declarator_2r_paren:
      Declarator = Declarator->elements()[1];
      continue;
    case Rule::ptr_declarator_0noptr_declarator:
    case Rule::declarator_0ptr_declarator:
      Declarator = Declarator->elements()[0];
      continue;

    default:
      assert(false && "unhandled declarator for IsFunction");
      return IsFunction;
    }
  }
  llvm_unreachable("unreachable");
}
bool guardFunction(llvm::ArrayRef<const ForestNode *> RHS,
                   const TokenStream &Tokens) {
  assert(RHS.size() == 1 &&
         RHS.front()->symbol() == (SymbolID)(cxx::Symbol::declarator));
  return isFunctionDeclarator(RHS.front());
}
bool guardNonFunction(llvm::ArrayRef<const ForestNode *> RHS,
                      const TokenStream &Tokens) {
  assert(RHS.size() == 1 &&
         RHS.front()->symbol() == (SymbolID)(cxx::Symbol::declarator));
  return !isFunctionDeclarator(RHS.front());
}

llvm::DenseMap<ExtensionID, RuleGuard> buildGuards() {
  return {
      {(ExtensionID)Extension::Override, guardOverride},
      {(ExtensionID)Extension::Final, guardFinal},
      {(ExtensionID)Extension::Import, guardImport},
      {(ExtensionID)Extension::Export, guardExport},
      {(ExtensionID)Extension::Module, guardModule},
      {(ExtensionID)Extension::FunctionDeclarator, guardFunction},
      {(ExtensionID)Extension::NonFunctionDeclarator, guardNonFunction},
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
