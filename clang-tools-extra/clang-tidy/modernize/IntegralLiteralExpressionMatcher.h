//===--- IntegralLiteralExpressionMatcher.h - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_INTEGRAL_LITERAL_EXPRESSION_MATCHER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_INTEGRAL_LITERAL_EXPRESSION_MATCHER_H

#include <clang/Lex/Token.h>
#include <llvm/ADT/ArrayRef.h>

namespace clang::tidy::modernize {

enum class LiteralSize {
  Unknown = 0,
  Int,
  UnsignedInt,
  Long,
  UnsignedLong,
  LongLong,
  UnsignedLongLong
};

// Parses an array of tokens and returns true if they conform to the rules of
// C++ for whole expressions involving integral literals.  Follows the operator
// precedence rules of C++.  Optionally exclude comma operator expressions.
class IntegralLiteralExpressionMatcher {
public:
  IntegralLiteralExpressionMatcher(ArrayRef<Token> Tokens, bool CommaAllowed)
      : Current(Tokens.begin()), End(Tokens.end()), CommaAllowed(CommaAllowed) {
  }

  bool match();
  LiteralSize largestLiteralSize() const;

private:
  bool advance();
  bool consume(tok::TokenKind Kind);
  bool nonTerminalChainedExpr(
      bool (IntegralLiteralExpressionMatcher::*NonTerminal)(),
      const std::function<bool(Token)> &IsKind);
  template <tok::TokenKind Kind>
  bool nonTerminalChainedExpr(
      bool (IntegralLiteralExpressionMatcher::*NonTerminal)()) {
    return nonTerminalChainedExpr(NonTerminal,
                                  [](Token Tok) { return Tok.is(Kind); });
  }
  template <tok::TokenKind K1, tok::TokenKind K2, tok::TokenKind... Ks>
  bool nonTerminalChainedExpr(
      bool (IntegralLiteralExpressionMatcher::*NonTerminal)()) {
    return nonTerminalChainedExpr(
        NonTerminal, [](Token Tok) { return Tok.isOneOf(K1, K2, Ks...); });
  }

  bool unaryOperator();
  bool unaryExpr();
  bool multiplicativeExpr();
  bool additiveExpr();
  bool shiftExpr();
  bool compareExpr();
  bool relationalExpr();
  bool equalityExpr();
  bool andExpr();
  bool exclusiveOrExpr();
  bool inclusiveOrExpr();
  bool logicalAndExpr();
  bool logicalOrExpr();
  bool conditionalExpr();
  bool commaExpr();
  bool expr();

  ArrayRef<Token>::iterator Current;
  ArrayRef<Token>::iterator End;
  LiteralSize LargestSize{LiteralSize::Unknown};
  bool CommaAllowed;
};

} // namespace clang::tidy::modernize

#endif
