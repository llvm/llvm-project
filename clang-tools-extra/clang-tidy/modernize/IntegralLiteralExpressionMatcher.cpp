//===--- IntegralLiteralExpressionMatcher.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntegralLiteralExpressionMatcher.h"

#include <cctype>
#include <stdexcept>

namespace clang {
namespace tidy {
namespace modernize {

// Validate that this literal token is a valid integer literal.  A literal token
// could be a floating-point token, which isn't acceptable as a value for an
// enumeration.  A floating-point token must either have a decimal point or an
// exponent ('E' or 'P').
static bool isIntegralConstant(const Token &Token) {
  const char *Begin = Token.getLiteralData();
  const char *End = Begin + Token.getLength();

  // Not a hexadecimal floating-point literal.
  if (Token.getLength() > 2 && Begin[0] == '0' && std::toupper(Begin[1]) == 'X')
    return std::none_of(Begin + 2, End, [](char C) {
      return C == '.' || std::toupper(C) == 'P';
    });

  // Not a decimal floating-point literal or complex literal.
  return std::none_of(Begin, End, [](char C) {
    return C == '.' || std::toupper(C) == 'E' || std::toupper(C) == 'I';
  });
}

bool IntegralLiteralExpressionMatcher::advance() {
  ++Current;
  return Current != End;
}

bool IntegralLiteralExpressionMatcher::consume(tok::TokenKind Kind) {
  if (Current->is(Kind)) {
    ++Current;
    return true;
  }

  return false;
}

bool IntegralLiteralExpressionMatcher::nonTerminalChainedExpr(
    bool (IntegralLiteralExpressionMatcher::*NonTerminal)(),
    const std::function<bool(Token)> &IsKind) {
  if (!(this->*NonTerminal)())
    return false;
  if (Current == End)
    return true;

  while (Current != End) {
    if (!IsKind(*Current))
      break;

    if (!advance())
      return false;

    if (!(this->*NonTerminal)())
      return false;
  }

  return true;
}

// Advance over unary operators.
bool IntegralLiteralExpressionMatcher::unaryOperator() {
  if (Current->isOneOf(tok::TokenKind::minus, tok::TokenKind::plus,
                       tok::TokenKind::tilde, tok::TokenKind::exclaim)) {
    return advance();
  }

  return true;
}

bool IntegralLiteralExpressionMatcher::unaryExpr() {
  if (!unaryOperator())
    return false;

  if (consume(tok::TokenKind::l_paren)) {
    if (Current == End)
      return false;

    if (!expr())
      return false;

    if (Current == End)
      return false;

    return consume(tok::TokenKind::r_paren);
  }

  if (!Current->isLiteral() || isStringLiteral(Current->getKind()) ||
      !isIntegralConstant(*Current)) {
    return false;
  }
  ++Current;
  return true;
}

bool IntegralLiteralExpressionMatcher::multiplicativeExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::star, tok::TokenKind::slash,
                                tok::TokenKind::percent>(
      &IntegralLiteralExpressionMatcher::unaryExpr);
}

bool IntegralLiteralExpressionMatcher::additiveExpr() {
  return nonTerminalChainedExpr<tok::plus, tok::minus>(
      &IntegralLiteralExpressionMatcher::multiplicativeExpr);
}

bool IntegralLiteralExpressionMatcher::shiftExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::lessless,
                                tok::TokenKind::greatergreater>(
      &IntegralLiteralExpressionMatcher::additiveExpr);
}

bool IntegralLiteralExpressionMatcher::compareExpr() {
  if (!shiftExpr())
    return false;
  if (Current == End)
    return true;

  if (Current->is(tok::TokenKind::spaceship)) {
    if (!advance())
      return false;

    if (!shiftExpr())
      return false;
  }

  return true;
}

bool IntegralLiteralExpressionMatcher::relationalExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::less, tok::TokenKind::greater,
                                tok::TokenKind::lessequal,
                                tok::TokenKind::greaterequal>(
      &IntegralLiteralExpressionMatcher::compareExpr);
}

bool IntegralLiteralExpressionMatcher::equalityExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::equalequal,
                                tok::TokenKind::exclaimequal>(
      &IntegralLiteralExpressionMatcher::relationalExpr);
}

bool IntegralLiteralExpressionMatcher::andExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::amp>(
      &IntegralLiteralExpressionMatcher::equalityExpr);
}

bool IntegralLiteralExpressionMatcher::exclusiveOrExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::caret>(
      &IntegralLiteralExpressionMatcher::andExpr);
}

bool IntegralLiteralExpressionMatcher::inclusiveOrExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::pipe>(
      &IntegralLiteralExpressionMatcher::exclusiveOrExpr);
}

bool IntegralLiteralExpressionMatcher::logicalAndExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::ampamp>(
      &IntegralLiteralExpressionMatcher::inclusiveOrExpr);
}

bool IntegralLiteralExpressionMatcher::logicalOrExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::pipepipe>(
      &IntegralLiteralExpressionMatcher::logicalAndExpr);
}

bool IntegralLiteralExpressionMatcher::conditionalExpr() {
  if (!logicalOrExpr())
    return false;
  if (Current == End)
    return true;

  if (Current->is(tok::TokenKind::question)) {
    if (!advance())
      return false;

    // A gcc extension allows x ? : y as a synonym for x ? x : y.
    if (Current->is(tok::TokenKind::colon)) {
      if (!advance())
        return false;

      if (!expr())
        return false;

      return true;
    }

    if (!expr())
      return false;
    if (Current == End)
      return false;

    if (!Current->is(tok::TokenKind::colon))
      return false;

    if (!advance())
      return false;

    if (!expr())
      return false;
  }
  return true;
}

bool IntegralLiteralExpressionMatcher::commaExpr() {
  return nonTerminalChainedExpr<tok::TokenKind::comma>(
      &IntegralLiteralExpressionMatcher::conditionalExpr);
}

bool IntegralLiteralExpressionMatcher::expr() { return commaExpr(); }

bool IntegralLiteralExpressionMatcher::match() {
  return expr() && Current == End;
}

} // namespace modernize
} // namespace tidy
} // namespace clang
