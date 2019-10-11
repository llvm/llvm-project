//===--- SourceCodeBuilder.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/SourceCodeBuilders.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "llvm/ADT/Twine.h"
#include <string>

using namespace clang;
using namespace tooling;

const Expr *tooling::reallyIgnoreImplicit(const Expr &E) {
  const Expr *Expr = E.IgnoreImplicit();
  if (const auto *CE = dyn_cast<CXXConstructExpr>(Expr)) {
    if (CE->getNumArgs() > 0 &&
        CE->getArg(0)->getSourceRange() == Expr->getSourceRange())
      return CE->getArg(0)->IgnoreImplicit();
  }
  return Expr;
}

bool tooling::mayEverNeedParens(const Expr &E) {
  const Expr *Expr = reallyIgnoreImplicit(E);
  // We always want parens around unary, binary, and ternary operators, because
  // they are lower precedence.
  if (isa<UnaryOperator>(Expr) || isa<BinaryOperator>(Expr) ||
      isa<AbstractConditionalOperator>(Expr))
    return true;

  // We need parens around calls to all overloaded operators except: function
  // calls, subscripts, and expressions that are already part of an (implicit)
  // call to operator->. These latter are all in the same precedence level as
  // dot/arrow and that level is left associative, so they don't need parens
  // when appearing on the left.
  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(Expr))
    return Op->getOperator() != OO_Call && Op->getOperator() != OO_Subscript &&
           Op->getOperator() != OO_Arrow;

  return false;
}

bool tooling::needParensAfterUnaryOperator(const Expr &E) {
  const Expr *Expr = reallyIgnoreImplicit(E);
  if (isa<BinaryOperator>(Expr) || isa<AbstractConditionalOperator>(Expr))
    return true;

  if (const auto *Op = dyn_cast<CXXOperatorCallExpr>(Expr))
    return Op->getNumArgs() == 2 && Op->getOperator() != OO_PlusPlus &&
           Op->getOperator() != OO_MinusMinus && Op->getOperator() != OO_Call &&
           Op->getOperator() != OO_Subscript;

  return false;
}

llvm::Optional<std::string> tooling::buildParens(const Expr &E,
                                                 const ASTContext &Context) {
  StringRef Text = getText(E, Context);
  if (Text.empty())
    return llvm::None;
  if (mayEverNeedParens(E))
    return ("(" + Text + ")").str();
  return Text.str();
}

llvm::Optional<std::string>
tooling::buildDereference(const Expr &E, const ASTContext &Context) {
  if (const auto *Op = dyn_cast<UnaryOperator>(&E))
    if (Op->getOpcode() == UO_AddrOf) {
      // Strip leading '&'.
      StringRef Text =
          getText(*Op->getSubExpr()->IgnoreParenImpCasts(), Context);
      if (Text.empty())
        return llvm::None;
      return Text.str();
    }

  StringRef Text = getText(E, Context);
  if (Text.empty())
    return llvm::None;
  // Add leading '*'.
  if (needParensAfterUnaryOperator(E))
    return ("*(" + Text + ")").str();
  return ("*" + Text).str();
}

llvm::Optional<std::string> tooling::buildAddressOf(const Expr &E,
                                                    const ASTContext &Context) {
  if (const auto *Op = dyn_cast<UnaryOperator>(&E))
    if (Op->getOpcode() == UO_Deref) {
      // Strip leading '*'.
      StringRef Text =
          getText(*Op->getSubExpr()->IgnoreParenImpCasts(), Context);
      if (Text.empty())
        return llvm::None;
      return Text.str();
    }
  // Add leading '&'.
  StringRef Text = getText(E, Context);
  if (Text.empty())
    return llvm::None;
  if (needParensAfterUnaryOperator(E)) {
    return ("&(" + Text + ")").str();
  }
  return ("&" + Text).str();
}

llvm::Optional<std::string> tooling::buildDot(const Expr &E,
                                              const ASTContext &Context) {
  if (const auto *Op = llvm::dyn_cast<UnaryOperator>(&E))
    if (Op->getOpcode() == UO_Deref) {
      // Strip leading '*', add following '->'.
      const Expr *SubExpr = Op->getSubExpr()->IgnoreParenImpCasts();
      StringRef DerefText = getText(*SubExpr, Context);
      if (DerefText.empty())
        return llvm::None;
      if (needParensBeforeDotOrArrow(*SubExpr))
        return ("(" + DerefText + ")->").str();
      return (DerefText + "->").str();
    }

  // Add following '.'.
  StringRef Text = getText(E, Context);
  if (Text.empty())
    return llvm::None;
  if (needParensBeforeDotOrArrow(E)) {
    return ("(" + Text + ").").str();
  }
  return (Text + ".").str();
}

llvm::Optional<std::string> tooling::buildArrow(const Expr &E,
                                                const ASTContext &Context) {
  if (const auto *Op = llvm::dyn_cast<UnaryOperator>(&E))
    if (Op->getOpcode() == UO_AddrOf) {
      // Strip leading '&', add following '.'.
      const Expr *SubExpr = Op->getSubExpr()->IgnoreParenImpCasts();
      StringRef DerefText = getText(*SubExpr, Context);
      if (DerefText.empty())
        return llvm::None;
      if (needParensBeforeDotOrArrow(*SubExpr))
        return ("(" + DerefText + ").").str();
      return (DerefText + ".").str();
    }

  // Add following '->'.
  StringRef Text = getText(E, Context);
  if (Text.empty())
    return llvm::None;
  if (needParensBeforeDotOrArrow(E))
    return ("(" + Text + ")->").str();
  return (Text + "->").str();
}
