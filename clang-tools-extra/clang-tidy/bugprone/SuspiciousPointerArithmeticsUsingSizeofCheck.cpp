//===--- SuspiciousPointerArithmeticsUsingSizeofCheck.cpp - clang-tidy ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousPointerArithmeticsUsingSizeofCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

constexpr llvm::StringLiteral BinOp{"bin-op"};
constexpr llvm::StringLiteral PointedType{"pointed-type"};
constexpr llvm::StringLiteral ScaleExpr{"scale-expr"};

const auto IgnoredTypes = qualType(
    anyOf(asString("char"), asString("unsigned char"), asString("signed char"),
          asString("int8_t"), asString("uint8_t"), asString("std::byte"),
          asString("const char"), asString("const unsigned char"),
          asString("const signed char"), asString("const int8_t"),
          asString("const uint8_t"), asString("const std::byte")));
const auto InterestingPointer = pointerType(unless(pointee(IgnoredTypes)));

const auto ScaledIntegerTraitExprs = /*stmt(anyOf(*/ sizeOfExpr(expr()) /*))*/;

CharUnits getSizeOfType(const ASTContext &Ctx, const Type *Ty) {
  if (!Ty || Ty->isIncompleteType() || Ty->isDependentType() ||
      isa<DependentSizedArrayType>(Ty) || !Ty->isConstantSizeType())
    return CharUnits::Zero();
  return Ctx.getTypeSizeInChars(Ty);
}

} // namespace

SuspiciousPointerArithmeticsUsingSizeofCheck::
    SuspiciousPointerArithmeticsUsingSizeofCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void SuspiciousPointerArithmeticsUsingSizeofCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      expr(anyOf(binaryOperator(hasAnyOperatorName("+", "+=", "-", "-="),
                                hasLHS(hasType(pointerType(
                                    pointee(qualType().bind(PointedType))))),
                                hasRHS(ScaledIntegerTraitExprs.bind(ScaleExpr)))
                     .bind(BinOp),
                 binaryOperator(hasAnyOperatorName("+", "-"),
                                hasRHS(hasType(pointerType(
                                    pointee(qualType().bind(PointedType))))),
                                hasLHS(ScaledIntegerTraitExprs.bind(ScaleExpr)))
                     .bind(BinOp))),
      this);
}

void SuspiciousPointerArithmeticsUsingSizeofCheck::check(
    const MatchFinder::MatchResult &Result) {
  const ASTContext &Ctx = *Result.Context;
  const auto *BO = Result.Nodes.getNodeAs<BinaryOperator>(BinOp);
  const auto *QT = Result.Nodes.getNodeAs<QualType>(PointedType);
  assert(BO && QT && "Broken matchers encountered.");

  const auto Size = getSizeOfType(Ctx, QT->getTypePtr()).getQuantity();
  if (Size == 1)
    return;

  diag(BO->getExprLoc(),
       "pointer arithmetic using a number scaled by '%0'; this value will be "
       "scaled again by the '%1' operator")
      << "sizeof" << BO->getOpcodeStr();
  diag(BO->getExprLoc(), "'%0' scales with '%1(%2)' == %3", DiagnosticIDs::Note)
      << BO->getOpcodeStr() << "sizeof" << "?" << Size;
}

} // namespace clang::tidy::bugprone
