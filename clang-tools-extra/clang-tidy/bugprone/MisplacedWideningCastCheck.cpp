//===--- MisplacedWideningCastCheck.cpp - clang-tidy-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MisplacedWideningCastCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

MisplacedWideningCastCheck::MisplacedWideningCastCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckImplicitCasts(Options.get("CheckImplicitCasts", false)) {}

void MisplacedWideningCastCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckImplicitCasts", CheckImplicitCasts);
}

void MisplacedWideningCastCheck::registerMatchers(MatchFinder *Finder) {
  const auto Calc =
      expr(anyOf(binaryOperator(hasAnyOperatorName("+", "-", "*", "<<")),
                 unaryOperator(hasOperatorName("~"))),
           hasType(isInteger()))
          .bind("Calc");

  const auto ExplicitCast = explicitCastExpr(hasDestinationType(isInteger()),
                                             has(ignoringParenImpCasts(Calc)));
  const auto ImplicitCast =
      implicitCastExpr(hasImplicitDestinationType(isInteger()),
                       has(ignoringParenImpCasts(Calc)));
  const auto Cast =
      traverse(TK_AsIs, expr(anyOf(ExplicitCast, ImplicitCast)).bind("Cast"));

  Finder->addMatcher(varDecl(hasInitializer(Cast)), this);
  Finder->addMatcher(returnStmt(hasReturnValue(Cast)), this);
  Finder->addMatcher(callExpr(hasAnyArgument(Cast)), this);
  Finder->addMatcher(binaryOperator(hasOperatorName("="), hasRHS(Cast)), this);
  Finder->addMatcher(
      binaryOperator(isComparisonOperator(), hasEitherOperand(Cast)), this);
}

static unsigned getMaxCalculationWidth(const ASTContext &Context,
                                       const Expr *E) {
  E = E->IgnoreParenImpCasts();

  if (const auto *Bop = dyn_cast<BinaryOperator>(E)) {
    unsigned LHSWidth = getMaxCalculationWidth(Context, Bop->getLHS());
    unsigned RHSWidth = getMaxCalculationWidth(Context, Bop->getRHS());
    if (Bop->getOpcode() == BO_Mul)
      return LHSWidth + RHSWidth;
    if (Bop->getOpcode() == BO_Add)
      return std::max(LHSWidth, RHSWidth) + 1;
    if (Bop->getOpcode() == BO_Rem) {
      Expr::EvalResult Result;
      if (Bop->getRHS()->EvaluateAsInt(Result, Context))
        return Result.Val.getInt().getActiveBits();
    } else if (Bop->getOpcode() == BO_Shl) {
      Expr::EvalResult Result;
      if (Bop->getRHS()->EvaluateAsInt(Result, Context)) {
        // We don't handle negative values and large values well. It is assumed
        // that compiler warnings are written for such values so the user will
        // fix that.
        return LHSWidth + Result.Val.getInt().getExtValue();
      }

      // Unknown bitcount, assume there is truncation.
      return 1024U;
    }
  } else if (const auto *Uop = dyn_cast<UnaryOperator>(E)) {
    // There is truncation when ~ is used.
    if (Uop->getOpcode() == UO_Not)
      return 1024U;

    QualType T = Uop->getType();
    return T->isIntegerType() ? Context.getIntWidth(T) : 1024U;
  } else if (const auto *I = dyn_cast<IntegerLiteral>(E)) {
    return I->getValue().getActiveBits();
  }

  return Context.getIntWidth(E->getType());
}

static int relativeIntSizes(BuiltinType::Kind Kind) {
  switch (Kind) {
  case BuiltinType::UChar:
    return 1;
  case BuiltinType::SChar:
    return 1;
  case BuiltinType::Char_U:
    return 1;
  case BuiltinType::Char_S:
    return 1;
  case BuiltinType::UShort:
    return 2;
  case BuiltinType::Short:
    return 2;
  case BuiltinType::UInt:
    return 3;
  case BuiltinType::Int:
    return 3;
  case BuiltinType::ULong:
    return 4;
  case BuiltinType::Long:
    return 4;
  case BuiltinType::ULongLong:
    return 5;
  case BuiltinType::LongLong:
    return 5;
  case BuiltinType::UInt128:
    return 6;
  case BuiltinType::Int128:
    return 6;
  default:
    return 0;
  }
}

static int relativeCharSizes(BuiltinType::Kind Kind) {
  switch (Kind) {
  case BuiltinType::UChar:
    return 1;
  case BuiltinType::SChar:
    return 1;
  case BuiltinType::Char_U:
    return 1;
  case BuiltinType::Char_S:
    return 1;
  case BuiltinType::Char16:
    return 2;
  case BuiltinType::Char32:
    return 3;
  default:
    return 0;
  }
}

static int relativeCharSizesW(BuiltinType::Kind Kind) {
  switch (Kind) {
  case BuiltinType::UChar:
    return 1;
  case BuiltinType::SChar:
    return 1;
  case BuiltinType::Char_U:
    return 1;
  case BuiltinType::Char_S:
    return 1;
  case BuiltinType::WChar_U:
    return 2;
  case BuiltinType::WChar_S:
    return 2;
  default:
    return 0;
  }
}

static bool isFirstWider(BuiltinType::Kind First, BuiltinType::Kind Second) {
  int FirstSize = 0, SecondSize = 0;
  if ((FirstSize = relativeIntSizes(First)) != 0 &&
      (SecondSize = relativeIntSizes(Second)) != 0)
    return FirstSize > SecondSize;
  if ((FirstSize = relativeCharSizes(First)) != 0 &&
      (SecondSize = relativeCharSizes(Second)) != 0)
    return FirstSize > SecondSize;
  if ((FirstSize = relativeCharSizesW(First)) != 0 &&
      (SecondSize = relativeCharSizesW(Second)) != 0)
    return FirstSize > SecondSize;
  return false;
}

void MisplacedWideningCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<CastExpr>("Cast");
  if (!CheckImplicitCasts && isa<ImplicitCastExpr>(Cast))
    return;
  if (Cast->getBeginLoc().isMacroID())
    return;

  const auto *Calc = Result.Nodes.getNodeAs<Expr>("Calc");
  if (Calc->getBeginLoc().isMacroID())
    return;

  if (Cast->isTypeDependent() || Cast->isValueDependent() ||
      Calc->isTypeDependent() || Calc->isValueDependent())
    return;

  ASTContext &Context = *Result.Context;

  QualType CastType = Cast->getType();
  QualType CalcType = Calc->getType();

  // Explicit truncation using cast.
  if (Context.getIntWidth(CastType) < Context.getIntWidth(CalcType))
    return;

  // If CalcType and CastType have same size then there is no real danger, but
  // there can be a portability problem.

  if (Context.getIntWidth(CastType) == Context.getIntWidth(CalcType)) {
    const auto *CastBuiltinType =
        dyn_cast<BuiltinType>(CastType->getUnqualifiedDesugaredType());
    const auto *CalcBuiltinType =
        dyn_cast<BuiltinType>(CalcType->getUnqualifiedDesugaredType());
    if (!CastBuiltinType || !CalcBuiltinType)
      return;
    if (!isFirstWider(CastBuiltinType->getKind(), CalcBuiltinType->getKind()))
      return;
  }

  // Don't write a warning if we can easily see that the result is not
  // truncated.
  if (Context.getIntWidth(CalcType) >= getMaxCalculationWidth(Context, Calc))
    return;

  diag(Cast->getBeginLoc(), "either cast from %0 to %1 is ineffective, or "
                            "there is loss of precision before the conversion")
      << CalcType << CastType;
}

} // namespace clang::tidy::bugprone
