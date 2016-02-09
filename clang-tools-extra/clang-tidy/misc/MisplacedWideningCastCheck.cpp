//===--- MisplacedWideningCastCheck.cpp - clang-tidy-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MisplacedWideningCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void MisplacedWideningCastCheck::registerMatchers(MatchFinder *Finder) {
  auto Calc = expr(anyOf(binaryOperator(anyOf(
                             hasOperatorName("+"), hasOperatorName("-"),
                             hasOperatorName("*"), hasOperatorName("<<"))),
                         unaryOperator(hasOperatorName("~"))),
                   hasType(isInteger()))
                  .bind("Calc");

  auto Cast = explicitCastExpr(anyOf(cStyleCastExpr(), cxxStaticCastExpr(),
                                     cxxReinterpretCastExpr()),
                               hasDestinationType(isInteger()),
                               has(Calc))
                  .bind("Cast");

  Finder->addMatcher(varDecl(has(Cast)), this);
  Finder->addMatcher(returnStmt(has(Cast)), this);
  Finder->addMatcher(binaryOperator(hasOperatorName("="), hasRHS(Cast)), this);
}

static unsigned getMaxCalculationWidth(ASTContext &Context, const Expr *E) {
  E = E->IgnoreParenImpCasts();

  if (const auto *Bop = dyn_cast<BinaryOperator>(E)) {
    unsigned LHSWidth = getMaxCalculationWidth(Context, Bop->getLHS());
    unsigned RHSWidth = getMaxCalculationWidth(Context, Bop->getRHS());
    if (Bop->getOpcode() == BO_Mul)
      return LHSWidth + RHSWidth;
    if (Bop->getOpcode() == BO_Add)
      return std::max(LHSWidth, RHSWidth) + 1;
    if (Bop->getOpcode() == BO_Rem) {
      llvm::APSInt Val;
      if (Bop->getRHS()->EvaluateAsInt(Val, Context))
        return Val.getActiveBits();
    } else if (Bop->getOpcode() == BO_Shl) {
      llvm::APSInt Bits;
      if (Bop->getRHS()->EvaluateAsInt(Bits, Context)) {
        // We don't handle negative values and large values well. It is assumed
        // that compiler warnings are written for such values so the user will
        // fix that.
        return LHSWidth + Bits.getExtValue();
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

void MisplacedWideningCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<ExplicitCastExpr>("Cast");
  if (Cast->getLocStart().isMacroID())
    return;

  const auto *Calc = Result.Nodes.getNodeAs<Expr>("Calc");
  if (Calc->getLocStart().isMacroID())
    return;

  ASTContext &Context = *Result.Context;

  QualType CastType = Cast->getType();
  QualType CalcType = Calc->getType();

  if (Context.getIntWidth(CastType) <= Context.getIntWidth(CalcType))
    return;

  if (Context.getIntWidth(CalcType) >= getMaxCalculationWidth(Context, Calc))
    return;

  diag(Cast->getLocStart(), "either cast from %0 to %1 is ineffective, or "
                            "there is loss of precision before the conversion")
      << CalcType << CastType;
}

} // namespace misc
} // namespace tidy
} // namespace clang
