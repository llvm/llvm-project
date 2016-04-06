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
#include "llvm/ADT/DenseMap.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

MisplacedWideningCastCheck::MisplacedWideningCastCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckImplicitCasts(Options.get("CheckImplicitCasts", true)) {}

void MisplacedWideningCastCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckImplicitCasts", CheckImplicitCasts);
}

void MisplacedWideningCastCheck::registerMatchers(MatchFinder *Finder) {
  auto Calc = expr(anyOf(binaryOperator(anyOf(
                             hasOperatorName("+"), hasOperatorName("-"),
                             hasOperatorName("*"), hasOperatorName("<<"))),
                         unaryOperator(hasOperatorName("~"))),
                   hasType(isInteger()))
                  .bind("Calc");

  auto ExplicitCast =
      explicitCastExpr(hasDestinationType(isInteger()), has(Calc));
  auto ImplicitCast =
      implicitCastExpr(hasImplicitDestinationType(isInteger()), has(Calc));
  auto Cast = expr(anyOf(ExplicitCast, ImplicitCast)).bind("Cast");

  Finder->addMatcher(varDecl(hasInitializer(Cast)), this);
  Finder->addMatcher(returnStmt(hasReturnValue(Cast)), this);
  Finder->addMatcher(callExpr(hasAnyArgument(Cast)), this);
  Finder->addMatcher(binaryOperator(hasOperatorName("="), hasRHS(Cast)), this);
  Finder->addMatcher(
      binaryOperator(anyOf(hasOperatorName("=="), hasOperatorName("!="),
                           hasOperatorName("<"), hasOperatorName("<="),
                           hasOperatorName(">"), hasOperatorName(">=")),
                     anyOf(hasLHS(Cast), hasRHS(Cast))),
      this);
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

static llvm::SmallDenseMap<int, int> createRelativeIntSizesMap() {
  llvm::SmallDenseMap<int, int> Result;
  Result[BuiltinType::UChar] = 1;
  Result[BuiltinType::SChar] = 1;
  Result[BuiltinType::Char_U] = 1;
  Result[BuiltinType::Char_S] = 1;
  Result[BuiltinType::UShort] = 2;
  Result[BuiltinType::Short] = 2;
  Result[BuiltinType::UInt] = 3;
  Result[BuiltinType::Int] = 3;
  Result[BuiltinType::ULong] = 4;
  Result[BuiltinType::Long] = 4;
  Result[BuiltinType::ULongLong] = 5;
  Result[BuiltinType::LongLong] = 5;
  Result[BuiltinType::UInt128] = 6;
  Result[BuiltinType::Int128] = 6;
  return Result;
}

static llvm::SmallDenseMap<int, int> createRelativeCharSizesMap() {
  llvm::SmallDenseMap<int, int> Result;
  Result[BuiltinType::UChar] = 1;
  Result[BuiltinType::SChar] = 1;
  Result[BuiltinType::Char_U] = 1;
  Result[BuiltinType::Char_S] = 1;
  Result[BuiltinType::Char16] = 2;
  Result[BuiltinType::Char32] = 3;
  return Result;
}

static llvm::SmallDenseMap<int, int> createRelativeCharSizesWMap() {
  llvm::SmallDenseMap<int, int> Result;
  Result[BuiltinType::UChar] = 1;
  Result[BuiltinType::SChar] = 1;
  Result[BuiltinType::Char_U] = 1;
  Result[BuiltinType::Char_S] = 1;
  Result[BuiltinType::WChar_U] = 2;
  Result[BuiltinType::WChar_S] = 2;
  return Result;
}

static bool isFirstWider(BuiltinType::Kind First, BuiltinType::Kind Second) {
  static const llvm::SmallDenseMap<int, int> RelativeIntSizes(
      createRelativeIntSizesMap());
  static const llvm::SmallDenseMap<int, int> RelativeCharSizes(
      createRelativeCharSizesMap());
  static const llvm::SmallDenseMap<int, int> RelativeCharSizesW(
      createRelativeCharSizesWMap());

  int FirstSize, SecondSize;
  if ((FirstSize = RelativeIntSizes.lookup(First)) &&
      (SecondSize = RelativeIntSizes.lookup(Second)))
    return FirstSize > SecondSize;
  if ((FirstSize = RelativeCharSizes.lookup(First)) &&
      (SecondSize = RelativeCharSizes.lookup(Second)))
    return FirstSize > SecondSize;
  if ((FirstSize = RelativeCharSizesW.lookup(First)) &&
      (SecondSize = RelativeCharSizesW.lookup(Second)))
    return FirstSize > SecondSize;
  return false;
}

void MisplacedWideningCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<CastExpr>("Cast");
  if (!CheckImplicitCasts && isa<ImplicitCastExpr>(Cast))
    return;
  if (Cast->getLocStart().isMacroID())
    return;

  const auto *Calc = Result.Nodes.getNodeAs<Expr>("Calc");
  if (Calc->getLocStart().isMacroID())
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
    if (CastBuiltinType && CalcBuiltinType &&
        !isFirstWider(CastBuiltinType->getKind(), CalcBuiltinType->getKind()))
      return;
  }

  // Don't write a warning if we can easily see that the result is not
  // truncated.
  if (Context.getIntWidth(CalcType) >= getMaxCalculationWidth(Context, Calc))
    return;

  diag(Cast->getLocStart(), "either cast from %0 to %1 is ineffective, or "
                            "there is loss of precision before the conversion")
      << CalcType << CastType;
}

} // namespace misc
} // namespace tidy
} // namespace clang
