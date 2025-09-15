//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TooSmallLoopVariableCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static constexpr llvm::StringLiteral LoopName =
    llvm::StringLiteral("forLoopName");
static constexpr llvm::StringLiteral LoopVarName =
    llvm::StringLiteral("loopVar");
static constexpr llvm::StringLiteral LoopVarCastName =
    llvm::StringLiteral("loopVarCast");
static constexpr llvm::StringLiteral LoopUpperBoundName =
    llvm::StringLiteral("loopUpperBound");
static constexpr llvm::StringLiteral LoopIncrementName =
    llvm::StringLiteral("loopIncrement");

namespace {

struct MagnitudeBits {
  unsigned WidthWithoutSignBit = 0U;
  unsigned BitFieldWidth = 0U;

  bool operator<(const MagnitudeBits &Other) const noexcept {
    return WidthWithoutSignBit < Other.WidthWithoutSignBit;
  }

  bool operator!=(const MagnitudeBits &Other) const noexcept {
    return WidthWithoutSignBit != Other.WidthWithoutSignBit ||
           BitFieldWidth != Other.BitFieldWidth;
  }
};

} // namespace

TooSmallLoopVariableCheck::TooSmallLoopVariableCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MagnitudeBitsUpperLimit(Options.get("MagnitudeBitsUpperLimit", 16U)) {}

void TooSmallLoopVariableCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MagnitudeBitsUpperLimit", MagnitudeBitsUpperLimit);
}

/// The matcher for loops with suspicious integer loop variable.
///
/// In this general example, assuming 'j' and 'k' are of integral type:
/// \code
///   for (...; j < 3 + 2; ++k) { ... }
/// \endcode
/// The following string identifiers are bound to these parts of the AST:
///   LoopVarName: 'j' (as a VarDecl)
///   LoopVarCastName: 'j' (after implicit conversion)
///   LoopUpperBoundName: '3 + 2' (as an Expr)
///   LoopIncrementName: 'k' (as an Expr)
///   LoopName: The entire for loop (as a ForStmt)
///
void TooSmallLoopVariableCheck::registerMatchers(MatchFinder *Finder) {
  StatementMatcher LoopVarMatcher =
      expr(ignoringParenImpCasts(
               anyOf(declRefExpr(to(varDecl(hasType(isInteger())))),
                     memberExpr(member(fieldDecl(hasType(isInteger())))))))
          .bind(LoopVarName);

  // We need to catch only those comparisons which contain any integer cast.
  StatementMatcher LoopVarConversionMatcher = traverse(
      TK_AsIs, implicitCastExpr(hasImplicitDestinationType(isInteger()),
                                has(ignoringParenImpCasts(LoopVarMatcher)))
                   .bind(LoopVarCastName));

  // We are interested in only those cases when the loop bound is a variable
  // value (not const, enum, etc.).
  StatementMatcher LoopBoundMatcher =
      expr(ignoringParenImpCasts(allOf(
               hasType(isInteger()), unless(integerLiteral()),
               unless(allOf(
                   hasType(isConstQualified()),
                   declRefExpr(to(varDecl(anyOf(
                       hasInitializer(ignoringParenImpCasts(integerLiteral())),
                       isConstexpr(), isConstinit())))))),
               unless(hasType(enumType())))))
          .bind(LoopUpperBoundName);

  // We use the loop increment expression only to make sure we found the right
  // loop variable.
  StatementMatcher IncrementMatcher =
      expr(ignoringParenImpCasts(hasType(isInteger()))).bind(LoopIncrementName);

  Finder->addMatcher(
      forStmt(
          hasCondition(anyOf(
              binaryOperator(hasOperatorName("<"),
                             hasLHS(LoopVarConversionMatcher),
                             hasRHS(LoopBoundMatcher)),
              binaryOperator(hasOperatorName("<="),
                             hasLHS(LoopVarConversionMatcher),
                             hasRHS(LoopBoundMatcher)),
              binaryOperator(hasOperatorName(">"), hasLHS(LoopBoundMatcher),
                             hasRHS(LoopVarConversionMatcher)),
              binaryOperator(hasOperatorName(">="), hasLHS(LoopBoundMatcher),
                             hasRHS(LoopVarConversionMatcher)))),
          hasIncrement(IncrementMatcher))
          .bind(LoopName),
      this);
}

/// Returns the magnitude bits of an integer type.
static MagnitudeBits calcMagnitudeBits(const ASTContext &Context,
                                       const QualType &IntExprType,
                                       const Expr *IntExpr) {
  assert(IntExprType->isIntegerType());

  unsigned SignedBits = IntExprType->isUnsignedIntegerType() ? 0U : 1U;

  if (const auto *BitField = IntExpr->getSourceBitField()) {
    unsigned BitFieldWidth = BitField->getBitWidthValue();
    return {BitFieldWidth - SignedBits, BitFieldWidth};
  }

  unsigned IntWidth = Context.getIntWidth(IntExprType);
  return {IntWidth - SignedBits, 0U};
}

/// Calculate the upper bound expression's magnitude bits, but ignore
/// constant like values to reduce false positives.
static MagnitudeBits
calcUpperBoundMagnitudeBits(const ASTContext &Context, const Expr *UpperBound,
                            const QualType &UpperBoundType) {
  // Ignore casting caused by constant values inside a binary operator.
  // We are interested in variable values' magnitude bits.
  if (const auto *BinOperator = dyn_cast<BinaryOperator>(UpperBound)) {
    const Expr *RHSE = BinOperator->getRHS()->IgnoreParenImpCasts();
    const Expr *LHSE = BinOperator->getLHS()->IgnoreParenImpCasts();

    QualType RHSEType = RHSE->getType();
    QualType LHSEType = LHSE->getType();

    if (!RHSEType->isIntegerType() || !LHSEType->isIntegerType())
      return {};

    bool RHSEIsConstantValue = RHSEType->isEnumeralType() ||
                               RHSEType.isConstQualified() ||
                               isa<IntegerLiteral>(RHSE);
    bool LHSEIsConstantValue = LHSEType->isEnumeralType() ||
                               LHSEType.isConstQualified() ||
                               isa<IntegerLiteral>(LHSE);

    // Avoid false positives produced by two constant values.
    if (RHSEIsConstantValue && LHSEIsConstantValue)
      return {};
    if (RHSEIsConstantValue)
      return calcMagnitudeBits(Context, LHSEType, LHSE);
    if (LHSEIsConstantValue)
      return calcMagnitudeBits(Context, RHSEType, RHSE);

    return std::max(calcMagnitudeBits(Context, LHSEType, LHSE),
                    calcMagnitudeBits(Context, RHSEType, RHSE));
  }

  return calcMagnitudeBits(Context, UpperBoundType, UpperBound);
}

static std::string formatIntegralType(const QualType &Type,
                                      const MagnitudeBits &Info) {
  std::string Name = Type.getAsString();
  if (!Info.BitFieldWidth)
    return Name;

  Name += ':';
  Name += std::to_string(Info.BitFieldWidth);
  return Name;
}

void TooSmallLoopVariableCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *LoopVar = Result.Nodes.getNodeAs<Expr>(LoopVarName);
  const auto *UpperBound =
      Result.Nodes.getNodeAs<Expr>(LoopUpperBoundName)->IgnoreParenImpCasts();
  const auto *LoopIncrement =
      Result.Nodes.getNodeAs<Expr>(LoopIncrementName)->IgnoreParenImpCasts();

  // We matched the loop variable incorrectly.
  if (LoopVar->getType() != LoopIncrement->getType())
    return;

  ASTContext &Context = *Result.Context;

  const QualType LoopVarType = LoopVar->getType();
  const MagnitudeBits LoopVarMagnitudeBits =
      calcMagnitudeBits(Context, LoopVarType, LoopVar);

  const MagnitudeBits LoopIncrementMagnitudeBits =
      calcMagnitudeBits(Context, LoopIncrement->getType(), LoopIncrement);
  // We matched the loop variable incorrectly.
  if (LoopIncrementMagnitudeBits != LoopVarMagnitudeBits)
    return;

  const QualType UpperBoundType = UpperBound->getType();
  const MagnitudeBits UpperBoundMagnitudeBits =
      calcUpperBoundMagnitudeBits(Context, UpperBound, UpperBoundType);

  if ((0U == UpperBoundMagnitudeBits.WidthWithoutSignBit) ||
      (LoopVarMagnitudeBits.WidthWithoutSignBit > MagnitudeBitsUpperLimit) ||
      (LoopVarMagnitudeBits.WidthWithoutSignBit >=
       UpperBoundMagnitudeBits.WidthWithoutSignBit))
    return;

  diag(LoopVar->getBeginLoc(),
       "loop variable has narrower type '%0' than iteration's upper bound '%1'")
      << formatIntegralType(LoopVarType, LoopVarMagnitudeBits)
      << formatIntegralType(UpperBoundType, UpperBoundMagnitudeBits);
}

} // namespace clang::tidy::bugprone
