//===--- NarrowingConversionsCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NarrowingConversionsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

NarrowingConversionsCheck::NarrowingConversionsCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnFloatingPointNarrowingConversion(
          Options.get("WarnOnFloatingPointNarrowingConversion", true)),
      PedanticMode(Options.get("PedanticMode", false)) {}

void NarrowingConversionsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnFloatingPointNarrowingConversion",
                WarnOnFloatingPointNarrowingConversion);
  Options.store(Opts, "PedanticMode", PedanticMode);
}

void NarrowingConversionsCheck::registerMatchers(MatchFinder *Finder) {
  // ceil() and floor() are guaranteed to return integers, even though the type
  // is not integral.
  const auto IsCeilFloorCallExpr = expr(callExpr(callee(functionDecl(
      hasAnyName("::ceil", "::std::ceil", "::floor", "::std::floor")))));

  // Casts:
  //   i = 0.5;
  //   void f(int); f(0.5);
  Finder->addMatcher(
      traverse(
          ast_type_traits::TK_AsIs,
          implicitCastExpr(hasImplicitDestinationType(builtinType()),
                           hasSourceExpression(hasType(builtinType())),
                           unless(hasSourceExpression(IsCeilFloorCallExpr)),
                           unless(hasParent(castExpr())),
                           unless(isInTemplateInstantiation()))
              .bind("cast")),
      this);

  // Binary operators:
  //   i += 0.5;
  Finder->addMatcher(binaryOperator(isAssignmentOperator(),
                                    hasLHS(expr(hasType(builtinType()))),
                                    hasRHS(expr(hasType(builtinType()))),
                                    unless(hasRHS(IsCeilFloorCallExpr)),
                                    unless(isInTemplateInstantiation()),
                                    // The `=` case generates an implicit cast
                                    // which is covered by the previous matcher.
                                    unless(hasOperatorName("=")))
                         .bind("binary_op"),
                     this);
}

static const BuiltinType *getBuiltinType(const Expr &E) {
  return E.getType().getCanonicalType().getTypePtr()->getAs<BuiltinType>();
}

static QualType getUnqualifiedType(const Expr &E) {
  return E.getType().getUnqualifiedType();
}

static APValue getConstantExprValue(const ASTContext &Ctx, const Expr &E) {
  llvm::APSInt IntegerConstant;
  if (E.isIntegerConstantExpr(IntegerConstant, Ctx))
    return APValue(IntegerConstant);
  APValue Constant;
  if (Ctx.getLangOpts().CPlusPlus && E.isCXX11ConstantExpr(Ctx, &Constant))
    return Constant;
  return {};
}

static bool getIntegerConstantExprValue(const ASTContext &Context,
                                        const Expr &E, llvm::APSInt &Value) {
  APValue Constant = getConstantExprValue(Context, E);
  if (!Constant.isInt())
    return false;
  Value = Constant.getInt();
  return true;
}

static bool getFloatingConstantExprValue(const ASTContext &Context,
                                         const Expr &E, llvm::APFloat &Value) {
  APValue Constant = getConstantExprValue(Context, E);
  if (!Constant.isFloat())
    return false;
  Value = Constant.getFloat();
  return true;
}

namespace {

struct IntegerRange {
  bool Contains(const IntegerRange &From) const {
    return llvm::APSInt::compareValues(Lower, From.Lower) <= 0 &&
           llvm::APSInt::compareValues(Upper, From.Upper) >= 0;
  }

  bool Contains(const llvm::APSInt &Value) const {
    return llvm::APSInt::compareValues(Lower, Value) <= 0 &&
           llvm::APSInt::compareValues(Upper, Value) >= 0;
  }

  llvm::APSInt Lower;
  llvm::APSInt Upper;
};

} // namespace

static IntegerRange createFromType(const ASTContext &Context,
                                   const BuiltinType &T) {
  if (T.isFloatingPoint()) {
    unsigned PrecisionBits = llvm::APFloatBase::semanticsPrecision(
        Context.getFloatTypeSemantics(T.desugar()));
    // Contrary to two's complement integer, floating point values are
    // symmetric and have the same number of positive and negative values.
    // The range of valid integers for a floating point value is:
    // [-2^PrecisionBits, 2^PrecisionBits]

    // Values are created with PrecisionBits plus two bits:
    // - One to express the missing negative value of 2's complement
    //   representation.
    // - One for the sign.
    llvm::APSInt UpperValue(PrecisionBits + 2, /*isUnsigned*/ false);
    UpperValue.setBit(PrecisionBits);
    llvm::APSInt LowerValue(PrecisionBits + 2, /*isUnsigned*/ false);
    LowerValue.setBit(PrecisionBits);
    LowerValue.setSignBit();
    return {LowerValue, UpperValue};
  }
  assert(T.isInteger() && "Unexpected builtin type");
  uint64_t TypeSize = Context.getTypeSize(&T);
  bool IsUnsignedInteger = T.isUnsignedInteger();
  return {llvm::APSInt::getMinValue(TypeSize, IsUnsignedInteger),
          llvm::APSInt::getMaxValue(TypeSize, IsUnsignedInteger)};
}

static bool isWideEnoughToHold(const ASTContext &Context,
                               const BuiltinType &FromType,
                               const BuiltinType &ToType) {
  IntegerRange FromIntegerRange = createFromType(Context, FromType);
  IntegerRange ToIntegerRange = createFromType(Context, ToType);
  return ToIntegerRange.Contains(FromIntegerRange);
}

static bool isWideEnoughToHold(const ASTContext &Context,
                               const llvm::APSInt &IntegerConstant,
                               const BuiltinType &ToType) {
  IntegerRange ToIntegerRange = createFromType(Context, ToType);
  return ToIntegerRange.Contains(IntegerConstant);
}

static llvm::SmallString<64> getValueAsString(const llvm::APSInt &Value,
                                              uint64_t HexBits) {
  llvm::SmallString<64> Str;
  Value.toString(Str, 10);
  if (HexBits > 0) {
    Str.append(" (0x");
    llvm::SmallString<32> HexValue;
    Value.toStringUnsigned(HexValue, 16);
    for (size_t I = HexValue.size(); I < (HexBits / 4); ++I)
      Str.append("0");
    Str.append(HexValue);
    Str.append(")");
  }
  return Str;
}

void NarrowingConversionsCheck::diagNarrowType(SourceLocation SourceLoc,
                                               const Expr &Lhs,
                                               const Expr &Rhs) {
  diag(SourceLoc, "narrowing conversion from %0 to %1")
      << getUnqualifiedType(Rhs) << getUnqualifiedType(Lhs);
}

void NarrowingConversionsCheck::diagNarrowTypeToSignedInt(
    SourceLocation SourceLoc, const Expr &Lhs, const Expr &Rhs) {
  diag(SourceLoc, "narrowing conversion from %0 to signed type %1 is "
                  "implementation-defined")
      << getUnqualifiedType(Rhs) << getUnqualifiedType(Lhs);
}

void NarrowingConversionsCheck::diagNarrowIntegerConstant(
    SourceLocation SourceLoc, const Expr &Lhs, const Expr &Rhs,
    const llvm::APSInt &Value) {
  diag(SourceLoc,
       "narrowing conversion from constant value %0 of type %1 to %2")
      << getValueAsString(Value, /*NoHex*/ 0) << getUnqualifiedType(Rhs)
      << getUnqualifiedType(Lhs);
}

void NarrowingConversionsCheck::diagNarrowIntegerConstantToSignedInt(
    SourceLocation SourceLoc, const Expr &Lhs, const Expr &Rhs,
    const llvm::APSInt &Value, const uint64_t HexBits) {
  diag(SourceLoc, "narrowing conversion from constant value %0 of type %1 "
                  "to signed type %2 is implementation-defined")
      << getValueAsString(Value, HexBits) << getUnqualifiedType(Rhs)
      << getUnqualifiedType(Lhs);
}

void NarrowingConversionsCheck::diagNarrowConstant(SourceLocation SourceLoc,
                                                   const Expr &Lhs,
                                                   const Expr &Rhs) {
  diag(SourceLoc, "narrowing conversion from constant %0 to %1")
      << getUnqualifiedType(Rhs) << getUnqualifiedType(Lhs);
}

void NarrowingConversionsCheck::diagConstantCast(SourceLocation SourceLoc,
                                                 const Expr &Lhs,
                                                 const Expr &Rhs) {
  diag(SourceLoc, "constant value should be of type of type %0 instead of %1")
      << getUnqualifiedType(Lhs) << getUnqualifiedType(Rhs);
}

void NarrowingConversionsCheck::diagNarrowTypeOrConstant(
    const ASTContext &Context, SourceLocation SourceLoc, const Expr &Lhs,
    const Expr &Rhs) {
  APValue Constant = getConstantExprValue(Context, Rhs);
  if (Constant.isInt())
    return diagNarrowIntegerConstant(SourceLoc, Lhs, Rhs, Constant.getInt());
  if (Constant.isFloat())
    return diagNarrowConstant(SourceLoc, Lhs, Rhs);
  return diagNarrowType(SourceLoc, Lhs, Rhs);
}

void NarrowingConversionsCheck::handleIntegralCast(const ASTContext &Context,
                                                   SourceLocation SourceLoc,
                                                   const Expr &Lhs,
                                                   const Expr &Rhs) {
  const BuiltinType *ToType = getBuiltinType(Lhs);
  // From [conv.integral]p7.3.8:
  // Conversions to unsigned integer is well defined so no warning is issued.
  // "The resulting value is the smallest unsigned value equal to the source
  // value modulo 2^n where n is the number of bits used to represent the
  // destination type."
  if (ToType->isUnsignedInteger())
    return;
  const BuiltinType *FromType = getBuiltinType(Rhs);
  llvm::APSInt IntegerConstant;
  if (getIntegerConstantExprValue(Context, Rhs, IntegerConstant)) {
    if (!isWideEnoughToHold(Context, IntegerConstant, *ToType))
      diagNarrowIntegerConstantToSignedInt(SourceLoc, Lhs, Rhs, IntegerConstant,
                                           Context.getTypeSize(FromType));
    return;
  }
  if (!isWideEnoughToHold(Context, *FromType, *ToType))
    diagNarrowTypeToSignedInt(SourceLoc, Lhs, Rhs);
}

void NarrowingConversionsCheck::handleIntegralToBoolean(
    const ASTContext &Context, SourceLocation SourceLoc, const Expr &Lhs,
    const Expr &Rhs) {
  // Conversion from Integral to Bool value is well defined.

  // We keep this function (even if it is empty) to make sure that
  // handleImplicitCast and handleBinaryOperator are symmetric in their behavior
  // and handle the same cases.
}

void NarrowingConversionsCheck::handleIntegralToFloating(
    const ASTContext &Context, SourceLocation SourceLoc, const Expr &Lhs,
    const Expr &Rhs) {
  const BuiltinType *ToType = getBuiltinType(Lhs);
  llvm::APSInt IntegerConstant;
  if (getIntegerConstantExprValue(Context, Rhs, IntegerConstant)) {
    if (!isWideEnoughToHold(Context, IntegerConstant, *ToType))
      diagNarrowIntegerConstant(SourceLoc, Lhs, Rhs, IntegerConstant);
    return;
  }
  const BuiltinType *FromType = getBuiltinType(Rhs);
  if (!isWideEnoughToHold(Context, *FromType, *ToType))
    diagNarrowType(SourceLoc, Lhs, Rhs);
}

void NarrowingConversionsCheck::handleFloatingToIntegral(
    const ASTContext &Context, SourceLocation SourceLoc, const Expr &Lhs,
    const Expr &Rhs) {
  llvm::APFloat FloatConstant(0.0);

  // We always warn when Rhs is non-constexpr.
  if (!getFloatingConstantExprValue(Context, Rhs, FloatConstant))
    return diagNarrowType(SourceLoc, Lhs, Rhs);

  QualType DestType = Lhs.getType();
  unsigned DestWidth = Context.getIntWidth(DestType);
  bool DestSigned = DestType->isSignedIntegerOrEnumerationType();
  llvm::APSInt Result = llvm::APSInt(DestWidth, !DestSigned);
  bool IsExact = false;
  bool Overflows = FloatConstant.convertToInteger(
                       Result, llvm::APFloat::rmTowardZero, &IsExact) &
                   llvm::APFloat::opInvalidOp;
  // We warn iff the constant floating point value is not exactly representable.
  if (Overflows || !IsExact)
    return diagNarrowConstant(SourceLoc, Lhs, Rhs);

  if (PedanticMode)
    return diagConstantCast(SourceLoc, Lhs, Rhs);
}

void NarrowingConversionsCheck::handleFloatingToBoolean(
    const ASTContext &Context, SourceLocation SourceLoc, const Expr &Lhs,
    const Expr &Rhs) {
  return diagNarrowTypeOrConstant(Context, SourceLoc, Lhs, Rhs);
}

void NarrowingConversionsCheck::handleBooleanToSignedIntegral(
    const ASTContext &Context, SourceLocation SourceLoc, const Expr &Lhs,
    const Expr &Rhs) {
  // Conversion from Bool to SignedIntegral value is well defined.

  // We keep this function (even if it is empty) to make sure that
  // handleImplicitCast and handleBinaryOperator are symmetric in their behavior
  // and handle the same cases.
}

void NarrowingConversionsCheck::handleFloatingCast(const ASTContext &Context,
                                                   SourceLocation SourceLoc,
                                                   const Expr &Lhs,
                                                   const Expr &Rhs) {
  if (WarnOnFloatingPointNarrowingConversion) {
    const BuiltinType *ToType = getBuiltinType(Lhs);
    APValue Constant = getConstantExprValue(Context, Rhs);
    if (Constant.isFloat()) {
      // From [dcl.init.list]p7.2:
      // Floating point constant narrowing only takes place when the value is
      // not within destination range. We convert the value to the destination
      // type and check if the resulting value is infinity.
      llvm::APFloat Tmp = Constant.getFloat();
      bool UnusedLosesInfo;
      Tmp.convert(Context.getFloatTypeSemantics(ToType->desugar()),
                  llvm::APFloatBase::rmNearestTiesToEven, &UnusedLosesInfo);
      if (Tmp.isInfinity())
        diagNarrowConstant(SourceLoc, Lhs, Rhs);
      return;
    }
    const BuiltinType *FromType = getBuiltinType(Rhs);
    if (ToType->getKind() < FromType->getKind())
      diagNarrowType(SourceLoc, Lhs, Rhs);
  }
}

void NarrowingConversionsCheck::handleBinaryOperator(const ASTContext &Context,
                                                     SourceLocation SourceLoc,
                                                     const Expr &Lhs,
                                                     const Expr &Rhs) {
  assert(!Lhs.isInstantiationDependent() && !Rhs.isInstantiationDependent() &&
         "Dependent types must be check before calling this function");
  const BuiltinType *LhsType = getBuiltinType(Lhs);
  const BuiltinType *RhsType = getBuiltinType(Rhs);
  if (RhsType == nullptr || LhsType == nullptr)
    return;
  if (RhsType->getKind() == BuiltinType::Bool && LhsType->isSignedInteger())
    return handleBooleanToSignedIntegral(Context, SourceLoc, Lhs, Rhs);
  if (RhsType->isInteger() && LhsType->getKind() == BuiltinType::Bool)
    return handleIntegralToBoolean(Context, SourceLoc, Lhs, Rhs);
  if (RhsType->isInteger() && LhsType->isFloatingPoint())
    return handleIntegralToFloating(Context, SourceLoc, Lhs, Rhs);
  if (RhsType->isInteger() && LhsType->isInteger())
    return handleIntegralCast(Context, SourceLoc, Lhs, Rhs);
  if (RhsType->isFloatingPoint() && LhsType->getKind() == BuiltinType::Bool)
    return handleFloatingToBoolean(Context, SourceLoc, Lhs, Rhs);
  if (RhsType->isFloatingPoint() && LhsType->isInteger())
    return handleFloatingToIntegral(Context, SourceLoc, Lhs, Rhs);
  if (RhsType->isFloatingPoint() && LhsType->isFloatingPoint())
    return handleFloatingCast(Context, SourceLoc, Lhs, Rhs);
}

bool NarrowingConversionsCheck::handleConditionalOperator(
    const ASTContext &Context, const Expr &Lhs, const Expr &Rhs) {
  if (const auto *CO = llvm::dyn_cast<ConditionalOperator>(&Rhs)) {
    // We have an expression like so: `output = cond ? lhs : rhs`
    // From the point of view of narrowing conversion we treat it as two
    // expressions `output = lhs` and `output = rhs`.
    handleBinaryOperator(Context, CO->getLHS()->getExprLoc(), Lhs,
                         *CO->getLHS());
    handleBinaryOperator(Context, CO->getRHS()->getExprLoc(), Lhs,
                         *CO->getRHS());
    return true;
  }
  return false;
}

void NarrowingConversionsCheck::handleImplicitCast(
    const ASTContext &Context, const ImplicitCastExpr &Cast) {
  if (Cast.getExprLoc().isMacroID())
    return;
  const Expr &Lhs = Cast;
  const Expr &Rhs = *Cast.getSubExpr();
  if (Lhs.isInstantiationDependent() || Rhs.isInstantiationDependent())
    return;
  if (handleConditionalOperator(Context, Lhs, Rhs))
    return;
  SourceLocation SourceLoc = Lhs.getExprLoc();
  switch (Cast.getCastKind()) {
  case CK_BooleanToSignedIntegral:
    return handleBooleanToSignedIntegral(Context, SourceLoc, Lhs, Rhs);
  case CK_IntegralToBoolean:
    return handleIntegralToBoolean(Context, SourceLoc, Lhs, Rhs);
  case CK_IntegralToFloating:
    return handleIntegralToFloating(Context, SourceLoc, Lhs, Rhs);
  case CK_IntegralCast:
    return handleIntegralCast(Context, SourceLoc, Lhs, Rhs);
  case CK_FloatingToBoolean:
    return handleFloatingToBoolean(Context, SourceLoc, Lhs, Rhs);
  case CK_FloatingToIntegral:
    return handleFloatingToIntegral(Context, SourceLoc, Lhs, Rhs);
  case CK_FloatingCast:
    return handleFloatingCast(Context, SourceLoc, Lhs, Rhs);
  default:
    break;
  }
}

void NarrowingConversionsCheck::handleBinaryOperator(const ASTContext &Context,
                                                     const BinaryOperator &Op) {
  if (Op.getBeginLoc().isMacroID())
    return;
  const Expr &Lhs = *Op.getLHS();
  const Expr &Rhs = *Op.getRHS();
  if (Lhs.isInstantiationDependent() || Rhs.isInstantiationDependent())
    return;
  if (handleConditionalOperator(Context, Lhs, Rhs))
    return;
  handleBinaryOperator(Context, Rhs.getBeginLoc(), Lhs, Rhs);
}

void NarrowingConversionsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Op = Result.Nodes.getNodeAs<BinaryOperator>("binary_op"))
    return handleBinaryOperator(*Result.Context, *Op);
  if (const auto *Cast = Result.Nodes.getNodeAs<ImplicitCastExpr>("cast"))
    return handleImplicitCast(*Result.Context, *Cast);
  llvm_unreachable("must be binary operator or cast expression");
}
} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
