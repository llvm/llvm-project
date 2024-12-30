//===--- NarrowingConversionsCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NarrowingConversionsCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER_P(QualType, hasAnyType, std::vector<StringRef>, Names) {
  if (Names.empty())
    return false;

  std::string Name = Node.getLocalUnqualifiedType().getAsString();
  return llvm::any_of(Names, [&Name](StringRef Ref) { return Ref == Name; });
}

AST_MATCHER(FieldDecl, hasIntBitwidth) {
  assert(Node.isBitField());
  const ASTContext &Ctx = Node.getASTContext();
  unsigned IntBitWidth = Ctx.getIntWidth(Ctx.IntTy);
  unsigned CurrentBitWidth = Node.getBitWidthValue(Ctx);
  return IntBitWidth == CurrentBitWidth;
}

} // namespace

NarrowingConversionsCheck::NarrowingConversionsCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnIntegerNarrowingConversion(
          Options.get("WarnOnIntegerNarrowingConversion", true)),
      WarnOnIntegerToFloatingPointNarrowingConversion(
          Options.get("WarnOnIntegerToFloatingPointNarrowingConversion", true)),
      WarnOnFloatingPointNarrowingConversion(
          Options.get("WarnOnFloatingPointNarrowingConversion", true)),
      WarnWithinTemplateInstantiation(
          Options.get("WarnWithinTemplateInstantiation", false)),
      WarnOnEquivalentBitWidth(Options.get("WarnOnEquivalentBitWidth", true)),
      IgnoreConversionFromTypes(Options.get("IgnoreConversionFromTypes", "")),
      PedanticMode(Options.get("PedanticMode", false)) {}

void NarrowingConversionsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnIntegerNarrowingConversion",
                WarnOnIntegerNarrowingConversion);
  Options.store(Opts, "WarnOnIntegerToFloatingPointNarrowingConversion",
                WarnOnIntegerToFloatingPointNarrowingConversion);
  Options.store(Opts, "WarnOnFloatingPointNarrowingConversion",
                WarnOnFloatingPointNarrowingConversion);
  Options.store(Opts, "WarnWithinTemplateInstantiation",
                WarnWithinTemplateInstantiation);
  Options.store(Opts, "WarnOnEquivalentBitWidth", WarnOnEquivalentBitWidth);
  Options.store(Opts, "IgnoreConversionFromTypes", IgnoreConversionFromTypes);
  Options.store(Opts, "PedanticMode", PedanticMode);
}

void NarrowingConversionsCheck::registerMatchers(MatchFinder *Finder) {
  // ceil() and floor() are guaranteed to return integers, even though the type
  // is not integral.
  const auto IsCeilFloorCallExpr = expr(callExpr(callee(functionDecl(
      hasAnyName("::ceil", "::std::ceil", "::floor", "::std::floor")))));

  std::vector<StringRef> IgnoreConversionFromTypesVec =
      utils::options::parseStringList(IgnoreConversionFromTypes);

  // We may want to exclude other types from the checks, such as `size_type`
  // and `difference_type`. These are often used to count elements, represented
  // in 64 bits and assigned to `int`. Rarely are people counting >2B elements.
  const auto IsConversionFromIgnoredType =
      anyOf(hasType(namedDecl(hasAnyName(IgnoreConversionFromTypesVec))),
            allOf(unless(hasType(namedDecl())),
                  hasType(qualType(hasAnyType(IgnoreConversionFromTypesVec)))));

  // `IsConversionFromIgnoredType` will ignore narrowing calls from those types,
  // but not expressions that are promoted to an ignored type as a result of a
  // binary expression with one of those types.
  // For example, it will continue to reject:
  // `int narrowed = int_value + container.size()`.
  // We attempt to address common incidents of compound expressions with
  // `IsIgnoredTypeTwoLevelsDeep`, allowing binary expressions that have one
  // operand of the ignored types and the other operand of another integer type.
  const auto IsIgnoredTypeTwoLevelsDeep =
      anyOf(IsConversionFromIgnoredType,
            binaryOperator(hasOperands(IsConversionFromIgnoredType,
                                       hasType(isInteger()))));

  // Bitfields are special. Due to integral promotion [conv.prom/5] bitfield
  // member access expressions are frequently wrapped by an implicit cast to
  // `int` if that type can represent all the values of the bitfield.
  //
  // Consider these examples:
  //   struct SmallBitfield { unsigned int id : 4; };
  //   x.id & 1;             (case-1)
  //   x.id & 1u;            (case-2)
  //   x.id << 1u;           (case-3)
  //   (unsigned)x.id << 1;  (case-4)
  //
  // Due to the promotion rules, we would get a warning for case-1. It's
  // debatable how useful this is, but the user at least has a convenient way of
  // //fixing// it by adding the `u` unsigned-suffix to the literal as
  // demonstrated by case-2. However, this won't work for shift operators like
  // the one in case-3. In case of a normal binary operator, both operands
  // contribute to the result type. However, the type of the shift expression is
  // the promoted type of the left operand. One could still suppress this
  // superfluous warning by explicitly casting the bitfield member access as
  // case-4 demonstrates, but why? The compiler already knew that the value from
  // the member access should safely fit into an `int`, why do we have this
  // warning in the first place? So, hereby we suppress this specific scenario.
  //
  // Note that the bitshift operation might invoke unspecified/undefined
  // behavior, but that's another topic, this checker is about detecting
  // conversion-related defects.
  //
  // Example AST for `x.id << 1`:
  //   BinaryOperator 'int' '<<'
  //   |-ImplicitCastExpr 'int' <IntegralCast>
  //   | `-ImplicitCastExpr 'unsigned int' <LValueToRValue>
  //   |   `-MemberExpr 'unsigned int' lvalue bitfield .id
  //   |     `-DeclRefExpr 'SmallBitfield' lvalue ParmVar 'x' 'SmallBitfield'
  //   `-IntegerLiteral 'int' 1
  const auto ImplicitIntWidenedBitfieldValue = implicitCastExpr(
      hasCastKind(CK_IntegralCast), hasType(asString("int")),
      has(castExpr(hasCastKind(CK_LValueToRValue),
                   has(ignoringParens(memberExpr(hasDeclaration(
                       fieldDecl(isBitField(), unless(hasIntBitwidth())))))))));

  // Casts:
  //   i = 0.5;
  //   void f(int); f(0.5);
  Finder->addMatcher(
      traverse(TK_AsIs, implicitCastExpr(
                            hasImplicitDestinationType(
                                hasUnqualifiedDesugaredType(builtinType())),
                            hasSourceExpression(hasType(
                                hasUnqualifiedDesugaredType(builtinType()))),
                            unless(hasSourceExpression(IsCeilFloorCallExpr)),
                            unless(hasParent(castExpr())),
                            WarnWithinTemplateInstantiation
                                ? stmt()
                                : stmt(unless(isInTemplateInstantiation())),
                            IgnoreConversionFromTypes.empty()
                                ? castExpr()
                                : castExpr(unless(hasSourceExpression(
                                      IsIgnoredTypeTwoLevelsDeep))),
                            unless(ImplicitIntWidenedBitfieldValue))
                            .bind("cast")),
      this);

  // Binary operators:
  //   i += 0.5;
  Finder->addMatcher(
      binaryOperator(
          isAssignmentOperator(),
          hasLHS(expr(hasType(hasUnqualifiedDesugaredType(builtinType())))),
          hasRHS(expr(hasType(hasUnqualifiedDesugaredType(builtinType())))),
          unless(hasRHS(IsCeilFloorCallExpr)),
          WarnWithinTemplateInstantiation
              ? binaryOperator()
              : binaryOperator(unless(isInTemplateInstantiation())),
          IgnoreConversionFromTypes.empty()
              ? binaryOperator()
              : binaryOperator(unless(hasRHS(IsIgnoredTypeTwoLevelsDeep))),
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
  if (auto IntegerConstant = E.getIntegerConstantExpr(Ctx))
    return APValue(*IntegerConstant);
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
  bool contains(const IntegerRange &From) const {
    return llvm::APSInt::compareValues(Lower, From.Lower) <= 0 &&
           llvm::APSInt::compareValues(Upper, From.Upper) >= 0;
  }

  bool contains(const llvm::APSInt &Value) const {
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
  return ToIntegerRange.contains(FromIntegerRange);
}

static bool isWideEnoughToHold(const ASTContext &Context,
                               const llvm::APSInt &IntegerConstant,
                               const BuiltinType &ToType) {
  IntegerRange ToIntegerRange = createFromType(Context, ToType);
  return ToIntegerRange.contains(IntegerConstant);
}

// Returns true iff the floating point constant can be losslessly represented
// by an integer in the given destination type. eg. 2.0 can be accurately
// represented by an int32_t, but neither 2^33 nor 2.001 can.
static bool isFloatExactlyRepresentable(const ASTContext &Context,
                                        const llvm::APFloat &FloatConstant,
                                        const QualType &DestType) {
  unsigned DestWidth = Context.getIntWidth(DestType);
  bool DestSigned = DestType->isSignedIntegerOrEnumerationType();
  llvm::APSInt Result = llvm::APSInt(DestWidth, !DestSigned);
  bool IsExact = false;
  bool Overflows = FloatConstant.convertToInteger(
                       Result, llvm::APFloat::rmTowardZero, &IsExact) &
                   llvm::APFloat::opInvalidOp;
  return !Overflows && IsExact;
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

bool NarrowingConversionsCheck::isWarningInhibitedByEquivalentSize(
    const ASTContext &Context, const BuiltinType &FromType,
    const BuiltinType &ToType) const {
  // With this option, we don't warn on conversions that have equivalent width
  // in bits. eg. uint32 <-> int32.
  if (!WarnOnEquivalentBitWidth) {
    uint64_t FromTypeSize = Context.getTypeSize(&FromType);
    uint64_t ToTypeSize = Context.getTypeSize(&ToType);
    if (FromTypeSize == ToTypeSize) {
      return true;
    }
  }
  return false;
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
  if (WarnOnIntegerNarrowingConversion) {
    const BuiltinType *ToType = getBuiltinType(Lhs);
    // From [conv.integral]p7.3.8:
    // Conversions to unsigned integer is well defined so no warning is issued.
    // "The resulting value is the smallest unsigned value equal to the source
    // value modulo 2^n where n is the number of bits used to represent the
    // destination type."
    if (ToType->isUnsignedInteger())
      return;
    const BuiltinType *FromType = getBuiltinType(Rhs);

    // With this option, we don't warn on conversions that have equivalent width
    // in bits. eg. uint32 <-> int32.
    if (!WarnOnEquivalentBitWidth) {
      uint64_t FromTypeSize = Context.getTypeSize(FromType);
      uint64_t ToTypeSize = Context.getTypeSize(ToType);
      if (FromTypeSize == ToTypeSize)
        return;
    }

    llvm::APSInt IntegerConstant;
    if (getIntegerConstantExprValue(Context, Rhs, IntegerConstant)) {
      if (!isWideEnoughToHold(Context, IntegerConstant, *ToType))
        diagNarrowIntegerConstantToSignedInt(SourceLoc, Lhs, Rhs,
                                             IntegerConstant,
                                             Context.getTypeSize(FromType));
      return;
    }
    if (!isWideEnoughToHold(Context, *FromType, *ToType))
      diagNarrowTypeToSignedInt(SourceLoc, Lhs, Rhs);
  }
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
  if (WarnOnIntegerToFloatingPointNarrowingConversion) {
    const BuiltinType *ToType = getBuiltinType(Lhs);
    llvm::APSInt IntegerConstant;
    if (getIntegerConstantExprValue(Context, Rhs, IntegerConstant)) {
      if (!isWideEnoughToHold(Context, IntegerConstant, *ToType))
        diagNarrowIntegerConstant(SourceLoc, Lhs, Rhs, IntegerConstant);
      return;
    }

    const BuiltinType *FromType = getBuiltinType(Rhs);
    if (isWarningInhibitedByEquivalentSize(Context, *FromType, *ToType))
      return;
    if (!isWideEnoughToHold(Context, *FromType, *ToType))
      diagNarrowType(SourceLoc, Lhs, Rhs);
  }
}

void NarrowingConversionsCheck::handleFloatingToIntegral(
    const ASTContext &Context, SourceLocation SourceLoc, const Expr &Lhs,
    const Expr &Rhs) {
  llvm::APFloat FloatConstant(0.0);
  if (getFloatingConstantExprValue(Context, Rhs, FloatConstant)) {
    if (!isFloatExactlyRepresentable(Context, FloatConstant, Lhs.getType()))
      return diagNarrowConstant(SourceLoc, Lhs, Rhs);

    if (PedanticMode)
      return diagConstantCast(SourceLoc, Lhs, Rhs);

    return;
  }

  const BuiltinType *FromType = getBuiltinType(Rhs);
  const BuiltinType *ToType = getBuiltinType(Lhs);
  if (isWarningInhibitedByEquivalentSize(Context, *FromType, *ToType))
    return;
  diagNarrowType(SourceLoc, Lhs, Rhs); // Assumed always lossy.
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
      bool UnusedLosesInfo = false;
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
  if (LhsType == RhsType)
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
  if (getBuiltinType(Lhs) == getBuiltinType(Rhs))
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
} // namespace clang::tidy::bugprone
