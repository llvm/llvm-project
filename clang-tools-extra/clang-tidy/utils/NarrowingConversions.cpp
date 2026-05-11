//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NarrowingConversions.h"

namespace clang::tidy::utils {

// [dcl.init.list]/p7.2: long double -> double or float, or double -> float;
// unless the source is a constant expression whose value fits in the target
// representation range.
static bool isFloatToFloatNarrowing(QualType From, QualType To,
                                    const Expr *Init, const ASTContext &Ctx) {
  if (Ctx.getFloatingTypeOrder(From, To) <= 0)
    return false;
  if (Init->isValueDependent())
    return true;
  Expr::EvalResult EvalResult;
  if (!Init->EvaluateAsRValue(EvalResult, Ctx) || !EvalResult.Val.isFloat())
    return true;
  llvm::APFloat Value = EvalResult.Val.getFloat();
  bool LosesInfo = false;
  const llvm::APFloat::opStatus Status =
      Value.convert(Ctx.getFloatTypeSemantics(To),
                    llvm::APFloat::rmNearestTiesToEven, &LosesInfo);
  return (Status & llvm::APFloat::opOverflow) != 0;
}

// [dcl.init.list]/p7.3: integer or unscoped enum to floating-point; unless
// the source is a constant expression that fits and round-trips exactly.
static bool isIntToFloatNarrowing(const Expr *Init, QualType To,
                                  const ASTContext &Ctx) {
  if (Init->isValueDependent())
    return true;
  std::optional<llvm::APSInt> OptVal = Init->getIntegerConstantExpr(Ctx);
  if (!OptVal)
    return true;
  llvm::APFloat FPVal(Ctx.getFloatTypeSemantics(To));
  const llvm::APFloat::opStatus Status = FPVal.convertFromAPInt(
      *OptVal, OptVal->isSigned(), llvm::APFloat::rmNearestTiesToEven);
  if (Status != llvm::APFloat::opOK)
    return true;
  llvm::APSInt RoundTrip(OptVal->getBitWidth(), !OptVal->isSigned());
  bool IsExact = false;
  FPVal.convertToInteger(RoundTrip, llvm::APFloat::rmTowardZero, &IsExact);
  return !IsExact || RoundTrip != *OptVal;
}

// [dcl.init.list]/p7.4: integer or unscoped enum to integer that cannot
// represent all values; unless source is a constant expression that fits.
static bool isIntToIntNarrowing(QualType From, QualType To, const Expr *Init,
                                const ASTContext &Ctx) {
  const bool FromSigned = From->isSignedIntegerOrEnumerationType();
  const unsigned FromWidth = Ctx.getIntWidth(From);
  const bool ToSigned = To->isSignedIntegerOrEnumerationType();
  const unsigned ToWidth = Ctx.getIntWidth(To);

  if ((FromWidth < ToWidth + (FromSigned == ToSigned)) &&
      !(FromSigned && !ToSigned))
    return false;

  if (Init->isValueDependent())
    return true;
  std::optional<llvm::APSInt> OptVal = Init->getIntegerConstantExpr(Ctx);
  if (!OptVal)
    return true;
  const llvm::APSInt Val = OptVal->extend(OptVal->getBitWidth() + 1);
  llvm::APSInt Converted = Val;
  Converted = Converted.trunc(ToWidth);
  Converted.setIsSigned(ToSigned);
  Converted = Converted.extend(Val.getBitWidth());
  Converted.setIsSigned(Val.isSigned());
  return Converted != Val;
}

bool isNarrowingConversion(QualType From, QualType To, const Expr *Init,
                           const ASTContext &Ctx) {
  From = From.getCanonicalType().getUnqualifiedType();
  To = To.getCanonicalType().getUnqualifiedType();
  if (From == To)
    return false;

  if (To->isBooleanType() &&
      (From->isPointerType() || From->isMemberPointerType()))
    return true;

  if (From->isRealFloatingType() && To->isIntegralOrEnumerationType())
    return true;

  if (From->isRealFloatingType() && To->isRealFloatingType())
    return isFloatToFloatNarrowing(From, To, Init, Ctx);

  if (From->isIntegralOrUnscopedEnumerationType() && To->isRealFloatingType())
    return isIntToFloatNarrowing(Init, To, Ctx);

  if (From->isIntegralOrUnscopedEnumerationType() &&
      To->isIntegralOrUnscopedEnumerationType())
    return isIntToIntNarrowing(From, To, Init, Ctx);

  return false;
}

} // namespace clang::tidy::utils
