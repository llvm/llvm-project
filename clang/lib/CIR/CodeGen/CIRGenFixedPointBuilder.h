//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper for Fixed point code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"

#include "mlir/IR/Value.h"

namespace clang::CIRGen {
// A CIR-specific generating version of the llvm::FixedPointBuilder.
struct CIRGenFixedPointBuilder {
  CIRGenFixedPointBuilder(CIRGenBuilderTy &builder, mlir::Location loc)
      : builder(builder), loc(loc) {}

  mlir::Value createFixedToFloating(mlir::Value src,
                                    const llvm::FixedPointSemantics &srcSema,
                                    mlir::Type dstTy) {
    mlir::Type opTy = getAccommodatingFloatType(dstTy, srcSema);
    // Convert the raw fixed-point value directly to floating point. If the
    // value is too large to fit, it will be rounded, not truncated.
    mlir::Value result =
        builder.createCast(loc, cir::CastKind::int_to_float, src, opTy);
    // Rescale the integral-in-floating point by the scaling factor.  This is
    // lossless, except for overflow to infinity which is unlikely.
    const llvm::fltSemantics &opSemantics =
        mlir::cast<cir::FPTypeInterface>(opTy).getFloatSemantics();
    llvm::APFloat scaleVal(
        std::pow(2.0, -static_cast<int>(srcSema.getScale())));
    bool losesInfo;
    scaleVal.convert(opSemantics, llvm::APFloat::rmNearestTiesToEven,
                     &losesInfo);
    (void)losesInfo;

    cir::ConstantOp fpConst = builder.getConstFP(loc, opTy, scaleVal);
    result = builder.createFMul(loc, result, fpConst);

    if (opTy != dstTy)
      result = builder.createFloatingCast(result, dstTy);
    return result;
  }

  mlir::Value createFloatingToFixed(mlir::Value src,
                                    const llvm::FixedPointSemantics &dstSema) {

    bool useSigned = dstSema.isSigned() || dstSema.hasUnsignedPadding();
    mlir::Value result = src;
    mlir::Type opTy = getAccommodatingFloatType(src.getType(), dstSema);

    if (opTy != src.getType())
      result = builder.createFloatingCast(result, opTy);

    // Rescale the floating point value so that its significant bits (for the
    // purposes of the conversion) are in the integral range.
    const llvm::fltSemantics &opSemantics =
        mlir::cast<cir::FPTypeInterface>(opTy).getFloatSemantics();
    llvm::APFloat scaleVal(std::pow(2.0, dstSema.getScale()));
    bool losesInfo;
    scaleVal.convert(opSemantics, llvm::APFloat::rmNearestTiesToEven,
                     &losesInfo);
    (void)losesInfo;

    cir::ConstantOp fpConst = builder.getConstFP(loc, opTy, scaleVal);
    result = builder.createFMul(loc, result, fpConst);

    cir::IntType resultTy = cir::IntType::get(
        builder.getContext(), dstSema.getWidth(), dstSema.isSigned());

    if (dstSema.isSaturated()) {
      result = builder.emitIntrinsicCallOp(
          loc, useSigned ? "fptosi.sat" : "fptoui.sat", resultTy, result);
    } else {
      result = builder.createCast(loc, cir::CastKind::float_to_int, result,
                                  resultTy);
    }

    // When saturating unsigned-with-padding using signed operations, we may
    // get negative values. Emit an extra clamp to zero.
    if (dstSema.isSaturated() && dstSema.hasUnsignedPadding()) {
      mlir::Value zero = builder.getNullValue(result.getType(), loc);
      mlir::Value isNeg =
          builder.createCompare(loc, cir::CmpOpKind::lt, result, zero);
      result = builder.createSelect(loc, isNeg, zero, result);
    }

    return result;
  }

  mlir::Value createFixedToInteger(mlir::Value src,
                                   const llvm::FixedPointSemantics &srcSema,
                                   unsigned dstWidth, bool dstIsSigned) {
    return convert(
        src, srcSema,
        llvm::FixedPointSemantics::GetIntegerSemantics(dstWidth, dstIsSigned),
        /*dstIsInteger=*/true);
  }

  mlir::Value createIntegerToFixed(mlir::Value src, unsigned srcIsSigned,
                                   const llvm::FixedPointSemantics &dstSema) {
    unsigned srcWidth;
    if (mlir::isa<cir::BoolType>(src.getType())) {
      assert(!srcIsSigned);
      srcWidth = 1;
      src = builder.createBoolToInt(
          src, cir::IntType::get(builder.getContext(), 1, /*isSigned=*/false));
    } else {
      srcWidth = mlir::cast<cir::IntType>(src.getType()).getWidth();
    }
    return convert(
        src,
        llvm::FixedPointSemantics::GetIntegerSemantics(srcWidth, srcIsSigned),
        dstSema, /*dstIsInteger=*/false);
  }

  mlir::Value createFixedToFixed(mlir::Value src,
                                 const llvm::FixedPointSemantics &srcSema,
                                 const llvm::FixedPointSemantics &dstSema) {
    return convert(src, srcSema, dstSema, /*dstIsInteger=*/false);
  }

  mlir::Value createAdd(mlir::Value lhs,
                        const llvm::FixedPointSemantics &lhsSema,
                        mlir::Value rhs,
                        const llvm::FixedPointSemantics &rhsSema) {
    auto commonSema = getCommonBinopSemantic(lhsSema, rhsSema);

    mlir::Value wideLhs = createFixedToFixed(lhs, lhsSema, commonSema);
    mlir::Value wideRhs = createFixedToFixed(rhs, rhsSema, commonSema);

    mlir::Value result;
    if (commonSema.isSaturated()) {
      result = builder.createAdd(loc, wideLhs, wideRhs,
                                 cir::OverflowBehavior::Saturated);
    } else {
      result = builder.createAdd(loc, wideLhs, wideRhs);
    }

    return createFixedToFixed(result, commonSema,
                              lhsSema.getCommonSemantics(rhsSema));
  }

  mlir::Value createSub(mlir::Value lhs,
                        const llvm::FixedPointSemantics &lhsSema,
                        mlir::Value rhs,
                        const llvm::FixedPointSemantics &rhsSema) {
    auto commonSema = getCommonBinopSemantic(lhsSema, rhsSema);

    mlir::Value wideLhs = createFixedToFixed(lhs, lhsSema, commonSema);
    mlir::Value wideRhs = createFixedToFixed(rhs, rhsSema, commonSema);

    mlir::Value result;
    if (commonSema.isSaturated()) {
      result = builder.createSub(loc, wideLhs, wideRhs,
                                 cir::OverflowBehavior::Saturated);
    } else {
      result = builder.createSub(loc, wideLhs, wideRhs);
    }

    // Subtraction can end up below 0 for padded unsigned operations, so emit
    // an extra clamp in that case.
    if (commonSema.isSaturated() && commonSema.hasUnsignedPadding()) {
      mlir::Value zero = builder.getNullValue(result.getType(), loc);
      mlir::Value ltZero =
          builder.createCompare(loc, cir::CmpOpKind::lt, result, zero);
      result = builder.createSelect(loc, ltZero, zero, result);
    }

    return createFixedToFixed(result, commonSema,
                              lhsSema.getCommonSemantics(rhsSema));
  }

  mlir::Value createMul(mlir::Value lhs,
                        const llvm::FixedPointSemantics &lhsSema,
                        mlir::Value rhs,
                        const llvm::FixedPointSemantics &rhsSema) {
    auto commonSema = getCommonBinopSemantic(lhsSema, rhsSema);
    bool useSigned = commonSema.isSigned() || commonSema.hasUnsignedPadding();

    mlir::Value wideLhs = createFixedToFixed(lhs, lhsSema, commonSema);
    mlir::Value wideRhs = createFixedToFixed(rhs, rhsSema, commonSema);

    llvm::SmallString<13> intrinId;
    cir::ConstantOp scale;

    if (useSigned) {
      intrinId = "smul.fix";
      scale = builder.getSInt32(commonSema.getScale(), loc);
    } else {
      intrinId = "umul.fix";
      scale = builder.getUInt32(commonSema.getScale(), loc);
    }

    if (commonSema.isSaturated())
      intrinId += ".sat";

    mlir::Value result =
        builder.emitIntrinsicCallOp(loc, intrinId, wideLhs.getType(),
                                    mlir::ValueRange{wideLhs, wideRhs, scale});

    return createFixedToFixed(result, commonSema,
                              lhsSema.getCommonSemantics(rhsSema));
  }

  mlir::Value createDiv(mlir::Value lhs,
                        const llvm::FixedPointSemantics &lhsSema,
                        mlir::Value rhs,
                        const llvm::FixedPointSemantics &rhsSema) {
    auto commonSema = getCommonBinopSemantic(lhsSema, rhsSema);
    bool useSigned = commonSema.isSigned() || commonSema.hasUnsignedPadding();

    mlir::Value wideLhs = createFixedToFixed(lhs, lhsSema, commonSema);
    mlir::Value wideRhs = createFixedToFixed(rhs, rhsSema, commonSema);

    llvm::SmallString<13> intrinId;
    cir::ConstantOp scale;

    if (useSigned) {
      intrinId = "sdiv.fix";
      scale = builder.getSInt32(commonSema.getScale(), loc);
    } else {
      intrinId = "udiv.fix";
      scale = builder.getUInt32(commonSema.getScale(), loc);
    }

    if (commonSema.isSaturated())
      intrinId += ".sat";

    mlir::Value result =
        builder.emitIntrinsicCallOp(loc, intrinId, wideLhs.getType(),
                                    mlir::ValueRange{wideLhs, wideRhs, scale});

    return createFixedToFixed(result, commonSema,
                              lhsSema.getCommonSemantics(rhsSema));
  }

  mlir::Value createCmp(mlir::Value lhs,
                        const llvm::FixedPointSemantics &lhsSema,
                        mlir::Value rhs,
                        const llvm::FixedPointSemantics &rhsSema,
                        cir::CmpOpKind kind) {
    auto commonSema = getCommonBinopSemantic(lhsSema, rhsSema);

    mlir::Value wideLhs = createFixedToFixed(lhs, lhsSema, commonSema);
    mlir::Value wideRhs = createFixedToFixed(rhs, rhsSema, commonSema);

    return builder.createCompare(loc, kind, wideLhs, wideRhs);
  }

  mlir::Value createShl(mlir::Value lhs,
                        const llvm::FixedPointSemantics &lhsSema,
                        mlir::Value rhs) {
    mlir::Value result;
    if (lhsSema.isSaturated()) {
      // We have to cast the RHS to the matching int type, but we have to do so
      // through unsigned so we can ensure we get zext.
      auto rhsIntTy = mlir::cast<cir::IntType>(rhs.getType());
      auto rhsUnsignedTy = cir::IntType::get(
          builder.getContext(), rhsIntTy.getWidth(), /*isSigned=*/false);

      mlir::Value rhsUnsigned =
          builder.createCast(cir::CastKind::integral, rhs, rhsUnsignedTy);
      mlir::Value rhsResized = builder.createCast(cir::CastKind::integral,
                                                  rhsUnsigned, lhs.getType());

      bool useSigned = lhsSema.isSigned() || lhsSema.hasUnsignedPadding();
      result = builder.emitIntrinsicCallOp(
          loc, useSigned ? "sshl.sat" : "ushl.sat", lhs.getType(),
          mlir::ValueRange{lhs, rhsResized});
    } else {
      result = builder.createShiftLeft(loc, lhs, rhs);
    }

    return result;
  }

  mlir::Value createShr(mlir::Value lhs, mlir::Value rhs) {
    return builder.createShiftRight(loc, lhs, rhs);
  }

private:
  mlir::Value convert(mlir::Value src, const llvm::FixedPointSemantics &srcSema,
                      const llvm::FixedPointSemantics &dstSema,
                      bool dstIsInteger) {
    unsigned srcWidth = srcSema.getWidth();
    unsigned dstWidth = dstSema.getWidth();
    unsigned srcScale = srcSema.getScale();
    unsigned dstScale = dstSema.getScale();
    bool srcIsSigned = srcSema.isSigned();
    bool dstIsSigned = dstSema.isSigned();

    mlir::Value result = src;
    unsigned resultWidth = srcWidth;

    // Downscale.
    if (dstScale < srcScale) {
      // When converting to integers, we round towards zero. For negative
      // numbers, right shifting rounds towards negative infinity. In this case,
      // we can just round up before shifting.
      if (dstIsInteger && srcIsSigned) {
        mlir::Value zero = builder.getNullValue(result.getType(), loc);
        mlir::Value isNegative =
            builder.createCompare(loc, cir::CmpOpKind::lt, result, zero);
        mlir::Value lowBits = builder.getConstAPInt(
            loc, result.getType(),
            llvm::APInt::getLowBitsSet(srcWidth, srcScale));
        mlir::Value rounded = builder.createAdd(loc, result, lowBits);
        result = builder.createSelect(loc, isNegative, rounded, result);
      }
      result = builder.createShiftRight(loc, result, srcScale - dstScale);
    }

    cir::IntType dstIntTy =
        cir::IntType::get(builder.getContext(), dstWidth, dstSema.isSigned());

    if (!dstSema.isSaturated()) {
      // Resize.
      result = builder.createIntCast(result, dstIntTy);
      // Upscale.
      if (dstScale > srcScale)
        result = builder.createShiftLeft(loc, result, dstScale - srcScale);
    } else {
      // Adjust the number of fractional bits.
      if (dstScale > srcScale) {
        // Compare to DstWidth to prevent resizing twice.
        resultWidth = std::max(srcWidth + dstScale - srcScale, dstWidth);
        cir::IntType upscaledTy =
            cir::IntType::get(builder.getContext(), resultWidth, srcIsSigned);
        result = builder.createIntCast(result, upscaledTy);
        result = builder.createShiftLeft(loc, result, dstScale - srcScale);
      }

      // Handle saturation.
      bool fewerIntBits = dstSema.getIntegralBits() < srcSema.getIntegralBits();
      if (fewerIntBits) {
        mlir::Value max = builder.getConstAPInt(
            loc, result.getType(),
            llvm::APFixedPoint::getMax(dstSema).getValue().extOrTrunc(
                resultWidth));

        mlir::Value tooHigh =
            builder.createCompare(loc, cir::CmpOpKind::gt, result, max);
        result = builder.createSelect(loc, tooHigh, max, result);
      }

      // Cannot overflow min to dest type if src is unsigned since all fixed
      // point types can cover the unsigned min of 0.
      if (srcIsSigned && (fewerIntBits || !dstIsSigned)) {
        mlir::Value min = builder.getConstAPInt(
            loc, result.getType(),
            llvm::APFixedPoint::getMin(dstSema).getValue().extOrTrunc(
                resultWidth));
        mlir::Value tooLow =
            builder.createCompare(loc, cir::CmpOpKind::lt, result, min);
        result = builder.createSelect(loc, tooLow, min, result);
      }

      // Resize the integer part to get the final destination size.
      if (resultWidth != dstWidth)
        result = builder.createIntCast(result, dstIntTy);
    }
    return result;
  }

  mlir::Type getAccommodatingFloatType(mlir::Type ty,
                                       const llvm::FixedPointSemantics &sema) {
    const llvm::fltSemantics *floatSema =
        &mlir::cast<cir::FPTypeInterface>(ty).getFloatSemantics();
    while (!sema.fitsInFloatSemantics(*floatSema))
      floatSema = llvm::APFixedPoint::promoteFloatSemantics(floatSema);
    cir::FPTypeInterface accommodating =
        cir::getFloatingPointType(*floatSema, builder.getContext());
    assert(accommodating && "no float type for semantics?");
    return accommodating;
  }
  /// Get the common semantic for two semantics, with the added imposition that
  /// saturated padded types retain the padding bit.
  llvm::FixedPointSemantics
  getCommonBinopSemantic(const llvm::FixedPointSemantics &lhsSema,
                         const llvm::FixedPointSemantics &rhsSema) {
    auto c = lhsSema.getCommonSemantics(rhsSema);
    bool bothPadded =
        lhsSema.hasUnsignedPadding() && rhsSema.hasUnsignedPadding();
    return llvm::FixedPointSemantics(
        c.getWidth() + static_cast<unsigned>(bothPadded && c.isSaturated()),
        c.getScale(), c.isSigned(), c.isSaturated(), bothPadded);
  }

  CIRGenBuilderTy &builder;
  mlir::Location loc;
};
} // namespace clang::CIRGen
