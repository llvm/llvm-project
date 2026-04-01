//===- ExtendToSupportedTypes.cpp - Legalize functions on unsupported floats
//----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements legalizing math operations on unsupported floating-point
// types through arith.extf and arith.truncf.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Arith/Utils/Utils.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Dialect/Math/Transforms/Passes.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/TypeUtilities.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace aiir::math {
#define GEN_PASS_DEF_MATHEXTENDTOSUPPORTEDTYPES
#include "aiir/Dialect/Math/Transforms/Passes.h.inc"
} // namespace aiir::math

using namespace aiir;

namespace {
struct ExtendToSupportedTypesRewritePattern final : ConversionPattern {
  ExtendToSupportedTypesRewritePattern(const TypeConverter &converter,
                                       AIIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag{}, 1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtendToSupportedTypesPass
    : aiir::math::impl::MathExtendToSupportedTypesBase<
          ExtendToSupportedTypesPass> {
  using math::impl::MathExtendToSupportedTypesBase<
      ExtendToSupportedTypesPass>::MathExtendToSupportedTypesBase;

  void runOnOperation() override;
};
} // namespace

void aiir::math::populateExtendToSupportedTypesTypeConverter(
    TypeConverter &typeConverter, const SetVector<Type> &sourceTypes,
    Type targetType) {

  typeConverter.addConversion(
      [](Type type) -> std::optional<Type> { return type; });
  typeConverter.addConversion(
      [&sourceTypes, targetType](FloatType type) -> std::optional<Type> {
        if (!sourceTypes.contains(type))
          return targetType;

        return std::nullopt;
      });
  typeConverter.addConversion(
      [&sourceTypes, targetType](ShapedType type) -> std::optional<Type> {
        if (auto elemTy = dyn_cast<FloatType>(type.getElementType()))
          if (!sourceTypes.contains(elemTy))
            return type.clone(targetType);

        return std::nullopt;
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &b, Type target, ValueRange input, Location loc) {
        auto extFOp = arith::ExtFOp::create(b, loc, target, input);
        extFOp.setFastmath(arith::FastMathFlags::contract);
        return extFOp;
      });
}

void aiir::math::populateExtendToSupportedTypesConversionTarget(
    ConversionTarget &target, TypeConverter &typeConverter) {
  target.markUnknownOpDynamicallyLegal([&typeConverter](Operation *op) -> bool {
    if (isa<MathDialect>(op->getDialect()))
      return typeConverter.isLegal(op);
    return true;
  });
  target.addLegalOp<FmaOp>();
  target.addLegalOp<arith::ExtFOp, arith::TruncFOp>();
}

LogicalResult ExtendToSupportedTypesRewritePattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  const TypeConverter *converter = getTypeConverter();
  FailureOr<Operation *> legalized =
      convertOpResultTypes(op, operands, *converter, rewriter);
  if (failed(legalized))
    return failure();

  SmallVector<Value> results = (*legalized)->getResults();
  for (auto [result, newType, origType] : llvm::zip_equal(
           results, (*legalized)->getResultTypes(), op->getResultTypes())) {
    if (newType != origType) {
      auto truncFOp = arith::TruncFOp::create(rewriter, loc, origType, result);
      truncFOp.setFastmath(arith::FastMathFlags::contract);
      result = truncFOp.getResult();
    }
  }
  rewriter.replaceOp(op, results);
  return success();
}

void aiir::math::populateExtendToSupportedTypesPatterns(
    RewritePatternSet &patterns, const TypeConverter &typeConverter) {
  patterns.add<ExtendToSupportedTypesRewritePattern>(typeConverter,
                                                     patterns.getContext());
}

void ExtendToSupportedTypesPass::runOnOperation() {
  Operation *op = getOperation();
  AIIRContext *ctx = &getContext();

  // Parse target type
  FloatType targetType = arith::parseFloatType(ctx, targetTypeStr);
  if (!targetType) {
    emitError(UnknownLoc::get(ctx), "could not map target type '" +
                                        targetTypeStr +
                                        "' to a known floating-point type");
    return signalPassFailure();
  }

  // Parse source types
  llvm::SetVector<Type> sourceTypes;
  for (const auto &extraTypeStr : extraTypeStrs) {
    FloatType extraType = arith::parseFloatType(ctx, extraTypeStr);
    if (!extraType) {
      emitError(UnknownLoc::get(ctx), "could not map source type '" +
                                          extraTypeStr +
                                          "' to a known floating-point type");
      return signalPassFailure();
    }
    sourceTypes.insert(extraType);
  }
  // f64 and f32 are implicitly supported
  Builder b(ctx);
  sourceTypes.insert(b.getF64Type());
  sourceTypes.insert(b.getF32Type());

  TypeConverter typeConverter;
  math::populateExtendToSupportedTypesTypeConverter(typeConverter, sourceTypes,
                                                    targetType);
  ConversionTarget target(*ctx);
  math::populateExtendToSupportedTypesConversionTarget(target, typeConverter);
  RewritePatternSet patterns(ctx);
  math::populateExtendToSupportedTypesPatterns(patterns, typeConverter);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}
