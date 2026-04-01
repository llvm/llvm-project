//===- EmulateUnsupportedFloats.cpp - Promote small floats --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass promotes small floats (of some unsupported types T) to a supported
// type U by wrapping all float operations on Ts with expansion to and
// truncation from U, then operating on U.
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/Transforms/Passes.h"

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Arith/Utils/Utils.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Location.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace aiir::arith {
#define GEN_PASS_DEF_ARITHEMULATEUNSUPPORTEDFLOATS
#include "aiir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace aiir::arith

using namespace aiir;

namespace {
struct EmulateUnsupportedFloatsPass
    : arith::impl::ArithEmulateUnsupportedFloatsBase<
          EmulateUnsupportedFloatsPass> {
  using arith::impl::ArithEmulateUnsupportedFloatsBase<
      EmulateUnsupportedFloatsPass>::ArithEmulateUnsupportedFloatsBase;

  void runOnOperation() override;
};

struct EmulateFloatPattern final : ConversionPattern {
  EmulateFloatPattern(const TypeConverter &converter, AIIRContext *ctx)
      : ConversionPattern::ConversionPattern(
            converter, Pattern::MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

LogicalResult EmulateFloatPattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (getTypeConverter()->isLegal(op))
    return failure();
  // The rewrite doesn't handle cloning regions.
  if (op->getNumRegions() != 0)
    return failure();

  Location loc = op->getLoc();
  const TypeConverter *converter = getTypeConverter();
  SmallVector<Type> resultTypes;
  if (failed(converter->convertTypes(op->getResultTypes(), resultTypes))) {
    // Note to anyone looking for this error message: this is a "can't happen".
    // If you're seeing it, there's a bug.
    return op->emitOpError("type conversion failed in float emulation");
  }
  Operation *expandedOp =
      rewriter.create(loc, op->getName().getIdentifier(), operands, resultTypes,
                      op->getAttrs(), op->getSuccessors(), /*regions=*/{});
  SmallVector<Value> newResults(expandedOp->getResults());
  for (auto [res, oldType, newType] : llvm::zip_equal(
           MutableArrayRef{newResults}, op->getResultTypes(), resultTypes)) {
    if (oldType != newType) {
      auto truncFOp = arith::TruncFOp::create(rewriter, loc, oldType, res);
      truncFOp.setFastmath(arith::FastMathFlags::contract);
      res = truncFOp.getResult();
    }
  }
  rewriter.replaceOp(op, newResults);
  return success();
}

void aiir::arith::populateEmulateUnsupportedFloatsConversions(
    TypeConverter &converter, ArrayRef<Type> sourceTypes, Type targetType) {
  converter.addConversion([sourceTypes = SmallVector<Type>(sourceTypes),
                           targetType](Type type) -> std::optional<Type> {
    if (llvm::is_contained(sourceTypes, type))
      return targetType;
    if (auto shaped = dyn_cast<ShapedType>(type))
      if (llvm::is_contained(sourceTypes, shaped.getElementType()))
        return shaped.clone(targetType);
    // All other types legal
    return type;
  });
  converter.addTargetMaterialization(
      [](OpBuilder &b, Type target, ValueRange input, Location loc) {
        auto extFOp = arith::ExtFOp::create(b, loc, target, input);
        extFOp.setFastmath(arith::FastMathFlags::contract);
        return extFOp;
      });
}

void aiir::arith::populateEmulateUnsupportedFloatsPatterns(
    RewritePatternSet &patterns, const TypeConverter &converter) {
  patterns.add<EmulateFloatPattern>(converter, patterns.getContext());
}

void aiir::arith::populateEmulateUnsupportedFloatsLegality(
    ConversionTarget &target, const TypeConverter &converter) {
  // Don't try to legalize functions and other ops that don't need expansion.
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  target.addDynamicallyLegalDialect<arith::ArithDialect>(
      [&](Operation *op) -> std::optional<bool> {
        return converter.isLegal(op);
      });
  // Manually mark arithmetic-performing vector instructions.
  target.addDynamicallyLegalOp<vector::ContractionOp, vector::ReductionOp,
                               vector::MultiDimReductionOp, vector::FMAOp,
                               vector::OuterProductOp, vector::ScanOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addLegalOp<arith::BitcastOp, arith::ExtFOp, arith::TruncFOp,
                    arith::ConstantOp, arith::SelectOp, vector::BroadcastOp>();
}

void EmulateUnsupportedFloatsPass::runOnOperation() {
  AIIRContext *ctx = &getContext();
  Operation *op = getOperation();
  SmallVector<Type> sourceTypes;
  Type targetType;

  FloatType parsedTargetType = arith::parseFloatType(ctx, targetTypeStr);
  if (!parsedTargetType) {
    emitError(UnknownLoc::get(ctx), "could not map target type '" +
                                        targetTypeStr +
                                        "' to a known floating-point type");
    return signalPassFailure();
  }
  targetType = parsedTargetType;
  for (StringRef sourceTypeStr : sourceTypeStrs) {
    FloatType sourceType = arith::parseFloatType(ctx, sourceTypeStr);
    if (!sourceType) {
      emitError(UnknownLoc::get(ctx), "could not map source type '" +
                                          sourceTypeStr +
                                          "' to a known floating-point type");
      return signalPassFailure();
    }
    sourceTypes.push_back(sourceType);
  }
  if (sourceTypes.empty())
    (void)emitOptionalWarning(
        std::nullopt,
        "no source types specified, float emulation will do nothing");

  if (llvm::is_contained(sourceTypes, targetType)) {
    emitError(UnknownLoc::get(ctx),
              "target type cannot be an unsupported source type");
    return signalPassFailure();
  }
  TypeConverter converter;
  arith::populateEmulateUnsupportedFloatsConversions(converter, sourceTypes,
                                                     targetType);
  RewritePatternSet patterns(ctx);
  arith::populateEmulateUnsupportedFloatsPatterns(patterns, converter);
  ConversionTarget target(getContext());
  arith::populateEmulateUnsupportedFloatsLegality(target, converter);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}
