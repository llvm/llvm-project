//===- ComplexToSPIRV.cpp - Complex to SPIR-V Patterns --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Complex dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "aiir/Dialect/Complex/IR/Complex.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "aiir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "aiir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "complex-to-spirv-pattern"

using namespace aiir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

struct ConstantOpPattern final : OpConversionPattern<complex::ConstantOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto spirvType =
        getTypeConverter()->convertType<ShapedType>(constOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(constOp,
                                         "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        constOp, spirvType,
        DenseElementsAttr::get(spirvType, constOp.getValue().getValue()));
    return success();
  }
};

struct CreateOpPattern final : OpConversionPattern<complex::CreateOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::CreateOp createOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(createOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(createOp,
                                         "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        createOp, spirvType, adaptor.getOperands());
    return success();
  }
};

struct ReOpPattern final : OpConversionPattern<complex::ReOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ReOp reOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(reOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(reOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        reOp, adaptor.getComplex(), llvm::ArrayRef(0));
    return success();
  }
};

struct ImOpPattern final : OpConversionPattern<complex::ImOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ImOp imOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(imOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(imOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        imOp, adaptor.getComplex(), llvm::ArrayRef(1));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void aiir::populateComplexToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  AIIRContext *context = patterns.getContext();

  patterns.add<ConstantOpPattern, CreateOpPattern, ReOpPattern, ImOpPattern>(
      typeConverter, context);
}
