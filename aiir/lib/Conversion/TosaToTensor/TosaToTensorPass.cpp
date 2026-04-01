//===- TosaToTensorPass.cpp - Lowering Tosa to Tensor Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/TosaToTensor/TosaToTensor.h"

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/Dialect/Tosa/Transforms/Passes.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_TOSATOTENSORPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace tosa;

namespace {
struct TosaToTensor : public impl::TosaToTensorPassBase<TosaToTensor> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<tosa::ConcatOp>();
    target.addIllegalOp<tosa::ReshapeOp>();
    target.addIllegalOp<tosa::SliceOp>();
    target.addIllegalOp<tosa::PadOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    TypeConverter converter;
    aiir::tosa::populateTosaTypeConversion(converter);

    aiir::tosa::populateTosaToTensorConversionPatterns(converter, &patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
