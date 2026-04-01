//===- TosaToLinalgPass.cpp - Lowering Tosa to Linalg Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/TosaToLinalg/TosaToLinalg.h"

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/Dialect/Tosa/Transforms/Passes.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_TOSATOLINALGNAMED
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct TosaToLinalgNamed
    : public impl::TosaToLinalgNamedBase<TosaToLinalgNamed> {
public:
  TosaToLinalgNamed(const TosaToLinalgNamedOptions &options)
      : impl::TosaToLinalgNamedBase<TosaToLinalgNamed>(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, math::MathDialect,
                tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    TypeConverter converter;
    tosa::populateTosaTypeConversion(converter);

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, tosa::TosaDialect,
                           tensor::TensorDialect, scf::SCFDialect>();

    // Not every TOSA op can be legalized to linalg.
    target.addIllegalOp<tosa::Conv2DOp>();
    target.addIllegalOp<tosa::Conv3DOp>();
    target.addIllegalOp<tosa::DepthwiseConv2DOp>();
    target.addIllegalOp<tosa::MaxPool2dOp>();
    target.addIllegalOp<tosa::AvgPool2dOp>();
    target.addIllegalOp<tosa::MatMulOp>();
    target.addIllegalOp<tosa::TransposeOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FunctionOpInterface func = getOperation();
    TosaToLinalgNamedOptions options;
    options.preferConv2DKernelLayoutHWCF = preferConv2DKernelLayoutHWCF;
    tosa::populateTosaToLinalgNamedConversionPatterns(converter, &patterns,
                                                      options);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
aiir::tosa::createTosaToLinalgNamed(const TosaToLinalgNamedOptions &options) {
  return std::make_unique<TosaToLinalgNamed>(options);
}
