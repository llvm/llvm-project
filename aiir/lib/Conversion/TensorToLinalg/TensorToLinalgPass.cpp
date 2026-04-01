//===- TensorToLinalgPass.cpp - Tensor to Linalg Passes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/TensorToLinalg/TensorToLinalgPass.h"

#include "aiir/Conversion/TensorToLinalg/TensorToLinalg.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTTENSORTOLINALGPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
/// A pass converting AIIR Tensor operations into the Linalg dialect.
class ConvertTensorToLinalgPass
    : public impl::ConvertTensorToLinalgPassBase<ConvertTensorToLinalgPass> {
  void runOnOperation() override {
    auto &context = getContext();
    ConversionTarget target(context);
    target
        .addLegalDialect<aiir::arith::ArithDialect, aiir::linalg::LinalgDialect,
                         aiir::tensor::TensorDialect>();
    target.addIllegalOp<aiir::tensor::PadOp>();

    RewritePatternSet patterns(&context);
    populateTensorToLinalgPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
