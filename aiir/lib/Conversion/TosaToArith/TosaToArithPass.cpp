//===- TosaToArithPass.cpp - Lowering Tosa to Linalg Dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Arith dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/TosaToArith/TosaToArith.h"

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_TOSATOARITHPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace tosa;

namespace {
struct TosaToArith : public impl::TosaToArithPassBase<TosaToArith> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<tosa::ConstOp>();
    target.addLegalDialect<arith::ArithDialect>();

    aiir::tosa::populateTosaToArithConversionPatterns(&patterns);

    if (this->includeApplyRescale) {
      aiir::tosa::populateTosaRescaleToArithConversionPatterns(&patterns,
                                                               this->use32Bit);
      target.addIllegalOp<tosa::ApplyScaleOp>();
    }

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
