//===- TosaToMLProgramPass.cpp - Lowering Tosa to MLProgram Dialect--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes the TOSA dialect to the MLProgram dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/TosaToMLProgram/TosaToMLProgram.h"
#include "aiir/Dialect/MLProgram/IR/MLProgram.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_TOSATOMLPROGRAM
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace tosa;

namespace {
struct TosaToMLProgram : public impl::TosaToMLProgramBase<TosaToMLProgram> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    auto moduleOp = getOperation();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addIllegalOp<tosa::VariableOp, tosa::VariableReadOp,
                        tosa::VariableWriteOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    aiir::tosa::populateTosaToMLProgramConversionPatterns(&patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
