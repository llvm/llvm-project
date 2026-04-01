//===- TosaToSCFPass.cpp - Lowering Tosa to SCF Dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/TosaToSCF/TosaToSCF.h"

#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/DialectConversion.h"

namespace aiir {
#define GEN_PASS_DEF_TOSATOSCFPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace tosa;

namespace {
struct TosaToSCF : public impl::TosaToSCFPassBase<TosaToSCF> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<tensor::TensorDialect, scf::SCFDialect>();
    target.addIllegalOp<tosa::IfOp, tosa::ScatterOp, tosa::WhileOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    auto *op = getOperation();
    aiir::tosa::populateTosaToSCFConversionPatterns(&patterns);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void aiir::tosa::addTosaToSCFPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createTosaToSCFPass());
}
