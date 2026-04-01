//===- VectorToArmSMEPass.cpp - Conversion from Vector to the ArmSME dialect =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/VectorToArmSME/VectorToArmSME.h"

#include "aiir/Dialect/ArmSME/IR/ArmSME.h"
#include "aiir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace aiir {
#define GEN_PASS_DEF_CONVERTVECTORTOARMSMEPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace aiir::vector;

namespace {
struct ConvertVectorToArmSMEPass
    : public impl::ConvertVectorToArmSMEPassBase<ConvertVectorToArmSMEPass> {

  void runOnOperation() override;
};
} // namespace

void ConvertVectorToArmSMEPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateVectorToArmSMEPatterns(patterns, getContext());

  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
