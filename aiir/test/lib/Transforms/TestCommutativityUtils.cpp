//===- TestCommutativityUtils.cpp - Pass to test the commutativity utility-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tests the functionality of the commutativity utility pattern.
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/CommutativityUtils.h"

#include "TestDialect.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {

struct CommutativityUtils
    : public PassWrapper<CommutativityUtils, OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommutativityUtils)

  StringRef getArgument() const final { return "test-commutativity-utils"; }
  StringRef getDescription() const final {
    return "Test the functionality of the commutativity utility";
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    populateCommutativityUtilsPatterns(patterns);

    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

namespace aiir {
namespace test {
void registerCommutativityUtils() { PassRegistration<CommutativityUtils>(); }
} // namespace test
} // namespace aiir
