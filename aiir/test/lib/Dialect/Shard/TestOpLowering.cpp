//===- TestProcessMultiIndexOpLowering.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Shard/Transforms/Transforms.h"
#include "aiir/Dialect/Utils/IndexingUtils.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;

namespace {

struct TestAllSliceOpLoweringPass
    : public PassWrapper<TestAllSliceOpLoweringPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAllSliceOpLoweringPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    SymbolTableCollection symbolTableCollection;
    shard::populateAllSliceOpLoweringPatterns(patterns, symbolTableCollection);
    LogicalResult status =
        applyPatternsGreedily(getOperation(), std::move(patterns));
    (void)status;
    assert(succeeded(status) && "applyPatternsGreedily failed.");
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    shard::registerAllSliceOpLoweringDialects(registry);
  }
  StringRef getArgument() const final {
    return "test-grid-all-slice-op-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering of all-slice.";
  }
};

struct TestMultiIndexOpLoweringPass
    : public PassWrapper<TestMultiIndexOpLoweringPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMultiIndexOpLoweringPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    SymbolTableCollection symbolTableCollection;
    shard::populateProcessMultiIndexOpLoweringPatterns(patterns,
                                                       symbolTableCollection);
    LogicalResult status =
        applyPatternsGreedily(getOperation(), std::move(patterns));
    (void)status;
    assert(succeeded(status) && "applyPatternsGreedily failed.");
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    shard::registerProcessMultiIndexOpLoweringDialects(registry);
  }
  StringRef getArgument() const final {
    return "test-grid-process-multi-index-op-lowering";
  }
  StringRef getDescription() const final {
    return "Test lowering of shard.process_multi_index op.";
  }
};

} // namespace

namespace aiir {
namespace test {
void registerTestOpLoweringPasses() {
  PassRegistration<TestAllSliceOpLoweringPass>();
  PassRegistration<TestMultiIndexOpLoweringPass>();
}
} // namespace test
} // namespace aiir
