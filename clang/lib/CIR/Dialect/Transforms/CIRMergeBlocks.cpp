//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that applies the `cir.br` canonicalization
// patterns, merging each block into its single predecessor when that
// predecessor reaches it through an unconditional branch.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CIRMERGEBLOCKS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CIRMergeBlocksPass
    : public impl::CIRMergeBlocksBase<CIRMergeBlocksPass> {
  using CIRMergeBlocksBase::CIRMergeBlocksBase;

  void runOnOperation() override;
};

void CIRMergeBlocksPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  cir::BrOp::getCanonicalizationPatterns(patterns, &getContext());

  SmallVector<Operation *> branches;
  getOperation()->walk([&](cir::BrOp br) { branches.push_back(br); });

  // Only rewrite the branches collected above. Operations moved by a block
  // merge never join the worklist, so unused values are not erased.
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  if (failed(applyOpPatternsGreedily(branches, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRMergeBlocksPass() {
  return std::make_unique<CIRMergeBlocksPass>();
}
