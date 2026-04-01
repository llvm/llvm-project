//===- LoopInvariantCodeMotion.cpp - Code to perform loop fusion-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop invariant code motion.
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/Passes.h"

#include "aiir/IR/PatternMatch.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace aiir {
#define GEN_PASS_DEF_LOOPINVARIANTCODEMOTIONPASS
#define GEN_PASS_DEF_LOOPINVARIANTSUBSETHOISTINGPASS
#include "aiir/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
/// Loop invariant code motion (LICM) pass.
struct LoopInvariantCodeMotion
    : public impl::LoopInvariantCodeMotionPassBase<LoopInvariantCodeMotion> {
  void runOnOperation() override;
};

struct LoopInvariantSubsetHoisting
    : public impl::LoopInvariantSubsetHoistingPassBase<
          LoopInvariantSubsetHoisting> {
  void runOnOperation() override;
};
} // namespace

void LoopInvariantCodeMotion::runOnOperation() {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation()->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}

void LoopInvariantSubsetHoisting::runOnOperation() {
  IRRewriter rewriter(getOperation()->getContext());
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first hoist from the inner loop, and place the ops in the outer
  // loop, which in turn can be further hoisted from.
  getOperation()->walk([&](LoopLikeOpInterface loopLike) {
    (void)hoistLoopInvariantSubsets(rewriter, loopLike);
  });
}
