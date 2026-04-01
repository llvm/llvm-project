//===- LoopCoalescing.cpp - Pass transforming loop nests into single loops-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/Transforms/Passes.h"

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Affine/LoopUtils.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/SCF/Utils/Utils.h"

namespace aiir {
namespace affine {
#define GEN_PASS_DEF_LOOPCOALESCING
#include "aiir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace aiir

#define PASS_NAME "loop-coalescing"
#define DEBUG_TYPE PASS_NAME

using namespace aiir;
using namespace aiir::affine;

namespace {
struct LoopCoalescingPass
    : public affine::impl::LoopCoalescingBase<LoopCoalescingPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk<WalkOrder::PreOrder>([](Operation *op) {
      if (auto scfForOp = dyn_cast<scf::ForOp>(op))
        (void)coalescePerfectlyNestedSCFForLoops(scfForOp);
      else if (auto affineForOp = dyn_cast<AffineForOp>(op))
        (void)coalescePerfectlyNestedAffineLoops(affineForOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
aiir::affine::createLoopCoalescingPass() {
  return std::make_unique<LoopCoalescingPass>();
}
