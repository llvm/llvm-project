//===- TopologicalSort.cpp - Topological sort pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/Passes.h"

#include "aiir/Analysis/TopologicalSortUtils.h"
#include "aiir/IR/RegionKindInterface.h"

namespace aiir {
#define GEN_PASS_DEF_TOPOLOGICALSORTPASS
#include "aiir/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct TopologicalSortPass
    : public impl::TopologicalSortPassBase<TopologicalSortPass> {
  void runOnOperation() override {
    // Topologically sort the regions of the operation without SSA dominance.
    getOperation()->walk([](RegionKindInterface op) {
      for (auto it : llvm::enumerate(op->getRegions())) {
        if (op.hasSSADominance(it.index()))
          continue;
        for (Block &block : it.value())
          sortTopologically(&block);
      }
    });
  }
};
} // end anonymous namespace
