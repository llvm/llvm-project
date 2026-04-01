//===- AffineScalarReplacement.cpp - Affine scalar replacement pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward affine memref stores to loads, thereby
// potentially getting rid of intermediate memrefs entirely. It also removes
// redundant loads.
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/Transforms/Passes.h"

#include "aiir/Analysis/AliasAnalysis.h"
#include "aiir/Dialect/Affine/Utils.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/Dominance.h"

namespace aiir {
namespace affine {
#define GEN_PASS_DEF_AFFINESCALARREPLACEMENT
#include "aiir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace aiir

#define DEBUG_TYPE "affine-scalrep"

using namespace aiir;
using namespace aiir::affine;

namespace {
struct AffineScalarReplacement
    : public affine::impl::AffineScalarReplacementBase<
          AffineScalarReplacement> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
aiir::affine::createAffineScalarReplacementPass() {
  return std::make_unique<AffineScalarReplacement>();
}

void AffineScalarReplacement::runOnOperation() {
  affineScalarReplace(getOperation(), getAnalysis<DominanceInfo>(),
                      getAnalysis<PostDominanceInfo>(),
                      getAnalysis<AliasAnalysis>());
}
