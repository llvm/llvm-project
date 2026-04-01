//===- TestLinalgFusionTransforms.cpp - Test Linalg fusion patterns -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Linalg fusion patterns.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Linalg/Transforms/Transforms.h"
#include "aiir/Dialect/SCF/Transforms/Patterns.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "aiir/Transforms/Passes.h"

using namespace aiir;
using namespace aiir::linalg;

static LogicalResult fuseLinalgOpsGreedily(func::FuncOp f) {
  OpBuilder b(f);

  // Save original Linalg ops, we only want to make a pass over those.
  SmallVector<LinalgOp, 8> linalgOps;
  f.walk([&](LinalgOp op) {
    // TODO: support multi-results.
    if (op->getNumResults() <= 1)
      linalgOps.push_back(op);
  });

  // Tile and Fuse for tensors inputs (TODO: all tensor operands).
  bool changed = false;
  for (LinalgOp linalgOp : llvm::reverse(linalgOps)) {
    for (OpOperand &opOperand : linalgOp->getOpOperands()) {
      if (isa<MemRefType>(opOperand.get().getType()))
        continue;
      if (isa<RankedTensorType>(opOperand.get().getType())) {
        // Tile and Fuse tensor input.
        if (opOperand.getOperandNumber() >= linalgOp.getNumDpsInputs())
          continue;
        auto info = fuseProducerOfTensor(b, opOperand);
        if (failed(info))
          continue;
        auto *originalOp = info->originalProducer.getOperation();
        auto *originalOpInLinalgOpsVector = llvm::find(linalgOps, originalOp);
        if (originalOpInLinalgOpsVector != linalgOps.end())
          *originalOpInLinalgOpsVector = info->fusedProducer;
        // Don't mark for erasure in the tensor case, let DCE handle this.
        changed = true;
      }
    }
  }

  return changed ? success() : failure();
}

namespace {
struct TestLinalgGreedyFusion
    : public PassWrapper<TestLinalgGreedyFusion, OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgGreedyFusion)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  StringRef getArgument() const final { return "test-linalg-greedy-fusion"; }
  StringRef getDescription() const final {
    return "Test Linalg fusion by applying a greedy test transformation.";
  }
  void runOnOperation() override {
    AIIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    patterns.add<ExtractSliceOfPadTensorSwapPattern>(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    OpPassManager pm(func::FuncOp::getOperationName());
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    do {
      (void)applyPatternsGreedily(getOperation(), frozenPatterns);
      if (failed(runPipeline(pm, getOperation())))
        this->signalPassFailure();
    } while (succeeded(fuseLinalgOpsGreedily(getOperation())));
  }
};
} // namespace

namespace aiir {
namespace test {
void registerTestLinalgGreedyFusion() {
  PassRegistration<TestLinalgGreedyFusion>();
}

} // namespace test
} // namespace aiir
