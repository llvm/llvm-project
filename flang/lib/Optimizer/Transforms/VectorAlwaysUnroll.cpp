//===- VectorAlwaysUnroll.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass tags inner loops when their outer loop has a user provided
/// vectorization attribute:(`!dir$ vector always`, `!dir$ vector length`,
/// and `!dir$ simd`).
///
/// For each such loop, this pass attaches an `llvm.loop.unroll.full` annotation
/// to every `fir.do_loop` nested within it. Fully unrolling those inner loops
/// later (in LLVM's LoopFullUnrollPass), which allows outer-loop vectorization
/// of the annotated loop.
///
/// Full unrolling of nested loops is multiplicative in code size and compile
/// time, so tagging is guarded by a cost heuristic:
///   * only loops with compile-time-constant trip counts are considered
///     (LoopFullUnrollPass cannot unroll otherwise);
///   * and the estimated unrolled op count (trip product times per-iteration
///     op count) stays within `max-unroll-ops`.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

#include <optional>

namespace fir {
#define GEN_PASS_DEF_VECTORALWAYSUNROLL
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-vector-always-unroll"

namespace {

static std::optional<llvm::APInt>
computeLoopNestTripCount(mlir::omp::LoopNestOp loopNest) {
  mlir::OperandRange lbs = loopNest.getLoopLowerBounds();
  mlir::OperandRange ubs = loopNest.getLoopUpperBounds();
  mlir::OperandRange steps = loopNest.getLoopSteps();
  bool inclusive = loopNest.getLoopInclusive();

  std::uint64_t product = 1;
  for (unsigned i = 0, e = steps.size(); i < e; ++i) {
    std::optional<llvm::APInt> count =
        fir::computeTripCount(lbs[i], ubs[i], steps[i], inclusive);
    if (!count)
      return std::nullopt;
    product = llvm::SaturatingMultiply(product, count->getZExtValue());
  }
  return llvm::APInt(64, product);
}

static std::optional<std::uint64_t> estimateBlockCost(mlir::Block &block) {
  std::uint64_t ops = 0;
  for (mlir::Operation &op : block.without_terminator()) {
    std::optional<llvm::APInt> trip;
    mlir::Block *body = nullptr;
    if (auto loop = mlir::dyn_cast<fir::DoLoopOp>(&op)) {
      trip = loop.getStaticTripCount();
      body = loop.getBody();
    } else if (auto loopNest = mlir::dyn_cast<mlir::omp::LoopNestOp>(&op)) {
      trip = computeLoopNestTripCount(loopNest);
      body = &loopNest.getRegion().front();
    }

    if (body) {
      std::optional<std::uint64_t> bodyCost = estimateBlockCost(*body);
      if (!trip || !bodyCost)
        return std::nullopt;
      ops = llvm::SaturatingAdd(
          ops, llvm::SaturatingMultiply(trip->getZExtValue(), *bodyCost));
      continue;
    }

    // Count a non-loop operation as one op
    ops = llvm::SaturatingAdd(ops, std::uint64_t{1});
    for (mlir::Region &region : op.getRegions())
      for (mlir::Block &nested : region) {
        std::optional<std::uint64_t> cost = estimateBlockCost(nested);
        if (!cost)
          return std::nullopt;
        ops = llvm::SaturatingAdd(ops, *cost);
      }
  }
  return ops;
}

class VectorAlwaysUnrollPass
    : public fir::impl::VectorAlwaysUnrollBase<VectorAlwaysUnrollPass> {
public:
  using fir::impl::VectorAlwaysUnrollBase<
      VectorAlwaysUnrollPass>::VectorAlwaysUnrollBase;

  void runOnOperation() override;

private:
  /// Tag qualifying nested inner loops with `llvm.loop.unroll.full`
  /// annotations.
  void tagNest(fir::DoLoopOp outerLoop,
               mlir::LLVM::LoopAnnotationAttr unrollAnnotation);
};

} // namespace

void VectorAlwaysUnrollPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::func::FuncOp func = getOperation();
  mlir::MLIRContext *ctx = &getContext();

  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");

  mlir::BoolAttr trueAttr = mlir::BoolAttr::get(ctx, true);
  mlir::LLVM::LoopUnrollAttr unrollFull = mlir::LLVM::LoopUnrollAttr::get(
      ctx, /*disable=*/{}, /*count=*/{}, /*runtimeDisable=*/{},
      /*full=*/trueAttr, /*followupUnrolled=*/{}, /*followupRemainder=*/{},
      /*followupAll=*/{});
  mlir::LLVM::LoopAnnotationAttr unrollAnnotation =
      mlir::LLVM::LoopAnnotationAttr::get(
          ctx, /*disableNonforced=*/{}, /*vectorize=*/{}, /*interleave=*/{},
          /*unroll=*/unrollFull, /*unrollAndJam=*/{}, /*licm=*/{},
          /*distribute=*/{}, /*pipeline=*/{}, /*peeled=*/{}, /*unswitch=*/{},
          /*mustProgress=*/{}, /*isVectorized=*/{}, /*startLoc=*/{},
          /*endLoc=*/{}, /*parallelAccesses=*/{});

  func.walk([&](fir::DoLoopOp outerLoop) {
    // Only act on loops that request vectorization. Lowering encodes
    // `!dir$ vector always`, `!dir$ vector length`, and `!dir$ simd` as a
    // loop_annotation with vectorize.enable (disable = false).
    mlir::LLVM::LoopAnnotationAttr ann = outerLoop.getLoopAnnotationAttr();
    if (!ann)
      return;
    mlir::LLVM::LoopVectorizeAttr vec = ann.getVectorize();
    if (!vec)
      return;
    mlir::BoolAttr disable = vec.getDisable();
    if (!disable || disable.getValue())
      return;
    LLVM_DEBUG(llvm::dbgs()
               << "VectorAlwaysUnroll: outer loop at " << outerLoop.getLoc()
               << " (max-unroll-ops=" << maxUnrollOps << ")\n");
    tagNest(outerLoop, unrollAnnotation);
  });

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}

void VectorAlwaysUnrollPass::tagNest(
    fir::DoLoopOp outerLoop, mlir::LLVM::LoopAnnotationAttr unrollAnnotation) {
  std::optional<std::uint64_t> estimatedOps =
      estimateBlockCost(*outerLoop.getBody());
  if (!estimatedOps) {
    LLVM_DEBUG(llvm::dbgs()
               << "  abort nest: contains a non-constant trip count loop\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "  nest cost: estimatedOps=" << *estimatedOps
                          << "\n");

  if (*estimatedOps > static_cast<std::uint64_t>(maxUnrollOps)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  estimatedOps exceeds threshold; tagging nothing\n");
    return;
  }

  // The nest is small enough: tag every nested loop for full unrolling.
  outerLoop.walk([&](fir::DoLoopOp innerLoop) {
    if (innerLoop == outerLoop)
      return;
    LLVM_DEBUG(llvm::dbgs() << "    tagging loop at " << innerLoop.getLoc()
                            << " with unroll.full\n");
    mlir::LLVM::LoopAnnotationAttr existing = innerLoop.getLoopAnnotationAttr();
    if (!existing) {
      innerLoop.setLoopAnnotationAttr(unrollAnnotation);
      return;
    }

    if (existing.getUnroll()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    keep: loop already has an unroll annotation\n");
      return;
    }
    // Append the unroll.full annotation to the existing loop_annotation
    mlir::MLIRContext *ctx = innerLoop.getContext();
    mlir::LLVM::LoopAnnotationAttr merged = mlir::LLVM::LoopAnnotationAttr::get(
        ctx, existing.getDisableNonforced(), existing.getVectorize(),
        existing.getInterleave(), /*unroll=*/unrollAnnotation.getUnroll(),
        existing.getUnrollAndJam(), existing.getLicm(),
        existing.getDistribute(), existing.getPipeline(), existing.getPeeled(),
        existing.getUnswitch(), existing.getMustProgress(),
        existing.getIsVectorized(), existing.getStartLoc(),
        existing.getEndLoc(), existing.getParallelAccesses());
    innerLoop.setLoopAnnotationAttr(merged);
  });
}
