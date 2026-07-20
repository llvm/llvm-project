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
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

static std::optional<std::uint64_t> estimateUnrolledCost(fir::DoLoopOp loop) {
  std::optional<llvm::APInt> trip = loop.getStaticTripCount();
  if (!trip)
    return std::nullopt;
  std::uint64_t tripCount = trip->getZExtValue();

  std::uint64_t bodyOps = 0;
  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    auto nested = mlir::dyn_cast<fir::DoLoopOp>(&op);
    if (!nested) {
      // Count a non-loop operation counts as one op
      bodyOps = llvm::SaturatingAdd(bodyOps, std::uint64_t{1});
      continue;
    }
    std::optional<std::uint64_t> childCost = estimateUnrolledCost(nested);
    if (!childCost)
      return std::nullopt;
    bodyOps = llvm::SaturatingAdd(bodyOps, *childCost);
  }

  return llvm::SaturatingMultiply(tripCount, bodyOps);
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
  std::uint64_t estimatedOps = 0;
  for (mlir::Operation &op : outerLoop.getBody()->without_terminator()) {
    auto nested = mlir::dyn_cast<fir::DoLoopOp>(&op);
    if (!nested)
      continue;
    std::optional<std::uint64_t> cost = estimateUnrolledCost(nested);
    if (!cost) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  abort nest: contains a non-constant trip count loop\n");
      return;
    }
    estimatedOps = llvm::SaturatingAdd(estimatedOps, *cost);
  }

  LLVM_DEBUG(llvm::dbgs() << "  nest cost: estimatedOps=" << estimatedOps
                          << "\n");

  if (estimatedOps > static_cast<std::uint64_t>(maxUnrollOps)) {
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
