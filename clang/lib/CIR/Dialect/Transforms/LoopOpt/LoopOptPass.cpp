//===- LoopOptPass.cpp - Driver for CIR loop optimizations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detection layer for CIR loop optimizations. Transforms plug into
// runOnOperation below.
//
//===----------------------------------------------------------------------===//

#include "LoopNestPattern.h"

#include "../PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/Support/TimeProfiler.h"

using namespace mlir;
using namespace cir::loopopt;

namespace mlir {
#define GEN_PASS_DEF_LOOPOPT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

constexpr llvm::StringLiteral kTestAttr = "cir.loopopt.test";

llvm::StringRef getPatternName(LoopNestPatternKind kind) {
  switch (kind) {
  case LoopNestPatternKind::InnerAffineUpper:
    return "inner_affine_upper";
  case LoopNestPatternKind::OuterIVInnerStart:
    return "outer_iv_inner_start";
  case LoopNestPatternKind::InnerProductUpper:
    return "inner_product_upper";
  case LoopNestPatternKind::InvariantInnerStart:
    return "invariant_inner_start";
  }
  llvm_unreachable("unhandled LoopNestPatternKind");
}

struct LoopOptPass : public impl::LoopOptBase<LoopOptPass> {
  using LoopOptBase::LoopOptBase;
  void runOnOperation() override;
};

} // namespace

void LoopOptPass::runOnOperation() {
  // Nothing consumes the recognized nests yet, so outside the test option
  // there is no reason to walk the loops.
  // FIXME: Transforms run here.
  if (!testAnnotate) {
    markAllAnalysesPreserved();
    return;
  }

  llvm::TimeTraceScope scope("Loop Nest Classification");

  // The matcher inspects the invariant start's defining store, which may sit
  // outside both loops, so this dominance info has to cover the whole root.
  mlir::DominanceInfo dominance(getOperation());
  mlir::Builder builder(&getContext());
  // Affine parameters are recorded at the width of the loop counter they came
  // from, because a counter wider than 64 bits may have parameters that do not
  // fit in an i64.
  auto counterWidthAttr = [&](const llvm::APSInt &value) {
    return builder.getIntegerAttr(builder.getIntegerType(value.getBitWidth()),
                                  value);
  };

  // Attribute updates are safe during this walk. A mutating consumer must
  // re-match or select non-overlapping nests because adjacent pairs share a
  // loop.
  getOperation()->walk([&](cir::ForOp forOp) {
    mlir::FailureOr<CountedLoop> outer = matchCountedLoop(forOp);
    if (mlir::failed(outer))
      return;
    cir::ForOp innerFor = getSingleInnerFor(forOp);
    if (!innerFor)
      return;
    mlir::FailureOr<CountedLoop> inner = matchCountedLoop(innerFor);
    if (mlir::failed(inner))
      return;
    mlir::FailureOr<LoopNestPattern> nest =
        matchLoopNestPattern(*outer, *inner, dominance);
    if (mlir::failed(nest))
      return;

    llvm::SmallVector<mlir::NamedAttribute, 3> fields;
    fields.emplace_back(builder.getStringAttr("kind"),
                        builder.getStringAttr(getPatternName(nest->kind)));
    if (nest->affine) {
      fields.emplace_back(builder.getStringAttr("scale"),
                          counterWidthAttr(nest->affine->coefficient));
      fields.emplace_back(builder.getStringAttr("offset"),
                          counterWidthAttr(nest->affine->offset));
    }
    forOp->setAttr(kTestAttr, builder.getDictionaryAttr(fields));
  });
}

std::unique_ptr<Pass> mlir::createLoopOptPass() {
  return std::make_unique<LoopOptPass>();
}
