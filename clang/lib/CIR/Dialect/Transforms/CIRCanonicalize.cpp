//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass that canonicalizes CIR operations, eliminating
// redundant branches, empty scopes, and other unnecessary operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/Block.h"
#include "aiir/IR/Operation.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/Region.h"
#include "aiir/Support/LogicalResult.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace aiir;
using namespace cir;

namespace aiir {
#define GEN_PASS_DEF_CIRCANONICALIZE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace aiir

namespace {

//===----------------------------------------------------------------------===//
// CIRCanonicalizePass
//===----------------------------------------------------------------------===//

struct CIRCanonicalizePass
    : public impl::CIRCanonicalizeBase<CIRCanonicalizePass> {
  using CIRCanonicalizeBase::CIRCanonicalizeBase;

  // The same operation rewriting done here could have been performed
  // by CanonicalizerPass (adding hasCanonicalizer for target Ops and
  // implementing the same from above in CIRDialects.cpp). However, it's
  // currently too aggressive for static analysis purposes, since it might
  // remove things where a diagnostic can be generated.
  //
  // FIXME: perhaps we can add one more mode to GreedyRewriteConfig to
  // disable this behavior.
  void runOnOperation() override;
};

void CIRCanonicalizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  // Collect canonicalization patterns from CIR ops.
  aiir::Dialect *cir = getContext().getLoadedDialect<cir::CIRDialect>();
  for (aiir::RegisteredOperationName op :
       getContext().getRegisteredOperations())
    if (&op.getDialect() == cir)
      op.getCanonicalizationPatterns(patterns, &getContext());

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    assert(!cir::MissingFeatures::tryOp());
    assert(!cir::MissingFeatures::callOp());

    // Many operations are here to perform a manual `fold` in
    // applyOpPatternsGreedily.
    if (isa<BrOp, BrCondOp, CastOp, ScopeOp, SwitchOp, SelectOp, IncOp, DecOp,
            MinusOp, NotOp, AddOp, MulOp, AndOp, OrOp, XorOp, MaxOp, MinOp,
            ComplexCreateOp, ComplexImagOp, ComplexRealOp, VecCmpOp,
            VecCreateOp, VecExtractOp, VecShuffleOp, VecShuffleDynamicOp,
            VecTernaryOp, BitClrsbOp, BitClzOp, BitCtzOp, BitFfsOp, BitParityOp,
            BitPopcountOp, BitReverseOp, ByteSwapOp, RotateOp, ConstantOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> aiir::createCIRCanonicalizePass() {
  return std::make_unique<CIRCanonicalizePass>();
}
