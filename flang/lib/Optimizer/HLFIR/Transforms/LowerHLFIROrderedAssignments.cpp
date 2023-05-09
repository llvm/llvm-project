//===- LowerHLFIROrderedAssignments.cpp - Lower HLFIR ordered assignments -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass to lower HLFIR ordered assignments.
// Ordered assignments are all the operations with the
// OrderedAssignmentTreeOpInterface that implements user defined assignments,
// assignment to vector subscripted entities, and assignments inside forall and
// where.
// The pass lowers these operations to regular hlfir.assign, loops and, if
// needed, introduces temporary storage to fulfill Fortran semantics.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_LOWERHLFIRORDEREDASSIGNMENTS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

using namespace mlir;

namespace {

class ForallOpConversion : public mlir::OpRewritePattern<hlfir::ForallOp> {
public:
  explicit ForallOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::ForallOp forallOp,
                  mlir::PatternRewriter &rewriter) const override {
    TODO(forallOp.getLoc(), "FORALL construct or statement in HLFIR");
    return mlir::failure();
  }
};

class WhereOpConversion : public mlir::OpRewritePattern<hlfir::WhereOp> {
public:
  explicit WhereOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::WhereOp whereOp,
                  mlir::PatternRewriter &rewriter) const override {
    TODO(whereOp.getLoc(), "WHERE construct or statement in HLFIR");
    return mlir::failure();
  }
};

class RegionAssignConversion
    : public mlir::OpRewritePattern<hlfir::RegionAssignOp> {
public:
  explicit RegionAssignConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::RegionAssignOp regionAssignOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!regionAssignOp.getUserDefinedAssignment().empty())
      TODO(regionAssignOp.getLoc(), "user defined assignment in HLFIR");
    else
      TODO(regionAssignOp.getLoc(),
           "assignments to vector subscripted entity in HLFIR");
    return mlir::failure();
  }
};

class LowerHLFIROrderedAssignments
    : public hlfir::impl::LowerHLFIROrderedAssignmentsBase<
          LowerHLFIROrderedAssignments> {
public:
  void runOnOperation() override {
    // Running on a ModuleOp because this pass may generate FuncOp declaration
    // for runtime calls. This could be a FuncOp pass otherwise.
    auto module = this->getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    // Patterns are only defined for the OrderedAssignmentTreeOpInterface
    // operations that can be the root of ordered assignments. The other
    // operations will be taken care of while rewriting these trees (they
    // cannot exist outside of these operations given their verifiers/traits).
    patterns
        .insert<ForallOpConversion, WhereOpConversion, RegionAssignConversion>(
            context);
    mlir::ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *op) {
      return !mlir::isa<hlfir::OrderedAssignmentTreeOpInterface>(op);
    });
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in HLFIR ordered assignments lowering pass");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> hlfir::createLowerHLFIROrderedAssignmentsPass() {
  return std::make_unique<LowerHLFIROrderedAssignments>();
}
