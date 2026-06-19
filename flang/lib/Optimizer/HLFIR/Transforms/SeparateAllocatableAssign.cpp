//===- SeparateAllocatableAssign.cpp - Split realloc from assign ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Transform hlfir.assign with realloc semantics into a conditional
// reallocation of the LHS followed by a plain hlfir.assign (without realloc).
//
// Before:
//   hlfir.assign %rhs to %lhs realloc
//
// After:
//   %shape = shape_of(%rhs)
//   %new_lhs = genReallocIfNeeded(%lhs, %shape)  // host-side alloc
//   hlfir.assign %rhs to %new_lhs                // element copy
//
// This is useful for OpenACC/OpenMP offloading where the allocation must
// happen on the host before entering a device compute region.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace hlfir {
#define GEN_PASS_DEF_SEPARATEALLOCATABLEASSIGN
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "separate-allocatable-assign"

namespace {

class SeparateAllocatableAssignConversion
    : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  using mlir::OpRewritePattern<hlfir::AssignOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign,
                  mlir::PatternRewriter &rewriter) const override {
    if (!assign.isAllocatableAssignment())
      return rewriter.notifyMatchFailure(assign, "not an allocatable assign");

    hlfir::Entity rhs{assign.getRhs()};
    hlfir::Entity lhs{assign.getLhs()};

    if (!rhs.isArray())
      return rewriter.notifyMatchFailure(assign, "RHS is not an array");

    if (!lhs.isArray())
      return rewriter.notifyMatchFailure(assign, "LHS is not an array");

    mlir::Type rhsEleTy = rhs.getFortranElementType();
    if (!fir::isa_trivial(rhsEleTy))
      return rewriter.notifyMatchFailure(assign, "RHS type is not trivial");

    mlir::Type lhsEleTy = lhs.getFortranElementType();
    if (!fir::isa_trivial(lhsEleTy))
      return rewriter.notifyMatchFailure(assign, "LHS type is not trivial");

    if (lhsEleTy != rhsEleTy)
      return rewriter.notifyMatchFailure(assign, "element type mismatch");

    if (!fir::isBoxAddress(lhs.getType()))
      return rewriter.notifyMatchFailure(assign, "LHS is not a box address");

    mlir::Location loc = assign->getLoc();
    fir::FirOpBuilder builder(rewriter, assign.getOperation());
    builder.setInsertionPoint(assign);

    // Reallocation frees the old LHS storage. If the RHS reads that same
    // storage, the freed data would be read while producing the value to
    // assign, causing use-after-free.
    //
    // For a variable RHS, query fir::AliasAnalysis to decide whether the RHS
    // may access the LHS data and bail out if so. The aliasing question is
    // about the *data* the allocatable points to, not the descriptor address:
    // the RHS may reach the same storage through a different descriptor (e.g.
    // a pointer or a function result whose local descriptor does not alias the
    // LHS descriptor). To make the analysis reason about the data, materialize
    // a temporary load of the LHS descriptor (a loaded fir.box is a data view)
    // and use it as the LHS value in the query, then erase it.
    //
    // For an hlfir.expr RHS, the realloc is split out and the (lazy)
    // expression evaluation is left in place before it. Keeping the expression
    // evaluation from being moved across the deallocation is the
    // responsibility of the hlfir.assign lowering / expression bufferization,
    // so no aliasing analysis is performed here.
    if (!mlir::isa<hlfir::ExprType>(rhs.getType())) {
      fir::AliasAnalysis aliasAnalysis;
      auto lhsDataView = fir::LoadOp::create(builder, loc, lhs.getFirBase());
      mlir::AliasResult aliasRes =
          aliasAnalysis.alias(lhsDataView.getResult(), assign.getRhs());
      rewriter.eraseOp(lhsDataView);
      if (!aliasRes.isNo())
        return rewriter.notifyMatchFailure(assign, "LHS and RHS may alias");
    }

    LLVM_DEBUG(llvm::dbgs() << "SeparateAllocatableAssign: splitting realloc "
                               "from assign\n");

    mlir::Value rhsShape = hlfir::genShape(loc, builder, rhs);
    llvm::SmallVector<mlir::Value> rhsExtents =
        hlfir::getIndexExtents(loc, builder, rhsShape);

    // F2018 10.2.1.3: when the LHS is (re-)allocated, its lower bounds
    // come from LBOUND(rhs).  For variable RHS, extract the actual lower
    // bounds from the entity; for hlfir.expr RHS, LBOUND is always 1.
    llvm::SmallVector<mlir::Value> rhsLbounds;
    if (!mlir::isa<hlfir::ExprType>(rhs.getType())) {
      auto bounds = hlfir::genBounds(loc, builder, rhs);
      for (auto &[lb, ub] : bounds)
        rhsLbounds.push_back(lb);
    }

    fir::MutableBoxValue mutableBox(lhs.getFirBase(), /*lenParameters=*/{},
                                    /*mutableProperties=*/{});

    auto noopHandler = [](fir::ExtendedValue) {};
    llvm::SmallVector<mlir::Value> lenParams;
    fir::factory::MutableBoxReallocation realloc =
        fir::factory::genReallocIfNeeded(builder, loc, mutableBox, rhsExtents,
                                         lenParams, noopHandler);
    fir::factory::finalizeRealloc(builder, loc, mutableBox, rhsLbounds,
                                  /*takeLboundsIfRealloc=*/true, realloc);

    mlir::Value lhsBox = fir::LoadOp::create(builder, loc, lhs.getFirBase());
    hlfir::AssignOp::create(builder, loc, rhs, lhsBox,
                            /*realloc=*/false,
                            /*keep_lhs_length_if_realloc=*/false,
                            assign.isTemporaryLHS());

    rewriter.eraseOp(assign);
    return mlir::success();
  }
};

class SeparateAllocatableAssignPass
    : public hlfir::impl::SeparateAllocatableAssignBase<
          SeparateAllocatableAssignPass> {
public:
  using SeparateAllocatableAssignBase<
      SeparateAllocatableAssignPass>::SeparateAllocatableAssignBase;

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    patterns.insert<SeparateAllocatableAssignConversion>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in separate-allocatable-assign");
      signalPassFailure();
    }
  }
};
} // namespace
