//===- InlineHLFIRAssign.cpp - Inline hlfir.assign ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Transform hlfir.assign array operations into loop nests performing element
// per element assignments. The inlining is done for trivial data types always,
// though, we may add performance/code-size heuristics in future.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Analysis/ArraySectionAnalyzer.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

namespace hlfir {
#define GEN_PASS_DEF_INLINEHLFIRASSIGN
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "inline-hlfir-assign"

static llvm::cl::opt<bool> inlineAllocatableExprAssignFlag(
    "inline-hlfir-allocatable-expr-assign",
    llvm::cl::desc("Enable inlining of allocatable assignments when RHS is an "
                   "hlfir.expr (e.g., from hlfir.elemental)"),
    llvm::cl::init(false));

namespace {
/// Expand hlfir.assign of array RHS to array LHS into a loop nest
/// of element-by-element assignments:
///   hlfir.assign %4 to %5 : !fir.ref<!fir.array<3x3xf32>>,
///                           !fir.ref<!fir.array<3x3xf32>>
/// into:
///   fir.do_loop %arg1 = %c1 to %c3 step %c1 unordered {
///     fir.do_loop %arg2 = %c1 to %c3 step %c1 unordered {
///       %6 = hlfir.designate %4 (%arg2, %arg1)  :
///           (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
///       %7 = fir.load %6 : !fir.ref<f32>
///       %8 = hlfir.designate %5 (%arg2, %arg1)  :
///           (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
///       hlfir.assign %7 to %8 : f32, !fir.ref<f32>
///     }
///   }
///
/// The transformation is correct only when LHS and RHS do not alias.
/// When RHS is an array expression, then there is no aliasing.
/// This transformation does not support runtime checking for
/// non-conforming LHS/RHS arrays' shapes currently.
class InlineHLFIRAssignConversion
    : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  using mlir::OpRewritePattern<hlfir::AssignOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign,
                  mlir::PatternRewriter &rewriter) const override {
    if (assign.isAllocatableAssignment())
      return rewriter.notifyMatchFailure(assign,
                                         "AssignOp may imply allocation");

    hlfir::Entity rhs{assign.getRhs()};

    if (!rhs.isArray())
      return rewriter.notifyMatchFailure(assign,
                                         "AssignOp's RHS is not an array");

    mlir::Type rhsEleTy = rhs.getFortranElementType();
    if (!fir::isa_trivial(rhsEleTy))
      return rewriter.notifyMatchFailure(
          assign, "AssignOp's RHS data type is not trivial");

    hlfir::Entity lhs{assign.getLhs()};
    if (!lhs.isArray())
      return rewriter.notifyMatchFailure(assign,
                                         "AssignOp's LHS is not an array");

    mlir::Type lhsEleTy = lhs.getFortranElementType();
    if (!fir::isa_trivial(lhsEleTy))
      return rewriter.notifyMatchFailure(
          assign, "AssignOp's LHS data type is not trivial");

    if (lhsEleTy != rhsEleTy)
      return rewriter.notifyMatchFailure(assign,
                                         "RHS/LHS element types mismatch");

    if (!mlir::isa<hlfir::ExprType>(rhs.getType())) {
      // If RHS is not an hlfir.expr, then we should prove that
      // LHS and RHS do not alias.
      // TODO: if they may alias, we can insert hlfir.as_expr for RHS,
      // and proceed with the inlining.
      fir::AliasAnalysis aliasAnalysis;
      mlir::AliasResult aliasRes = aliasAnalysis.alias(lhs, rhs);
      if (!aliasRes.isNo()) {
        // Alias analysis reports potential aliasing, but we can use
        // ArraySectionAnalyzer to check if the slices are disjoint
        // or identical (which is safe for element-wise assignment).
        fir::ArraySectionAnalyzer::SlicesOverlapKind overlap =
            fir::ArraySectionAnalyzer::analyze(lhs, rhs);
        if (overlap == fir::ArraySectionAnalyzer::SlicesOverlapKind::Unknown) {
          LLVM_DEBUG(llvm::dbgs() << "InlineHLFIRAssign:\n"
                                  << "\tLHS: " << lhs << "\n"
                                  << "\tRHS: " << rhs << "\n"
                                  << "\tALIAS: " << aliasRes << "\n");
          return rewriter.notifyMatchFailure(assign, "RHS/LHS may alias");
        }
      }
    }

    mlir::Location loc = assign->getLoc();
    fir::FirOpBuilder builder(rewriter, assign.getOperation());
    builder.setInsertionPoint(assign);
    mlir::ArrayAttr accessGroups;
    if (auto attrs = assign.getOperation()->getAttrOfType<mlir::ArrayAttr>(
            fir::getAccessGroupsAttrName()))
      accessGroups = attrs;
    hlfir::genNoAliasArrayAssignment(
        loc, builder, rhs, lhs, flangomp::shouldUseWorkshareLowering(assign),
        /*temporaryLHS=*/false, /*combiner=*/nullptr, accessGroups);
    rewriter.eraseOp(assign);
    return mlir::success();
  }
};

/// Expand hlfir.assign of hlfir.expr RHS to allocatable LHS.
/// When RHS is an hlfir.expr (e.g., from hlfir.elemental), there is no
/// aliasing concern because expressions don't represent memory locations.
/// This allows us to inline the assignment even for allocatables.
///
/// The generated code:
/// 1. Gets the shape from the RHS expression
/// 2. Uses genReallocIfNeeded to handle allocation/reallocation properly
/// 3. Generates a loop nest to assign elements (via storage handler callback)
/// 4. Finalizes the reallocation
///
/// Example transformation for: allocatable_array = elemental_expr
///   hlfir.assign %expr to %alloc realloc : !hlfir.expr<?xf64>,
///                                          !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
/// into:
///   // Check allocation status and reallocate if needed
///   // ... (genReallocIfNeeded handles this) ...
///   // Loop over elements
///   fir.do_loop %i = %c1 to %extent step %c1 unordered {
///     %rhs_val = hlfir.apply %expr, %i : ...
///     %lhs_elem = hlfir.designate %lhs_box (%i) : ...
///     hlfir.assign %rhs_val to %lhs_elem : f64, !fir.ref<f64>
///   }
class InlineAllocatableExprAssignConversion
    : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  using mlir::OpRewritePattern<hlfir::AssignOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign,
                  mlir::PatternRewriter &rewriter) const override {
    // This pattern only handles allocatable assignments
    if (!assign.isAllocatableAssignment())
      return rewriter.notifyMatchFailure(
          assign, "AssignOp is not an allocatable assignment");

    hlfir::Entity rhs{assign.getRhs()};
    hlfir::Entity lhs{assign.getLhs()};

    // RHS must be an hlfir.expr (this is the key condition - no aliasing)
    if (!mlir::isa<hlfir::ExprType>(rhs.getType()))
      return rewriter.notifyMatchFailure(
          assign,
          "RHS is not an hlfir.expr - cannot inline allocatable assign");

    // RHS must be an array
    if (!rhs.isArray())
      return rewriter.notifyMatchFailure(assign,
                                         "AssignOp's RHS is not an array");

    // Check element types are trivial and match
    mlir::Type rhsEleTy = rhs.getFortranElementType();
    if (!fir::isa_trivial(rhsEleTy))
      return rewriter.notifyMatchFailure(
          assign, "AssignOp's RHS data type is not trivial");

    mlir::Type lhsEleTy = lhs.getFortranElementType();
    if (!fir::isa_trivial(lhsEleTy))
      return rewriter.notifyMatchFailure(
          assign, "AssignOp's LHS data type is not trivial");

    if (lhsEleTy != rhsEleTy)
      return rewriter.notifyMatchFailure(assign,
                                         "RHS/LHS element types mismatch");

    // LHS must be a reference to a box (allocatable)
    mlir::Type lhsType = lhs.getType();
    if (!fir::isBoxAddress(lhsType))
      return rewriter.notifyMatchFailure(assign,
                                         "LHS is not a reference to a box");

    LLVM_DEBUG(llvm::dbgs()
               << "InlineHLFIRAssign: inlining allocatable expr assignment\n");

    mlir::Location loc = assign->getLoc();
    fir::FirOpBuilder builder(rewriter, assign.getOperation());
    builder.setInsertionPoint(assign);

    // Get the shape of the RHS expression
    mlir::Value rhsShape = hlfir::genShape(loc, builder, rhs);
    llvm::SmallVector<mlir::Value> rhsExtents =
        hlfir::getIndexExtents(loc, builder, rhsShape);

    // Create a MutableBoxValue for the LHS allocatable
    mlir::Value lhsBoxRef = lhs.getFirBase();

    // Create MutableBoxValue - for trivial types, no length params needed
    fir::MutableBoxValue mutableBox(lhsBoxRef, /*lenParameters=*/{},
                                    /*mutableProperties=*/{});

    // Use genReallocIfNeeded to handle allocation/reallocation properly.
    // This implements Fortran 10.2.1.3 point 3:
    // - If not allocated, allocate with RHS shape
    // - If allocated with same shape, keep existing allocation
    // - If allocated with different shape, reallocate
    //
    // The storage handler callback performs the actual assignment loop.
    bool useWorkshare = flangomp::shouldUseWorkshareLowering(assign);
    auto storageHandler = [&](fir::ExtendedValue storage) {
      hlfir::Entity lhsEntity{
          fir::getBase(fir::factory::createBoxValue(builder, loc, storage))};

      llvm::SmallVector<mlir::Value> extents =
          fir::factory::getExtents(loc, builder, storage);

      // Generate loop nest to assign elements
      hlfir::LoopNest loopNest = hlfir::genLoopNest(
          loc, builder, extents, /*isUnordered=*/true, useWorkshare);
      builder.setInsertionPointToStart(loopNest.body);

      // Get RHS element via hlfir.apply
      hlfir::Entity rhsElement =
          hlfir::getElementAt(loc, builder, rhs, loopNest.oneBasedIndices);
      rhsElement = hlfir::loadTrivialScalar(loc, builder, rhsElement);

      // Get LHS element
      hlfir::Entity lhsElement = hlfir::getElementAt(loc, builder, lhsEntity,
                                                     loopNest.oneBasedIndices);

      // Assign the element (scalar, non-allocatable)
      hlfir::AssignOp::create(builder, loc, rhsElement, lhsElement,
                              /*realloc=*/false,
                              /*keep_lhs_length_if_realloc=*/false,
                              /*temporary_lhs=*/false);

      // Restore insertion point after loop
      builder.setInsertionPointAfter(loopNest.outerOp);
    };

    // No length params for trivial types
    llvm::SmallVector<mlir::Value> lenParams;

    // Generate reallocation logic with assignment in the callback
    fir::factory::MutableBoxReallocation realloc =
        fir::factory::genReallocIfNeeded(builder, loc, mutableBox, rhsExtents,
                                         lenParams, storageHandler);

    // Finalize: free old storage if reallocated and update the mutable box
    fir::factory::finalizeRealloc(builder, loc, mutableBox, /*lbounds=*/{},
                                  /*takeLboundsIfRealloc=*/true, realloc);

    // Erase the original assign
    rewriter.eraseOp(assign);
    return mlir::success();
  }
};

class InlineHLFIRAssignPass
    : public hlfir::impl::InlineHLFIRAssignBase<InlineHLFIRAssignPass> {
public:
  using InlineHLFIRAssignBase<InlineHLFIRAssignPass>::InlineHLFIRAssignBase;

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    patterns.insert<InlineHLFIRAssignConversion>(context);

    // Optionally add the allocatable expr assignment pattern
    if (inlineAllocatableExprAssignFlag) {
      LLVM_DEBUG(llvm::dbgs()
                 << "InlineHLFIRAssign: enabling allocatable expr assignment "
                    "inlining\n");
      patterns.insert<InlineAllocatableExprAssignConversion>(context);
    }

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in hlfir.assign inlining");
      signalPassFailure();
    }
  }
};
} // namespace
