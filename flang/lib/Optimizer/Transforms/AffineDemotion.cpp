//===-- AffineDemotion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation is a prototype that demote affine dialects operations
// after optimizations to FIR loops operations.
// It is used after the AffinePromotion pass.
// It is not part of the production pipeline and would need more work in order
// to be used in production.
// More information can be found in this presentation:
// https://slides.com/rajanwalia/deck
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

namespace fir {
#define GEN_PASS_DEF_AFFINEDIALECTDEMOTION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-affine-demotion"

using namespace fir;
using namespace mlir;

namespace {

/// Check whether the FIR base reference points to an array with
/// dynamic (runtime-determined) extents, e.g. `!fir.ref<!fir.array<?x?xf32>>`.
/// `fir.coordinate_of` cannot handle such arrays because it needs
/// compile-time-known dimensions to linearise the multi-dimensional index.
static bool baseHasDynamicExtents(mlir::Value base) {
  mlir::Type ty = base.getType();
  if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(ty))
    ty = refTy.getEleTy();
  else if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
    ty = heapTy.getEleTy();
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
    return seqTy.hasDynamicExtents();
  return false;
}

/// Convert 0-based memref indices (already reversed to column-major order)
/// to Fortran indices expected by fir.array_coor.
///
/// For fir.shape (implicit lb=1):    Fortran_idx = 0based + 1
/// For fir.shape_shift (explicit lb): Fortran_idx = 0based + lb_k
static SmallVector<Value>
toFortranIndices(mlir::Value shape, ArrayRef<Value> zeroBasedIndices,
                 mlir::Location loc, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> result;

  if (auto shapeShiftOp = shape.getDefiningOp<fir::ShapeShiftOp>()) {
    auto pairs = shapeShiftOp.getPairs();
    for (unsigned k = 0; k < zeroBasedIndices.size(); ++k) {
      mlir::Value lb = pairs[k * 2]; // lower bound for dimension k
      result.push_back(
          arith::AddIOp::create(rewriter, loc, zeroBasedIndices[k], lb));
    }
  } else {
    // fir.shape or anything else — lower bound is 1
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    for (auto idx : zeroBasedIndices)
      result.push_back(arith::AddIOp::create(rewriter, loc, idx, one));
  }
  return result;
}

/// Build a `fir.array_coor` that addresses a box-typed array from
/// 0-based affine indices, honoring the descriptor's lower bounds.
///
/// Each dimension's `lb` is read at runtime via `fir.box_dims`,
/// packed into a `fir.shift`, and added to the 0-based affine index to
/// form the Fortran-visible index.
static fir::ArrayCoorOp
buildBoxArrayCoor(mlir::Location loc, ConversionPatternRewriter &rewriter,
                  mlir::Type resultRefTy, mlir::Value boxBase,
                  mlir::ArrayRef<mlir::Value> zeroBasedIndices) {
  auto idxTy = rewriter.getIndexType();
  unsigned rank = zeroBasedIndices.size();

  SmallVector<mlir::Value> lbs;
  lbs.reserve(rank);
  for (unsigned k = 0; k < rank; ++k) {
    auto dim =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(k));
    auto boxDims = fir::BoxDimsOp::create(rewriter, loc, idxTy, idxTy, idxTy,
                                          boxBase, dim);
    lbs.push_back(boxDims.getLowerBound());
  }

  auto shiftTy = fir::ShiftType::get(rewriter.getContext(), rank);
  auto shiftOp = fir::ShiftOp::create(rewriter, loc, shiftTy, lbs);

  SmallVector<mlir::Value> fortranIndices;
  fortranIndices.reserve(rank);
  for (unsigned k = 0; k < rank; ++k)
    fortranIndices.push_back(
        arith::AddIOp::create(rewriter, loc, zeroBasedIndices[k], lbs[k]));

  return fir::ArrayCoorOp::create(rewriter, loc, resultRefTy, boxBase,
                                  /*shape=*/shiftOp.getResult(),
                                  /*slice=*/mlir::Value{}, fortranIndices,
                                  /*typeparams=*/mlir::ValueRange{});
}

/// Walk backwards from `base` to locate the `fir.shape` (or shapeshift)
/// that carries the runtime dimension sizes.
///
/// Handles three cases:
///   1. Explicit-shape arrays: base is from fir.declare → shape is attached.
///   2. Local allocatable arrays: base is from fir.allocmem → find the
///      fir.embox that wraps it and recover the shape from there.
///   3. Allocatable dummy / module arrays: base is from fir.box_addr →
///      use the original fir.box directly with fir.array_coor (the box
///      carries all shape info) pass fir.shift by reading the box discriptor
///      using fir.box_dims. In this case `outBoxBase` is set to the
///      box value and the returned shape is null.
static mlir::Value findShapeForBase(mlir::Value base, mlir::Value &outBoxBase) {
  outBoxBase = mlir::Value{};

  // Case 1: explicit-shape via fir.declare
  if (auto declareOp = base.getDefiningOp<fir::DeclareOp>())
    return declareOp.getShape();

  // Case 2: local allocatable — find fir.embox that wraps this heap pointer
  if (base.getDefiningOp<fir::AllocMemOp>()) {
    for (auto *user : base.getUsers()) {
      if (auto embox = mlir::dyn_cast<fir::EmboxOp>(user))
        return embox.getShape();
    }
  }

  // Case 3: allocatable dummy arg / module array — base is from fir.box_addr
  if (auto boxAddr = base.getDefiningOp<fir::BoxAddrOp>()) {
    outBoxBase = boxAddr.getVal();
    return mlir::Value{};
  }

  return mlir::Value{};
}

class AffineLoadConversion
    : public OpConversionPattern<mlir::affine::AffineLoadOp> {
public:
  using OpConversionPattern<mlir::affine::AffineLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::affine::AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> indices(adaptor.getIndices());
    auto maybeExpandedMap = affine::expandAffineMap(rewriter, op.getLoc(),
                                                    op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto expandedIndices = *maybeExpandedMap;

    // AffinePromotion reverses dimension order (column-major FIR → row-major
    // memref) and index order.  Reverse indices back for fir.coordinate_of
    // which uses Fortran's column-major layout.
    // ConvertConversion already strips the single fir.convert (FIR -> memref)
    // that AffinePromotion created, so `base` is the original FIR value.
    // Do NOT trace through any remaining fir.convert — those belong to the
    // source IR (e.g. linearisation converts from -O2 whole-array lowering).
    Value base = adaptor.getMemref();

    auto hasSequenceType = [](mlir::Type ty) -> bool {
      if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(ty))
        return mlir::isa<fir::SequenceType>(refTy.getEleTy());
      if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
        return mlir::isa<fir::SequenceType>(heapTy.getEleTy());
      return false;
    };

    if (!hasSequenceType(base.getType()))
      return op.emitError(
          "unsupported memref base: expected !fir.ref<!fir.array<...>> or "
          "!fir.heap<!fir.array<...>>; fir.box and plain memref bases "
          "are not yet handled by AffineDemotion");

    std::reverse(expandedIndices.begin(), expandedIndices.end());

    auto resultRefTy = fir::ReferenceType::get(op.getResult().getType());

    if (baseHasDynamicExtents(base)) {
      mlir::Value boxBase;
      mlir::Value shape = findShapeForBase(base, boxBase);

      if (shape) {
        auto fortranIndices =
            toFortranIndices(shape, expandedIndices, op.getLoc(), rewriter);
        auto arrayCoorOp = fir::ArrayCoorOp::create(
            rewriter, op.getLoc(), resultRefTy, base, shape,
            /*slice=*/mlir::Value{}, fortranIndices,
            /*typeparams=*/mlir::ValueRange{});
        rewriter.replaceOpWithNewOp<fir::LoadOp>(op, arrayCoorOp.getResult());
      } else if (boxBase) {

        auto arrayCoorOp = buildBoxArrayCoor(op.getLoc(), rewriter, resultRefTy,
                                             boxBase, expandedIndices);
        rewriter.replaceOpWithNewOp<fir::LoadOp>(op, arrayCoorOp.getResult());
      } else {
        return op.emitError(
            "cannot find shape or box for dynamic-extent array");
      }
    } else {
      auto coorOp = fir::CoordinateOp::create(
          rewriter, op.getLoc(), resultRefTy, base, expandedIndices);
      rewriter.replaceOpWithNewOp<fir::LoadOp>(op, coorOp.getResult());
    }
    return success();
  }
};

class AffineStoreConversion
    : public OpConversionPattern<mlir::affine::AffineStoreOp> {
public:
  using OpConversionPattern<mlir::affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::affine::AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> indices(op.getIndices());
    auto maybeExpandedMap = affine::expandAffineMap(rewriter, op.getLoc(),
                                                    op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto expandedIndices = *maybeExpandedMap;

    Value base = adaptor.getMemref();

    auto hasSequenceType = [](mlir::Type ty) -> bool {
      if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(ty))
        return mlir::isa<fir::SequenceType>(refTy.getEleTy());
      if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
        return mlir::isa<fir::SequenceType>(heapTy.getEleTy());
      return false;
    };

    if (!hasSequenceType(base.getType()))
      return op.emitError(
          "unsupported memref base: expected !fir.ref<!fir.array<...>> or "
          "!fir.heap<!fir.array<...>>; fir.box and plain memref bases "
          "are not yet handled by AffineDemotion");

    std::reverse(expandedIndices.begin(), expandedIndices.end());

    auto resultRefTy = fir::ReferenceType::get(op.getValueToStore().getType());

    if (baseHasDynamicExtents(base)) {
      mlir::Value boxBase;
      mlir::Value shape = findShapeForBase(base, boxBase);

      if (shape) {
        auto fortranIndices =
            toFortranIndices(shape, expandedIndices, op.getLoc(), rewriter);
        auto arrayCoorOp = fir::ArrayCoorOp::create(
            rewriter, op.getLoc(), resultRefTy, base, shape,
            /*slice=*/mlir::Value{}, fortranIndices,
            /*typeparams=*/mlir::ValueRange{});
        rewriter.replaceOpWithNewOp<fir::StoreOp>(op, adaptor.getValue(),
                                                  arrayCoorOp.getResult());
      } else if (boxBase) {

        auto arrayCoorOp = buildBoxArrayCoor(op.getLoc(), rewriter, resultRefTy,
                                             boxBase, expandedIndices);
        rewriter.replaceOpWithNewOp<fir::StoreOp>(op, adaptor.getValue(),
                                                  arrayCoorOp.getResult());
      } else {
        return op.emitError(
            "cannot find shape or box for dynamic-extent array");
      }
    } else {
      auto coorOp = fir::CoordinateOp::create(
          rewriter, op.getLoc(), resultRefTy, base, expandedIndices);
      rewriter.replaceOpWithNewOp<fir::StoreOp>(op, adaptor.getValue(),
                                                coorOp.getResult());
    }
    return success();
  }
};

class ConvertConversion : public mlir::OpRewritePattern<fir::ConvertOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  llvm::LogicalResult
  matchAndRewrite(fir::ConvertOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (mlir::isa<mlir::MemRefType>(op.getRes().getType())) {
      mlir::Type srcTy = op.getValue().getType();
      auto getSeqTy = [](mlir::Type t) -> fir::SequenceType {
        if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(t))
          return mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy());
        if (auto heapTy = mlir::dyn_cast<fir::HeapType>(t))
          return mlir::dyn_cast<fir::SequenceType>(heapTy.getEleTy());
        return {};
      };
      if (getSeqTy(srcTy)) {
        rewriter.replaceOp(op, op.getValue());
        return success();
      }
      rewriter.replaceOp(op, op.getValue());
    }
    return success();
  }
};

mlir::Type convertMemRef(mlir::MemRefType type) {
  return fir::SequenceType::get(SmallVector<int64_t>(type.getShape()),
                                type.getElementType());
}

class StdAllocConversion : public mlir::OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  llvm::LogicalResult
  matchAndRewrite(memref::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<fir::AllocaOp>(op, convertMemRef(op.getType()),
                                               op.getMemref());
    return success();
  }
};

class AffineDialectDemotion
    : public fir::impl::AffineDialectDemotionBase<AffineDialectDemotion> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    auto function = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "AffineDemotion: running on function:\n";
               function.print(llvm::dbgs()););

    mlir::RewritePatternSet patterns(context);
    patterns.insert<ConvertConversion>(context);
    patterns.insert<AffineLoadConversion>(context);
    patterns.insert<AffineStoreConversion>(context);
    patterns.insert<StdAllocConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalOp<memref::AllocOp>();
    target.addDynamicallyLegalOp<fir::ConvertOp>([](fir::ConvertOp op) {
      if (mlir::isa<mlir::MemRefType>(op.getRes().getType()))
        return false;
      return true;
    });
    target
        .addLegalDialect<FIROpsDialect, mlir::scf::SCFDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect>();

    if (mlir::failed(mlir::applyPartialConversion(function, target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting affine dialect\n");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createAffineDemotionPass() {
  return std::make_unique<AffineDialectDemotion>();
}
