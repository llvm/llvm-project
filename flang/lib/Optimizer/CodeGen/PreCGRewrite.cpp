//===-- PreCGRewrite.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "CGOps.h"
#include "PassDetail.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

//===----------------------------------------------------------------------===//
// Codegen rewrite: rewriting of subgraphs of ops
//===----------------------------------------------------------------------===//

using namespace fir;

#define DEBUG_TYPE "flang-codegen-rewrite"

static void populateShape(llvm::SmallVectorImpl<mlir::Value> &vec,
                          ShapeOp shape) {
  vec.append(shape.extents().begin(), shape.extents().end());
}

// Operands of fir.shape_shift split into two vectors.
static void populateShapeAndShift(llvm::SmallVectorImpl<mlir::Value> &shapeVec,
                                  llvm::SmallVectorImpl<mlir::Value> &shiftVec,
                                  ShapeShiftOp shift) {
  auto endIter = shift.pairs().end();
  for (auto i = shift.pairs().begin(); i != endIter;) {
    shiftVec.push_back(*i++);
    shapeVec.push_back(*i++);
  }
}

static void populateShift(llvm::SmallVectorImpl<mlir::Value> &vec,
                          ShiftOp shift) {
  vec.append(shift.origins().begin(), shift.origins().end());
}

namespace {

/// Convert fir.embox to the extended form where necessary.
///
/// The embox operation can take arguments that specify multidimensional array
/// properties at runtime. These properties may be shared between distinct
/// objects that have the same properties. Before we lower these small DAGs to
/// LLVM-IR, we gather all the information into a single extended operation. For
/// example,
/// ```
/// %1 = fir.shape_shift %4, %5 : (index, index) -> !fir.shapeshift<1>
/// %2 = fir.slice %6, %7, %8 : (index, index, index) -> !fir.slice<1>
/// %3 = fir.embox %0 (%1) [%2] : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
/// ```
/// can be rewritten as
/// ```
/// %1 = fircg.ext_embox %0(%5) origin %4[%6, %7, %8] : (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index) -> !fir.box<!fir.array<?xi32>>
/// ```
class EmboxConversion : public mlir::OpRewritePattern<EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EmboxOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    auto shapeVal = embox.getShape();
    // If the embox does not include a shape, then do not convert it
    if (shapeVal)
      return rewriteDynamicShape(embox, rewriter, shapeVal);
    if (auto boxTy = embox.getType().dyn_cast<BoxType>())
      if (auto seqTy = boxTy.getEleTy().dyn_cast<SequenceType>())
        if (seqTy.hasConstantShape())
          return rewriteStaticShape(embox, rewriter, seqTy);
    return mlir::failure();
  }

  mlir::LogicalResult rewriteStaticShape(EmboxOp embox,
                                         mlir::PatternRewriter &rewriter,
                                         SequenceType seqTy) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    auto idxTy = rewriter.getIndexType();
    for (auto ext : seqTy.getShape()) {
      auto iAttr = rewriter.getIndexAttr(ext);
      auto extVal = rewriter.create<mlir::ConstantOp>(loc, idxTy, iAttr);
      shapeOpers.push_back(extVal);
    }
    auto xbox = rewriter.create<cg::XEmboxOp>(
        loc, embox.getType(), embox.memref(), shapeOpers, llvm::None,
        llvm::None, llvm::None, embox.lenParams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }

  mlir::LogicalResult rewriteDynamicShape(EmboxOp embox,
                                          mlir::PatternRewriter &rewriter,
                                          mlir::Value shapeVal) const {
    auto loc = embox.getLoc();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    if (auto s = embox.getSlice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
        subcompOpers.append(sliceOp.fields().begin(), sliceOp.fields().end());
      }
    auto xbox = rewriter.create<cg::XEmboxOp>(
        loc, embox.getType(), embox.memref(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, embox.lenParams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert fir.rebox to the extended form where necessary.
///
/// For example,
/// ```
/// %5 = fir.rebox %3(%1) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
/// ```
/// converted to
/// ```
/// %5 = fircg.ext_rebox %3(%13) origin %12 : (!fir.box<!fir.array<?xi32>>, index, index) -> !fir.box<!fir.array<?xi32>>
/// ```
class ReboxConversion : public mlir::OpRewritePattern<ReboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReboxOp rebox,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = rebox.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (auto shapeVal = rebox.shape()) {
      if (auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp = dyn_cast<ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    if (auto s = rebox.slice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
        subcompOpers.append(sliceOp.fields().begin(), sliceOp.fields().end());
      }

    auto xRebox = rewriter.create<cg::XReboxOp>(
        loc, rebox.getType(), rebox.box(), shapeOpers, shiftOpers, sliceOpers,
        subcompOpers);
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << rebox << " to " << xRebox << '\n');
    rewriter.replaceOp(rebox, xRebox.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert all fir.array_coor to the extended form.
///
/// For example,
/// ```
///  %4 = fir.array_coor %addr (%1) [%2] %0 : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
/// ```
/// converted to
/// ```
/// %40 = fircg.ext_array_coor %addr(%9) origin %8[%4, %5, %6<%39> : (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index, index) -> !fir.ref<i32>
/// ```
class ArrayCoorConversion : public mlir::OpRewritePattern<ArrayCoorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCoorOp arrCoor,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = arrCoor.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (auto shapeVal = arrCoor.shape()) {
      if (auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp = dyn_cast<ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    if (auto s = arrCoor.slice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
        subcompOpers.append(sliceOp.fields().begin(), sliceOp.fields().end());
      }
    auto xArrCoor = rewriter.create<cg::XArrayCoorOp>(
        loc, arrCoor.getType(), arrCoor.memref(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, arrCoor.indices(), arrCoor.lenParams());
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << arrCoor << " to " << xArrCoor << '\n');
    rewriter.replaceOp(arrCoor, xArrCoor.getOperation()->getResults());
    return mlir::success();
  }
};

class CodeGenRewrite : public CodeGenRewriteBase<CodeGenRewrite> {
public:
  void runOnOperation() override final {
    auto op = getOperation();
    auto &context = getContext();
    mlir::OpBuilder rewriter(&context);
    mlir::ConversionTarget target(context);
    target.addLegalDialect<FIROpsDialect, FIRCodeGenDialect,
                           mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayCoorOp>();
    target.addIllegalOp<ReboxOp>();
    target.addDynamicallyLegalOp<EmboxOp>([](EmboxOp embox) {
      return !(embox.getShape() ||
               embox.getType().cast<BoxType>().getEleTy().isa<SequenceType>());
    });
    mlir::OwningRewritePatternList patterns;
    patterns.insert<EmboxConversion, ArrayCoorConversion, ReboxConversion>(
        &context);
    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in running the pre-codegen conversions");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createFirCodeGenRewritePass() {
  return std::make_unique<CodeGenRewrite>();
}
