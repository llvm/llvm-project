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
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// Codegen rewrite: rewriting of subgraphs of ops
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "flang-codegen-rewrite"

static void populateShape(llvm::SmallVectorImpl<mlir::Value> &vec,
                          fir::ShapeOp shape) {
  vec.append(shape.getExtents().begin(), shape.getExtents().end());
}

// Operands of fir.shape_shift split into two vectors.
static void populateShapeAndShift(llvm::SmallVectorImpl<mlir::Value> &shapeVec,
                                  llvm::SmallVectorImpl<mlir::Value> &shiftVec,
                                  fir::ShapeShiftOp shift) {
  for (auto i = shift.getPairs().begin(), endIter = shift.getPairs().end();
       i != endIter;) {
    shiftVec.push_back(*i++);
    shapeVec.push_back(*i++);
  }
}

static void populateShift(llvm::SmallVectorImpl<mlir::Value> &vec,
                          fir::ShiftOp shift) {
  vec.append(shift.getOrigins().begin(), shift.getOrigins().end());
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
/// %3 = fir.embox %0 (%1) [%2] : (!fir.ref<!fir.array<?xi32>>,
/// !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
/// ```
/// can be rewritten as
/// ```
/// %1 = fircg.ext_embox %0(%5) origin %4[%6, %7, %8] :
/// (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index) ->
/// !fir.box<!fir.array<?xi32>>
/// ```
class EmboxConversion : public mlir::OpRewritePattern<fir::EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    // If the embox does not include a shape, then do not convert it
    if (auto shapeVal = embox.getShape())
      return rewriteDynamicShape(embox, rewriter, shapeVal);
    if (auto boxTy = embox.getType().dyn_cast<fir::BoxType>())
      if (auto seqTy = boxTy.getEleTy().dyn_cast<fir::SequenceType>())
        if (seqTy.hasConstantShape())
          return rewriteStaticShape(embox, rewriter, seqTy);
    return mlir::failure();
  }

  mlir::LogicalResult rewriteStaticShape(fir::EmboxOp embox,
                                         mlir::PatternRewriter &rewriter,
                                         fir::SequenceType seqTy) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    auto idxTy = rewriter.getIndexType();
    for (auto ext : seqTy.getShape()) {
      auto iAttr = rewriter.getIndexAttr(ext);
      auto extVal = rewriter.create<mlir::arith::ConstantOp>(loc, idxTy, iAttr);
      shapeOpers.push_back(extVal);
    }
    auto xbox = rewriter.create<fir::cg::XEmboxOp>(
        loc, embox.getType(), embox.getMemref(), shapeOpers, llvm::None,
        llvm::None, llvm::None, llvm::None, embox.getTypeparams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }

  mlir::LogicalResult rewriteDynamicShape(fir::EmboxOp embox,
                                          mlir::PatternRewriter &rewriter,
                                          mlir::Value shapeVal) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (auto shapeOp = mlir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp())) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp =
          mlir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    llvm::SmallVector<mlir::Value> substrOpers;
    if (auto s = embox.getSlice())
      if (auto sliceOp =
              mlir::dyn_cast_or_null<fir::SliceOp>(s.getDefiningOp())) {
        sliceOpers.assign(sliceOp.getTriples().begin(),
                          sliceOp.getTriples().end());
        subcompOpers.assign(sliceOp.getFields().begin(),
                            sliceOp.getFields().end());
        substrOpers.assign(sliceOp.getSubstr().begin(),
                           sliceOp.getSubstr().end());
      }
    auto xbox = rewriter.create<fir::cg::XEmboxOp>(
        loc, embox.getType(), embox.getMemref(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, substrOpers, embox.getTypeparams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert fir.rebox to the extended form where necessary.
///
/// For example,
/// ```
/// %5 = fir.rebox %3(%1) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) ->
/// !fir.box<!fir.array<?xi32>>
/// ```
/// converted to
/// ```
/// %5 = fircg.ext_rebox %3(%13) origin %12 : (!fir.box<!fir.array<?xi32>>,
/// index, index) -> !fir.box<!fir.array<?xi32>>
/// ```
class ReboxConversion : public mlir::OpRewritePattern<fir::ReboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::ReboxOp rebox,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = rebox.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (auto shapeVal = rebox.getShape()) {
      if (auto shapeOp = mlir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp =
                   mlir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp =
                   mlir::dyn_cast<fir::ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    llvm::SmallVector<mlir::Value> substrOpers;
    if (auto s = rebox.getSlice())
      if (auto sliceOp =
              mlir::dyn_cast_or_null<fir::SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.getTriples().begin(),
                          sliceOp.getTriples().end());
        subcompOpers.append(sliceOp.getFields().begin(),
                            sliceOp.getFields().end());
        substrOpers.append(sliceOp.getSubstr().begin(),
                           sliceOp.getSubstr().end());
      }

    auto xRebox = rewriter.create<fir::cg::XReboxOp>(
        loc, rebox.getType(), rebox.getBox(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, substrOpers);
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
///  %4 = fir.array_coor %addr (%1) [%2] %0 : (!fir.ref<!fir.array<?xi32>>,
///  !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
/// ```
/// converted to
/// ```
/// %40 = fircg.ext_array_coor %addr(%9) origin %8[%4, %5, %6<%39> :
/// (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index, index) ->
/// !fir.ref<i32>
/// ```
class ArrayCoorConversion : public mlir::OpRewritePattern<fir::ArrayCoorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::ArrayCoorOp arrCoor,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = arrCoor.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (auto shapeVal = arrCoor.getShape()) {
      if (auto shapeOp = mlir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp =
                   mlir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp =
                   mlir::dyn_cast<fir::ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    if (auto s = arrCoor.getSlice())
      if (auto sliceOp =
              mlir::dyn_cast_or_null<fir::SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.getTriples().begin(),
                          sliceOp.getTriples().end());
        subcompOpers.append(sliceOp.getFields().begin(),
                            sliceOp.getFields().end());
        assert(sliceOp.getSubstr().empty() &&
               "Don't allow substring operations on array_coor. This "
               "restriction may be lifted in the future.");
      }
    auto xArrCoor = rewriter.create<fir::cg::XArrayCoorOp>(
        loc, arrCoor.getType(), arrCoor.getMemref(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, arrCoor.getIndices(),
        arrCoor.getTypeparams());
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << arrCoor << " to " << xArrCoor << '\n');
    rewriter.replaceOp(arrCoor, xArrCoor.getOperation()->getResults());
    return mlir::success();
  }
};

class CodeGenRewrite : public fir::CodeGenRewriteBase<CodeGenRewrite> {
public:
  void runOn(mlir::Operation *op, mlir::Region &region) {
    auto &context = getContext();
    mlir::OpBuilder rewriter(&context);
    mlir::ConversionTarget target(context);
    target.addLegalDialect<mlir::arith::ArithmeticDialect, fir::FIROpsDialect,
                           fir::FIRCodeGenDialect, mlir::func::FuncDialect>();
    target.addIllegalOp<fir::ArrayCoorOp>();
    target.addIllegalOp<fir::ReboxOp>();
    target.addDynamicallyLegalOp<fir::EmboxOp>([](fir::EmboxOp embox) {
      return !(embox.getShape() || embox.getType()
                                       .cast<fir::BoxType>()
                                       .getEleTy()
                                       .isa<fir::SequenceType>());
    });
    mlir::RewritePatternSet patterns(&context);
    patterns.insert<EmboxConversion, ArrayCoorConversion, ReboxConversion>(
        &context);
    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in running the pre-codegen conversions");
      signalPassFailure();
    }
    // Erase any residual.
    simplifyRegion(region);
  }

  void runOnOperation() override final {
    // Call runOn on all top level regions that may contain emboxOp/arrayCoorOp.
    auto mod = getOperation();
    for (auto func : mod.getOps<mlir::func::FuncOp>())
      runOn(func, func.getBody());
    for (auto global : mod.getOps<fir::GlobalOp>())
      runOn(global, global.getRegion());
  }

  // Clean up the region.
  void simplifyRegion(mlir::Region &region) {
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        for (auto &reg : op.getRegions())
          simplifyRegion(reg);
        maybeEraseOp(&op);
      }
    doDCE();
  }

  /// Run a simple DCE cleanup to remove any dead code after the rewrites.
  void doDCE() {
    std::vector<mlir::Operation *> workList;
    workList.swap(opsToErase);
    while (!workList.empty()) {
      for (auto *op : workList) {
        std::vector<mlir::Value> opOperands(op->operand_begin(),
                                            op->operand_end());
        LLVM_DEBUG(llvm::dbgs() << "DCE on " << *op << '\n');
        ++numDCE;
        op->erase();
        for (auto opnd : opOperands)
          maybeEraseOp(opnd.getDefiningOp());
      }
      workList.clear();
      workList.swap(opsToErase);
    }
  }

  void maybeEraseOp(mlir::Operation *op) {
    if (!op)
      return;
    if (op->hasTrait<mlir::OpTrait::IsTerminator>())
      return;
    if (mlir::isOpTriviallyDead(op))
      opsToErase.push_back(op);
  }

private:
  std::vector<mlir::Operation *> opsToErase;
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createFirCodeGenRewritePass() {
  return std::make_unique<CodeGenRewrite>();
}
