//===-- PreCGRewrite.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"

using namespace fir;

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

namespace {

/// Convert fir.embox to the extended form where necessary.
class EmboxConversion : public mlir::OpRewritePattern<EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EmboxOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = embox.getLoc();
    auto dimsVal = embox.getShape();
    // If the embox does not include a shape, then do not convert it
    if (!dimsVal)
      return mlir::failure();
    auto shapeOp = dyn_cast<ShapeOp>(dimsVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(dimsVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    mlir::NamedAttrList attrs;
    auto idxTy = rewriter.getIndexType();
    auto rank = shapeOp.getType().cast<ShapeType>().getRank();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::rankAttrName(), rankAttr));
    auto lenParamSize = embox.getLenParams().size();
    auto lenParamAttr = rewriter.getIntegerAttr(idxTy, lenParamSize);
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::lenParamAttrName(), lenParamAttr));
    auto shapeAttr = rewriter.getIntegerAttr(idxTy, shapeOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::shapeAttrName(), shapeAttr));
    auto shiftAttr = rewriter.getIntegerAttr(idxTy, shiftOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::shiftAttrName(), shiftAttr));
    llvm::SmallVector<mlir::Value, 8> sliceOpers;
    if (auto s = embox.getSlice())
      if (auto sliceOp =
          dyn_cast_or_null<SliceOp>(s.getDefiningOp()))
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
    auto sliceAttr = rewriter.getIntegerAttr(idxTy, sliceOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::sliceAttrName(), sliceAttr));
    auto xbox = rewriter.create<XEmboxOp>(loc, embox.getType(), embox.memref(),
                                          shapeOpers, shiftOpers, sliceOpers,
                                          embox.getLenParams(), attrs);
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert all fir.array_coor to the extended form.
class ArrayCoorConversion : public mlir::OpRewritePattern<ArrayCoorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCoorOp arrCoor,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = arrCoor.getLoc();
    auto shapeVal = arrCoor.getShape();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      if (shiftOp)
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    mlir::NamedAttrList attrs;
    auto idxTy = rewriter.getIndexType();
    auto rank = shapeOp.getType().cast<ShapeType>().getRank();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::rankAttrName(), rankAttr));
    auto lenParamSize = arrCoor.getLenParams().size();
    auto lenParamAttr = rewriter.getIntegerAttr(idxTy, lenParamSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::lenParamAttrName(), lenParamAttr));
    auto indexSize = arrCoor.getIndices().size();
    auto idxAttr = rewriter.getIntegerAttr(idxTy, indexSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::indexAttrName(), idxAttr));
    auto shapeSize = shapeOp.getNumOperands();
    auto dimAttr = rewriter.getIntegerAttr(idxTy, shapeSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::shapeAttrName(), dimAttr));
    llvm::SmallVector<mlir::Value, 8> sliceOpers;
    if (auto s = arrCoor.getSlice())
      if (auto sliceOp =
          dyn_cast_or_null<SliceOp>(s.getDefiningOp()))
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
    auto sliceAttr = rewriter.getIntegerAttr(idxTy, sliceOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::sliceAttrName(), sliceAttr));
    auto xArrCoor = rewriter.create<XArrayCoorOp>(
        loc, arrCoor.getType(), arrCoor.memref(), shapeOpers,
        shiftOpers, sliceOpers,
        arrCoor.getIndices(), arrCoor.getLenParams(), attrs);
    rewriter.replaceOp(arrCoor, xArrCoor.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert FIR structured control flow ops to CFG ops.
class CodeGenRewrite : public CodeGenRewriteBase<CodeGenRewrite> {
public:
  void runOnFunction() override final {
    auto &context = getContext();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<EmboxConversion, ArrayCoorConversion>(&context);
    mlir::ConversionTarget target(context);
    target.addLegalDialect<FIROpsDialect, mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayCoorOp>();
    target.addDynamicallyLegalOp<EmboxOp>(
        [](EmboxOp embox) { return !embox.getShape(); });

    // Do the conversions.
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in running the pre-codegen conversions");
      signalPassFailure();
    }

    // Erase any residual.
    simplifyRegion(getFunction().getBody());
  }

  // Clean up the region.
  void simplifyRegion(mlir::Region &region) {
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        if (op.getNumRegions() != 0)
          for (auto &reg : op.getRegions())
            simplifyRegion(reg);
        maybeEraseOp(&op);
      }

    for (auto *op : opsToErase)
      op->erase();
    opsToErase.clear();
  }

  void maybeEraseOp(mlir::Operation *op) {
    if (!op)
      return;

    // Erase any embox that was replaced.
    if (auto embox = dyn_cast<EmboxOp>(op))
      if (embox.getShape()) {
        assert(op->use_empty());
        opsToErase.push_back(op);
      }

    // Erase all fir.array_coor.
    if (isa<ArrayCoorOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }

    // Erase all fir.shape, fir.shape_shift, and fir.slice ops.
    if (isa<ShapeOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
    if (isa<ShapeShiftOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
    if (isa<SliceOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
  }

private:
  std::vector<mlir::Operation *> opsToErase;
};

} // namespace

/// Convert FIR's structured control flow ops to CFG ops.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> fir::createFirCodeGenRewritePass() {
  return std::make_unique<CodeGenRewrite>();
}
