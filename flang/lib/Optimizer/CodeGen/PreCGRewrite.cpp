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

namespace {

/// Convert fir.embox to the extended form where necessary.
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
          if (!scalarCharacter(seqTy))
            return rewriteStaticShape(embox, rewriter, seqTy);
    return mlir::failure();
  }

  static bool scalarCharacter(SequenceType seqTy) {
    if (auto eleTy = seqTy.getEleTy().dyn_cast<CharacterType>())
      return seqTy.getDimension() == 1;
    return false;
  }

  /// For element type `char<K>` the row is the LEN and must not be included in
  /// the shape structure.
  static std::size_t charAdjust(SequenceType seqTy) {
    return seqTy.getEleTy().isa<CharacterType>() ? 1 : 0;
  }

  mlir::LogicalResult rewriteStaticShape(EmboxOp embox,
                                         mlir::PatternRewriter &rewriter,
                                         SequenceType seqTy) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    auto idxTy = rewriter.getIndexType();
    for (auto ext : llvm::drop_begin(seqTy.getShape(), charAdjust(seqTy))) {
      auto iAttr = rewriter.getIndexAttr(ext);
      auto extVal = rewriter.create<mlir::ConstantOp>(loc, idxTy, iAttr);
      shapeOpers.push_back(extVal);
    }
    mlir::NamedAttrList attrs;
    auto rank = seqTy.getDimension() - charAdjust(seqTy);
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::rankAttrName(), rankAttr));
    auto zeroAttr = rewriter.getIntegerAttr(idxTy, 0);
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::lenParamAttrName(), zeroAttr));
    auto shapeAttr = rewriter.getIntegerAttr(idxTy, shapeOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::shapeAttrName(), shapeAttr));
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::shiftAttrName(), zeroAttr));
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::sliceAttrName(), zeroAttr));
    auto xbox = rewriter.create<XEmboxOp>(loc, embox.getType(), embox.memref(),
                                          shapeOpers, llvm::None, llvm::None,
                                          llvm::None, attrs);
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }

  mlir::LogicalResult rewriteDynamicShape(EmboxOp embox,
                                          mlir::PatternRewriter &rewriter,
                                          mlir::Value shapeVal) const {
    auto loc = embox.getLoc();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    unsigned rank;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
      rank = shapeOp.getType().cast<ShapeType>().getRank();
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      rank = shiftOp.getType().cast<ShapeShiftType>().getRank();
    }
    mlir::NamedAttrList attrs;
    auto idxTy = rewriter.getIndexType();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::rankAttrName(), rankAttr));
    auto lenParamSize = embox.lenParams().size();
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
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp()))
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
    auto sliceAttr = rewriter.getIntegerAttr(idxTy, sliceOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::sliceAttrName(), sliceAttr));
    auto xbox = rewriter.create<XEmboxOp>(loc, embox.getType(), embox.memref(),
                                          shapeOpers, shiftOpers, sliceOpers,
                                          embox.lenParams(), attrs);
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
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
    auto shapeVal = arrCoor.shape();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value, 8> shapeOpers;
    llvm::SmallVector<mlir::Value, 8> shiftOpers;
    unsigned rank;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
      rank = shapeOp.getType().cast<ShapeType>().getRank();
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      if (shiftOp)
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      rank = shiftOp.getType().cast<ShapeShiftType>().getRank();
    }
    mlir::NamedAttrList attrs;
    auto idxTy = rewriter.getIndexType();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::rankAttrName(), rankAttr));
    auto lenParamSize = arrCoor.lenParams().size();
    auto lenParamAttr = rewriter.getIntegerAttr(idxTy, lenParamSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::lenParamAttrName(), lenParamAttr));
    auto indexSize = arrCoor.indices().size();
    auto idxAttr = rewriter.getIntegerAttr(idxTy, indexSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::indexAttrName(), idxAttr));
    auto dimAttr = rewriter.getIntegerAttr(idxTy, shapeOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::shapeAttrName(), dimAttr));
    llvm::SmallVector<mlir::Value, 8> sliceOpers;
    if (auto s = arrCoor.slice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp()))
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
    auto sliceAttr = rewriter.getIntegerAttr(idxTy, sliceOpers.size());
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::sliceAttrName(), sliceAttr));
    auto xArrCoor = rewriter.create<XArrayCoorOp>(
        loc, arrCoor.getType(), arrCoor.memref(), shapeOpers, shiftOpers,
        sliceOpers, arrCoor.indices(), arrCoor.lenParams(), attrs);
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << arrCoor << " to " << xArrCoor << '\n');
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
    target.addDynamicallyLegalOp<EmboxOp>([](EmboxOp embox) {
      return !(embox.getShape() ||
               embox.getType().cast<BoxType>().getEleTy().isa<SequenceType>());
    });

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

/// Convert FIR's structured control flow ops to CFG ops.  This conversion
/// enables the `createLowerToCFGPass` to transform these to CFG form.
std::unique_ptr<mlir::Pass> fir::createFirCodeGenRewritePass() {
  return std::make_unique<CodeGenRewrite>();
}
