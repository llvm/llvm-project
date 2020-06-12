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

namespace {

/// Convert fir.embox to the extended form where necessary.
class EmboxConversion : public mlir::OpRewritePattern<EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EmboxOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = embox.getLoc();
    auto dimsVal = embox.getDims();
    if (!dimsVal)
      return mlir::failure();
    auto dimsOp = dyn_cast<GenDimsOp>(dimsVal.getDefiningOp());
    assert(dimsOp && "dims is not a fir.gendims");
    mlir::NamedAttrList attrs;
    auto lenParamSize = embox.getLenParams().size();
    auto idxTy = rewriter.getIndexType();
    auto lenParamAttr = rewriter.getIntegerAttr(idxTy, lenParamSize);
    attrs.push_back(
        rewriter.getNamedAttr(XEmboxOp::lenParamAttrName(), lenParamAttr));
    auto dimsSize = dimsOp.getNumOperands();
    auto dimAttr = rewriter.getIntegerAttr(idxTy, dimsSize);
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::dimsAttrName(), dimAttr));
    auto rank = dimsOp.getType().cast<fir::DimsType>().getRank();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(rewriter.getNamedAttr(XEmboxOp::rankAttrName(), rankAttr));
    auto xbox = rewriter.create<XEmboxOp>(loc, embox.getType(), embox.memref(),
                                          embox.getLenParams(),
                                          dimsOp.getOperands(), attrs);
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
    auto dimsVal = arrCoor.dims();
    auto dimsOp = dyn_cast<GenDimsOp>(dimsVal.getDefiningOp());
    assert(dimsOp && "dims is not a fir.gendims");
    mlir::NamedAttrList attrs;
    auto indexSize = arrCoor.coor().size();
    auto idxTy = rewriter.getIndexType();
    auto idxAttr = rewriter.getIntegerAttr(idxTy, indexSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::indexAttrName(), idxAttr));
    auto dimsSize = dimsOp.getNumOperands();
    auto dimAttr = rewriter.getIntegerAttr(idxTy, dimsSize);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::dimsAttrName(), dimAttr));
    auto rank = dimsOp.getType().cast<fir::DimsType>().getRank();
    auto rankAttr = rewriter.getIntegerAttr(idxTy, rank);
    attrs.push_back(
        rewriter.getNamedAttr(XArrayCoorOp::rankAttrName(), rankAttr));
    auto xArrCoor = rewriter.create<XArrayCoorOp>(
        loc, arrCoor.getType(), arrCoor.ref(), dimsOp.getOperands(),
        arrCoor.coor(), attrs);
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
        [](EmboxOp embox) { return !embox.getDims(); });

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
    // Erase any embox that was replaced.
    if (auto embox = dyn_cast_or_null<EmboxOp>(op))
      if (embox.getDims()) {
        assert(op->use_empty());
        opsToErase.push_back(op);
      }

    // Erase all fir.array_coor.
    if (auto arrCoor = dyn_cast_or_null<ArrayCoorOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }

    // Erase all fir.gendims ops.
    if (auto genDims = dyn_cast_or_null<GenDimsOp>(op)) {
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
