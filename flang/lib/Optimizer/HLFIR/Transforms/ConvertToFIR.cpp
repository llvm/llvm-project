//===- ConvertToFIR.cpp - Convert HLFIR to FIR ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass to lower HLFIR to FIR
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_CONVERTHLFIRTOFIR
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

using namespace mlir;

namespace {
class DeclareOpConversion : public mlir::OpRewritePattern<hlfir::DeclareOp> {
public:
  explicit DeclareOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::DeclareOp declareOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = declareOp->getLoc();
    mlir::Value memref = declareOp.getMemref();
    fir::FortranVariableFlagsAttr fortranAttrs;
    if (auto attrs = declareOp.getFortranAttrs())
      fortranAttrs =
          fir::FortranVariableFlagsAttr::get(rewriter.getContext(), *attrs);
    auto firBase = rewriter
                       .create<fir::DeclareOp>(
                           loc, memref.getType(), memref, declareOp.getShape(),
                           declareOp.getTypeparams(), declareOp.getUniqName(),
                           fortranAttrs)
                       .getResult();
    mlir::Value hlfirBase;
    mlir::Type hlfirBaseType = declareOp.getBase().getType();
    if (hlfirBaseType.isa<fir::BaseBoxType>()) {
      // Need to conditionally rebox/embox for optional.
      if (mlir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation())
              .isOptional())
        TODO(loc, "converting hlfir declare of optional box to fir");
      if (!firBase.getType().isa<fir::BaseBoxType>()) {
        llvm::SmallVector<mlir::Value> typeParams;
        auto maybeCharType =
            fir::unwrapSequenceType(fir::unwrapPassByRefType(hlfirBaseType))
                .dyn_cast<fir::CharacterType>();
        if (!maybeCharType || maybeCharType.hasDynamicLen())
          typeParams.append(declareOp.getTypeparams().begin(),
                            declareOp.getTypeparams().end());
        hlfirBase = rewriter.create<fir::EmboxOp>(
            loc, hlfirBaseType, firBase, declareOp.getShape(),
            /*slice=*/mlir::Value{}, typeParams);
      } else {
        // Rebox so that lower bounds are correct.
        hlfirBase = rewriter.create<fir::ReboxOp>(loc, hlfirBaseType, firBase,
                                                  declareOp.getShape(),
                                                  /*slice=*/mlir::Value{});
      }
    } else if (hlfirBaseType.isa<fir::BoxCharType>()) {
      assert(declareOp.getTypeparams().size() == 1 &&
             "must contain character length");
      hlfirBase = rewriter.create<fir::EmboxCharOp>(
          loc, hlfirBaseType, firBase, declareOp.getTypeparams()[0]);
    } else {
      if (hlfirBaseType != firBase.getType()) {
        declareOp.emitOpError()
            << "unhandled HLFIR variable type '" << hlfirBaseType << "'\n";
        return mlir::failure();
      }
      hlfirBase = firBase;
    }
    rewriter.replaceOp(declareOp, {hlfirBase, firBase});
    return mlir::success();
  }
};

class ConvertHLFIRtoFIR
    : public hlfir::impl::ConvertHLFIRtoFIRBase<ConvertHLFIRtoFIR> {
public:
  void runOnOperation() override {
    auto func = this->getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DeclareOpConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalDialect<hlfir::hlfirDialect>();
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in HLFIR to FIR conversion pass");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> hlfir::createConvertHLFIRtoFIRPass() {
  return std::make_unique<ConvertHLFIRtoFIR>();
}
