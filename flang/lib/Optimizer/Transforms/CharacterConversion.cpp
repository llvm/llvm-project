//===- CharacterConversion.cpp -- convert between character encodings -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-character-conversion"

namespace {

// TODO: Future hook to select some set of runtime calls.
struct CharacterConversionOptions {
  std::string runtimeName;
};

class CharacterConvertConversion
    : public mlir::OpRewritePattern<fir::CharConvertOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::CharConvertOp conv,
                  mlir::PatternRewriter &rewriter) const override {
    auto kindMap = fir::getKindMapping(conv->getParentOfType<mlir::ModuleOp>());
    auto loc = conv.getLoc();

    LLVM_DEBUG(llvm::dbgs()
               << "running character conversion on " << conv << '\n');

    // Establish a loop that executes count iterations.
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto idxTy = rewriter.getIndexType();
    auto castCnt = rewriter.create<fir::ConvertOp>(loc, idxTy, conv.count());
    auto countm1 = rewriter.create<mlir::arith::SubIOp>(loc, castCnt, one);
    auto loop = rewriter.create<fir::DoLoopOp>(loc, zero, countm1, one);
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(loop.getBody());

    // For each code point in the `from` string, convert naively to the `to`
    // string code point. Conversion is done blindly on size only, not value.
    auto getCharBits = [&](mlir::Type t) {
      auto chrTy = fir::unwrapSequenceType(fir::dyn_cast_ptrEleTy(t))
                       .cast<fir::CharacterType>();
      return kindMap.getCharacterBitsize(chrTy.getFKind());
    };
    auto fromBits = getCharBits(conv.from().getType());
    auto toBits = getCharBits(conv.to().getType());
    auto pointerType = [&](unsigned bits) {
      return fir::ReferenceType::get(fir::SequenceType::get(
          fir::SequenceType::ShapeRef{fir::SequenceType::getUnknownExtent()},
          rewriter.getIntegerType(bits)));
    };
    auto fromPtrTy = pointerType(fromBits);
    auto toTy = rewriter.getIntegerType(toBits);
    auto toPtrTy = pointerType(toBits);
    auto fromPtr = rewriter.create<fir::ConvertOp>(loc, fromPtrTy, conv.from());
    auto toPtr = rewriter.create<fir::ConvertOp>(loc, toPtrTy, conv.to());
    auto getEleTy = [&](unsigned bits) {
      return fir::ReferenceType::get(rewriter.getIntegerType(bits));
    };
    auto fromi = rewriter.create<fir::CoordinateOp>(
        loc, getEleTy(fromBits), fromPtr,
        mlir::ValueRange{loop.getInductionVar()});
    auto toi = rewriter.create<fir::CoordinateOp>(
        loc, getEleTy(toBits), toPtr, mlir::ValueRange{loop.getInductionVar()});
    auto load = rewriter.create<fir::LoadOp>(loc, fromi);
    mlir::Value icast =
        (fromBits >= toBits)
            ? rewriter.create<fir::ConvertOp>(loc, toTy, load).getResult()
            : rewriter.create<mlir::arith::ExtUIOp>(loc, toTy, load)
                  .getResult();
    rewriter.replaceOpWithNewOp<fir::StoreOp>(conv, icast, toi);
    rewriter.restoreInsertionPoint(insPt);
    return mlir::success();
  }
};

/// Rewrite the `fir.char_convert` op into a loop. This pass must be run only on
/// fir::CharConvertOp.
class CharacterConversion
    : public fir::CharacterConversionBase<CharacterConversion> {
public:
  void runOnOperation() override {
    CharacterConversionOptions clOpts{useRuntimeCalls.getValue()};
    if (clOpts.runtimeName.empty()) {
      auto *context = &getContext();
      auto *func = getOperation();
      mlir::RewritePatternSet patterns(context);
      patterns.insert<CharacterConvertConversion>(context);
      mlir::ConversionTarget target(*context);
      target.addLegalDialect<mlir::AffineDialect, fir::FIROpsDialect,
                             mlir::arith::ArithmeticDialect,
                             mlir::StandardOpsDialect>();

      // apply the patterns
      target.addIllegalOp<fir::CharConvertOp>();
      if (mlir::failed(mlir::applyPartialConversion(func, target,
                                                    std::move(patterns)))) {
        mlir::emitError(mlir::UnknownLoc::get(context),
                        "error in rewriting character convert op");
        signalPassFailure();
      }
      return;
    }

    // TODO: some sort of runtime supported conversion?
    signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> fir::createCharacterConversionPass() {
  return std::make_unique<CharacterConversion>();
}
