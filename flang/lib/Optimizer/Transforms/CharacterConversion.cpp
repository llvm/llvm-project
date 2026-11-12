//===- CharacterConversion.cpp -- convert between character encodings -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace fir {
#define GEN_PASS_DEF_CHARACTERCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

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

  llvm::LogicalResult
  matchAndRewrite(fir::CharConvertOp conv,
                  mlir::PatternRewriter &rewriter) const override {
    auto kindMap = fir::getKindMapping(conv->getParentOfType<mlir::ModuleOp>());
    auto loc = conv.getLoc();

    LLVM_DEBUG(llvm::dbgs()
               << "running character conversion on " << conv << '\n');

    // Establish a loop that executes count iterations.
    auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto idxTy = rewriter.getIndexType();
    auto castCnt =
        fir::ConvertOp::create(rewriter, loc, idxTy, conv.getCount());
    auto countm1 = mlir::arith::SubIOp::create(rewriter, loc, castCnt, one);
    auto loop = fir::DoLoopOp::create(rewriter, loc, zero, countm1, one);
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(loop.getBody());

    // For each code point in the `from` string, convert naively to the `to`
    // string code point. Conversion is done blindly on size only, not value.
    auto getCharBits = [&](mlir::Type t) {
      auto chrTy = mlir::cast<fir::CharacterType>(
          fir::unwrapSequenceType(fir::dyn_cast_ptrEleTy(t)));
      return kindMap.getCharacterBitsize(chrTy.getFKind());
    };
    auto fromBits = getCharBits(conv.getFrom().getType());
    auto toBits = getCharBits(conv.getTo().getType());
    auto pointerType = [&](unsigned bits) {
      return fir::ReferenceType::get(fir::SequenceType::get(
          fir::SequenceType::ShapeRef{fir::SequenceType::getUnknownExtent()},
          rewriter.getIntegerType(bits)));
    };
    auto fromPtrTy = pointerType(fromBits);
    auto toTy = rewriter.getIntegerType(toBits);
    auto toPtrTy = pointerType(toBits);
    auto fromPtr =
        fir::ConvertOp::create(rewriter, loc, fromPtrTy, conv.getFrom());
    auto toPtr = fir::ConvertOp::create(rewriter, loc, toPtrTy, conv.getTo());
    auto getEleTy = [&](unsigned bits) {
      return fir::ReferenceType::get(rewriter.getIntegerType(bits));
    };
    auto fromi =
        fir::CoordinateOp::create(rewriter, loc, getEleTy(fromBits), fromPtr,
                                  mlir::ValueRange{loop.getInductionVar()});
    auto toi =
        fir::CoordinateOp::create(rewriter, loc, getEleTy(toBits), toPtr,
                                  mlir::ValueRange{loop.getInductionVar()});
    auto load = fir::LoadOp::create(rewriter, loc, fromi);
    mlir::Value icast =
        (fromBits >= toBits)
            ? fir::ConvertOp::create(rewriter, loc, toTy, load).getResult()
            : mlir::arith::ExtUIOp::create(rewriter, loc, toTy, load)
                  .getResult();
    rewriter.replaceOpWithNewOp<fir::StoreOp>(conv, icast, toi);
    rewriter.restoreInsertionPoint(insPt);
    return mlir::success();
  }
};

/// Rewrite the `fir.char_convert` op into a loop. This pass must be run only on
/// fir::CharConvertOp.
class CharacterConversion
    : public fir::impl::CharacterConversionBase<CharacterConversion> {
public:
  using fir::impl::CharacterConversionBase<
      CharacterConversion>::CharacterConversionBase;

  void runOnOperation() override {
    CharacterConversionOptions clOpts{useRuntimeCalls.getValue()};
    if (clOpts.runtimeName.empty()) {
      auto *context = &getContext();
      auto *func = getOperation();
      mlir::RewritePatternSet patterns(context);
      patterns.insert<CharacterConvertConversion>(context);
      mlir::ConversionTarget target(*context);
      target.addLegalDialect<mlir::affine::AffineDialect, fir::FIROpsDialect,
                             mlir::arith::ArithDialect,
                             mlir::func::FuncDialect>();

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
