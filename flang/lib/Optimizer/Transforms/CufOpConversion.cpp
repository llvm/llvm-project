//===-- CufOpConversion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Runtime/allocatable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;
using namespace Fortran::runtime;

namespace {

template <typename OpTy>
static bool isBoxGlobal(OpTy op) {
  if (auto declareOp =
          mlir::dyn_cast<fir::DeclareOp>(op.getBox().getDefiningOp())) {
    if (mlir::isa<fir::AddrOfOp>(declareOp.getMemref().getDefiningOp()))
      return true;
  } else if (auto declareOp = mlir::dyn_cast<hlfir::DeclareOp>(
                 op.getBox().getDefiningOp())) {
    if (mlir::isa<fir::AddrOfOp>(declareOp.getMemref().getDefiningOp()))
      return true;
  }
  return false;
}

template <typename OpTy>
static mlir::LogicalResult convertOpToCall(OpTy op,
                                           mlir::PatternRewriter &rewriter,
                                           mlir::func::FuncOp func) {
  auto mod = op->template getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder(rewriter, mod);
  mlir::Location loc = op.getLoc();
  auto fTy = func.getFunctionType();

  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));

  mlir::Value hasStat = op.getHasStat() ? builder.createBool(loc, true)
                                        : builder.createBool(loc, false);

  mlir::Value errmsg;
  if (op.getErrmsg()) {
    errmsg = op.getErrmsg();
  } else {
    mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
    errmsg = builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  }
  llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, op.getBox(), hasStat, errmsg, sourceFile, sourceLine)};
  auto callOp = builder.create<fir::CallOp>(loc, func, args);
  rewriter.replaceOp(op, callOp);
  return mlir::success();
}

struct CufAllocateOpConversion
    : public mlir::OpRewritePattern<cuf::AllocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // TODO: Allocation with source will need a new entry point in the runtime.
    if (op.getSource())
      return mlir::failure();

    // TODO: Allocation using different stream.
    if (op.getStream())
      return mlir::failure();

    // TODO: Pinned is a reference to a logical value that can be set to true
    // when pinned allocation succeed. This will require a new entry point.
    if (op.getPinned())
      return mlir::failure();

    // TODO: Allocation of module variable will need more work as the descriptor
    // will be duplicated and needs to be synced after allocation.
    if (isBoxGlobal(op))
      return mlir::failure();

    // Allocation for local descriptor falls back on the standard runtime
    // AllocatableAllocate as the dedicated allocator is set in the descriptor
    // before the call.
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(AllocatableAllocate)>(loc,
                                                                   builder);
    return convertOpToCall<cuf::AllocateOp>(op, rewriter, func);
  }
};

struct CufDeallocateOpConversion
    : public mlir::OpRewritePattern<cuf::DeallocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::DeallocateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // TODO: Allocation of module variable will need more work as the descriptor
    // will be duplicated and needs to be synced after allocation.
    if (isBoxGlobal(op))
      return mlir::failure();

    // Deallocation for local descriptor falls back on the standard runtime
    // AllocatableDeallocate as the dedicated deallocator is set in the
    // descriptor before the call.
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(AllocatableDeallocate)>(loc,
                                                                     builder);
    return convertOpToCall<cuf::DeallocateOp>(op, rewriter, func);
  }
};

class CufOpConversion : public fir::impl::CufOpConversionBase<CufOpConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);
    target.addIllegalOp<cuf::AllocateOp, cuf::DeallocateOp>();
    patterns.insert<CufAllocateOpConversion, CufDeallocateOpConversion>(ctx);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace
