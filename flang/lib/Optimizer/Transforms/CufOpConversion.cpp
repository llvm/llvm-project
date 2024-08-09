//===-- CufOpConversion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Runtime/CUDA/descriptor.h"
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
using namespace Fortran::runtime::cuda;

namespace {

template <typename OpTy>
static bool isBoxGlobal(OpTy op) {
  if (auto declareOp =
          mlir::dyn_cast_or_null<fir::DeclareOp>(op.getBox().getDefiningOp())) {
    if (mlir::isa_and_nonnull<fir::AddrOfOp>(
            declareOp.getMemref().getDefiningOp()))
      return true;
  } else if (auto declareOp = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
                 op.getBox().getDefiningOp())) {
    if (mlir::isa_and_nonnull<fir::AddrOfOp>(
            declareOp.getMemref().getDefiningOp()))
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

struct CufAllocOpConversion : public mlir::OpRewritePattern<cuf::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  CufAllocOpConversion(mlir::MLIRContext *context, mlir::DataLayout *dl,
                       fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), dl{dl}, typeConverter{typeConverter} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType());

    // Only convert cuf.alloc that allocates a descriptor.
    if (!boxTy)
      return failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocDesciptor)>(loc, builder);

    auto fTy = func.getFunctionType();
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));

    mlir::Type structTy = typeConverter->convertBoxTypeAsStruct(boxTy);
    std::size_t boxSize = dl->getTypeSizeInBits(structTy) / 8;
    mlir::Value sizeInBytes =
        builder.createIntegerConstant(loc, builder.getIndexType(), boxSize);

    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, sizeInBytes, sourceFile, sourceLine)};
    auto callOp = builder.create<fir::CallOp>(loc, func, args);
    auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                        callOp.getResult(0));
    rewriter.replaceOp(op, convOp);
    return mlir::success();
  }

private:
  mlir::DataLayout *dl;
  fir::LLVMTypeConverter *typeConverter;
};

struct CufFreeOpConversion : public mlir::OpRewritePattern<cuf::FreeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::FreeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Only convert cuf.free on descriptor.
    if (!mlir::isa<fir::ReferenceType>(op.getDevptr().getType()))
      return failure();
    auto refTy = mlir::dyn_cast<fir::ReferenceType>(op.getDevptr().getType());
    if (!mlir::isa<fir::BaseBoxType>(refTy.getEleTy()))
      return failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFFreeDesciptor)>(loc, builder);

    auto fTy = func.getFunctionType();
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, op.getDevptr(), sourceFile, sourceLine)};
    builder.create<fir::CallOp>(loc, func, args);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CufOpConversion : public fir::impl::CufOpConversionBase<CufOpConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);

    mlir::Operation *op = getOperation();
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();

    std::optional<mlir::DataLayout> dl =
        fir::support::getOrSetDataLayout(module, /*allowDefaultLayout=*/false);
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    target.addDynamicallyLegalOp<cuf::AllocOp>([](::cuf::AllocOp op) {
      return !mlir::isa<fir::BaseBoxType>(op.getInType());
    });
    target.addDynamicallyLegalOp<cuf::FreeOp>([](::cuf::FreeOp op) {
      if (auto refTy = mlir::dyn_cast_or_null<fir::ReferenceType>(
              op.getDevptr().getType())) {
        return !mlir::isa<fir::BaseBoxType>(refTy.getEleTy());
      }
      return true;
    });
    target.addDynamicallyLegalOp<cuf::AllocateOp>(
        [](::cuf::AllocateOp op) { return isBoxGlobal(op); });
    target.addDynamicallyLegalOp<cuf::DeallocateOp>(
        [](::cuf::DeallocateOp op) { return isBoxGlobal(op); });
    patterns.insert<CufAllocOpConversion>(ctx, &*dl, &typeConverter);
    patterns.insert<CufAllocateOpConversion, CufDeallocateOpConversion,
                    CufFreeOpConversion>(ctx);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace
