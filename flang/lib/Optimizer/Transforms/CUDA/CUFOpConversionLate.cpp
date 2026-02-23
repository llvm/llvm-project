//===-- CUFOpConversionLate.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/Runtime/CUDA/Descriptor.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include "flang/Support/Fortran.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFOPCONVERSIONLATE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;
using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

namespace {

static mlir::Value createConvertOp(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Type toTy,
                                   mlir::Value val) {
  if (val.getType() != toTy)
    return fir::ConvertOp::create(rewriter, loc, toTy, val);
  return val;
}

struct CUFDeviceAddressOpConversion
    : public mlir::OpRewritePattern<cuf::DeviceAddressOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFDeviceAddressOpConversion(mlir::MLIRContext *context,
                               const mlir::SymbolTable &symtab)
      : OpRewritePattern(context), symTab{symtab} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::DeviceAddressOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (auto global = symTab.lookup<fir::GlobalOp>(
            op.getHostSymbol().getRootReference().getValue())) {
      auto mod = op->getParentOfType<mlir::ModuleOp>();
      mlir::Location loc = op.getLoc();
      auto hostAddr = fir::AddrOfOp::create(
          rewriter, loc, fir::ReferenceType::get(global.getType()),
          op.getHostSymbol());
      fir::FirOpBuilder builder(rewriter, mod);
      mlir::func::FuncOp callee =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFGetDeviceAddress)>(loc,
                                                                     builder);
      auto fTy = callee.getFunctionType();
      mlir::Value conv =
          createConvertOp(rewriter, loc, fTy.getInput(0), hostAddr);
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, conv, sourceFile, sourceLine)};
      auto call = fir::CallOp::create(rewriter, loc, callee, args);
      mlir::Value addr = createConvertOp(rewriter, loc, hostAddr.getType(),
                                         call->getResult(0));
      rewriter.replaceOp(op, addr.getDefiningOp());
      return success();
    }
    return failure();
  }

private:
  const mlir::SymbolTable &symTab;
};

class CUFOpConversionLate
    : public fir::impl::CUFOpConversionLateBase<CUFOpConversionLate> {
  using CUFOpConversionLateBase::CUFOpConversionLateBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);
    mlir::Operation *op = getOperation();
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();
    mlir::SymbolTable symtab(module);
    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithDialect,
                           mlir::gpu::GPUDialect>();
    patterns.insert<CUFDeviceAddressOpConversion>(patterns.getContext(),
                                                  symtab);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace
