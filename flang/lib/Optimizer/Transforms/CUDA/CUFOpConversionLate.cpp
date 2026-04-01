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
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/Matchers.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFOPCONVERSIONLATE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace aiir;
using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

namespace {

static aiir::Value createConvertOp(aiir::PatternRewriter &rewriter,
                                   aiir::Location loc, aiir::Type toTy,
                                   aiir::Value val) {
  if (val.getType() != toTy)
    return fir::ConvertOp::create(rewriter, loc, toTy, val);
  return val;
}

struct CUFDeviceAddressOpConversion
    : public aiir::OpRewritePattern<cuf::DeviceAddressOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFDeviceAddressOpConversion(aiir::AIIRContext *context,
                               const aiir::SymbolTable &symtab)
      : OpRewritePattern(context), symTab{symtab} {}

  aiir::LogicalResult
  matchAndRewrite(cuf::DeviceAddressOp op,
                  aiir::PatternRewriter &rewriter) const override {
    if (auto global = symTab.lookup<fir::GlobalOp>(
            op.getHostSymbol().getRootReference().getValue())) {
      auto mod = op->getParentOfType<aiir::ModuleOp>();
      aiir::Location loc = op.getLoc();
      auto hostAddr = fir::AddrOfOp::create(
          rewriter, loc, fir::ReferenceType::get(global.getType()),
          op.getHostSymbol());
      fir::FirOpBuilder builder(rewriter, mod);
      aiir::func::FuncOp callee =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFGetDeviceAddress)>(loc,
                                                                     builder);
      auto fTy = callee.getFunctionType();
      aiir::Value conv =
          createConvertOp(rewriter, loc, fTy.getInput(0), hostAddr);
      aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      aiir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
      llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, conv, sourceFile, sourceLine)};
      auto call = fir::CallOp::create(rewriter, loc, callee, args);
      aiir::Value addr = createConvertOp(rewriter, loc, hostAddr.getType(),
                                         call->getResult(0));
      rewriter.replaceOp(op, addr.getDefiningOp());
      return success();
    }
    return failure();
  }

private:
  const aiir::SymbolTable &symTab;
};

class CUFOpConversionLate
    : public fir::impl::CUFOpConversionLateBase<CUFOpConversionLate> {
  using CUFOpConversionLateBase::CUFOpConversionLateBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    aiir::RewritePatternSet patterns(ctx);
    aiir::ConversionTarget target(*ctx);
    aiir::Operation *op = getOperation();
    aiir::ModuleOp module = aiir::dyn_cast<aiir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();
    aiir::SymbolTable symtab(module);
    target.addLegalDialect<fir::FIROpsDialect, aiir::arith::ArithDialect,
                           aiir::gpu::GPUDialect>();
    patterns.insert<CUFDeviceAddressOpConversion>(patterns.getContext(),
                                                  symtab);
    if (aiir::failed(aiir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace
