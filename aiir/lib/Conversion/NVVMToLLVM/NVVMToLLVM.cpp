//===- NVVMToLLVM.cpp - NVVM to LLVM dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation NVVM ops which is not supported in LLVM
// core.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/NVVMToLLVM/NVVMToLLVM.h"

#include "aiir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/TypeUtilities.h"
#include "aiir/IR/Value.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "nvvm-to-llvm"

namespace aiir {
#define GEN_PASS_DEF_CONVERTNVVMTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace NVVM;

namespace {

struct PtxLowering
    : public OpInterfaceRewritePattern<BasicPtxBuilderInterface> {
  using OpInterfaceRewritePattern<
      BasicPtxBuilderInterface>::OpInterfaceRewritePattern;

  PtxLowering(AIIRContext *context, PatternBenefit benefit = 2)
      : OpInterfaceRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(BasicPtxBuilderInterface op,
                                PatternRewriter &rewriter) const override {
    if (op.hasIntrinsic()) {
      LDBG() << "Ptx Builder does not lower \n\t" << op;
      return failure();
    }

    SmallVector<std::pair<Value, PTXRegisterMod>> asmValues;
    LDBG() << op.getPtx();

    bool needsManualMapping = op.getAsmValues(rewriter, asmValues);
    PtxBuilder generator(op, rewriter, needsManualMapping);
    for (auto &[asmValue, modifier] : asmValues) {
      LDBG() << asmValue << "\t Modifier : " << modifier;
      if (failed(generator.insertValue(asmValue, modifier)))
        return failure();
    }

    generator.buildAndReplaceOp();
    return success();
  }
};

struct ConvertNVVMToLLVMPass
    : public impl::ConvertNVVMToLLVMPassBase<ConvertNVVMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<::aiir::LLVM::LLVMDialect>();
    RewritePatternSet pattern(&getContext());
    aiir::populateNVVMToLLVMConversionPatterns(pattern);
    if (failed(
            applyPartialConversion(getOperation(), target, std::move(pattern))))
      signalPassFailure();
  }
};

/// Implement the interface to convert NVVM to LLVM.
struct NVVMToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  NVVMToLLVMDialectInterface(Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}

  void loadDependentDialects(AIIRContext *context) const final {
    context->loadDialect<NVVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateNVVMToLLVMConversionPatterns(patterns);
  }
};

} // namespace

void aiir::populateNVVMToLLVMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<PtxLowering>(patterns.getContext());
}

void aiir::registerConvertNVVMToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, NVVMDialect *dialect) {
    dialect->addInterfaces<NVVMToLLVMDialectInterface>();
  });
}
