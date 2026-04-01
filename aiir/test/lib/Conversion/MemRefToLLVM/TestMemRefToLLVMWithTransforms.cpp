//===- TestMemRefToLLVMWithTransforms.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "aiir/Conversion/LLVMCommon/ConversionTarget.h"
#include "aiir/Conversion/LLVMCommon/LoweringOptions.h"
#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/MemRef/Transforms/Transforms.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {

struct TestMemRefToLLVMWithTransforms
    : public PassWrapper<TestMemRefToLLVMWithTransforms,
                         OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMemRefToLLVMWithTransforms)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final {
    return "test-memref-to-llvm-with-transforms";
  }

  StringRef getDescription() const final {
    return "Tests conversion of MemRef dialects + `func.func` to LLVM dialect "
           "with MemRef transforms.";
  }

  void runOnOperation() override {
    AIIRContext *ctx = &getContext();
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    RewritePatternSet patterns(ctx);
    memref::populateExpandStridedMetadataPatterns(patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace aiir::test {
void registerTestMemRefToLLVMWithTransforms() {
  PassRegistration<TestMemRefToLLVMWithTransforms>();
}
} // namespace aiir::test
