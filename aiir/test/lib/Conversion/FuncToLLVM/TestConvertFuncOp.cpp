//===- TestConvertFuncOp.cpp - Test LLVM Conversion of Func FuncOp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"

#include "aiir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "aiir/Conversion/LLVMCommon/ConversionTarget.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {

/// Test helper Conversion Pattern to directly call `convertFuncOpToLLVMFuncOp`
/// to verify this utility function includes all functionalities of conversion
struct FuncOpConversion : public ConvertOpToLLVMPattern<func::FuncOp> {
  FuncOpConversion(const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<LLVM::LLVMFuncOp> newFuncOp = aiir::convertFuncOpToLLVMFuncOp(
        cast<FunctionOpInterface>(funcOp.getOperation()), rewriter,
        *getTypeConverter());
    if (failed(newFuncOp))
      return rewriter.notifyMatchFailure(funcOp, "Could not convert funcop");

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ReturnOpConversion : public ConvertOpToLLVMPattern<func::ReturnOp> {
  ReturnOpConversion(const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter) {}

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resTys;
    if (failed(typeConverter->convertTypes(returnOp->getResultTypes(), resTys)))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp, resTys,
                                                adaptor.getOperands());
    return success();
  }
};

static std::optional<Type>
convertSimpleATypeToStruct(test::SimpleAType simpleTy) {
  AIIRContext *ctx = simpleTy.getContext();
  SmallVector<Type> memberTys(2, IntegerType::get(ctx, /*width=*/8));
  return LLVM::LLVMStructType::getLiteral(ctx, memberTys);
}

struct TestConvertFuncOp
    : public PassWrapper<TestConvertFuncOp, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertFuncOp)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final { return "test-convert-func-op"; }

  StringRef getDescription() const final {
    return "Tests conversion of `func.func` to `llvm.func` for different "
           "attributes";
  }

  void runOnOperation() override {
    AIIRContext *ctx = &getContext();

    LowerToLLVMOptions options(ctx);
    // Populate type conversions.
    LLVMTypeConverter typeConverter(ctx, options);
    typeConverter.addConversion(convertSimpleATypeToStruct);

    RewritePatternSet patterns(ctx);
    patterns.add<FuncOpConversion>(typeConverter);
    patterns.add<ReturnOpConversion>(typeConverter);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace aiir::test {
void registerConvertFuncOpPass() { PassRegistration<TestConvertFuncOp>(); }
} // namespace aiir::test
