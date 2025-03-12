//====- LowerToLLVM.h- Lowering from CIR to LLVM --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for converting CIR modules to LLVM IR.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_LOWERTOLLVM_H
#define CLANG_CIR_LOWERTOLLVM_H

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {

namespace direct {

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage);

class CIRToLLVMReturnOpLowering
    : public mlir::OpConversionPattern<cir::ReturnOp> {
public:
  using mlir::OpConversionPattern<cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ReturnOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMAllocaOpLowering
    : public mlir::OpConversionPattern<cir::AllocaOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMAllocaOpLowering(mlir::TypeConverter const &typeConverter,
                            mlir::MLIRContext *context,
                            mlir::DataLayout const &dataLayout)
      : OpConversionPattern<cir::AllocaOp>(typeConverter, context),
        dataLayout(dataLayout) {}

  using mlir::OpConversionPattern<cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AllocaOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMLoadOpLowering : public mlir::OpConversionPattern<cir::LoadOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMLoadOpLowering(const mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context,
                          mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(cir::LoadOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMStoreOpLowering
    : public mlir::OpConversionPattern<cir::StoreOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMStoreOpLowering(const mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context,
                           mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(cir::StoreOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMConstantOpLowering
    : public mlir::OpConversionPattern<cir::ConstantOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMConstantOpLowering(const mlir::TypeConverter &typeConverter,
                              mlir::MLIRContext *context,
                              mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {
    setHasBoundedRewriteRecursion();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::ConstantOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMFuncOpLowering : public mlir::OpConversionPattern<cir::FuncOp> {
  static mlir::StringRef getLinkageAttrNameString() { return "linkage"; }

  void lowerFuncAttributes(
      cir::FuncOp func, bool filterArgAndResAttrs,
      mlir::SmallVectorImpl<mlir::NamedAttribute> &result) const;

public:
  using mlir::OpConversionPattern<cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::FuncOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGlobalOpLowering
    : public mlir::OpConversionPattern<cir::GlobalOp> {
  const mlir::DataLayout &dataLayout;

public:
  CIRToLLVMGlobalOpLowering(const mlir::TypeConverter &typeConverter,
                            mlir::MLIRContext *context,
                            const mlir::DataLayout &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {
    setHasBoundedRewriteRecursion();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;

private:
  mlir::LogicalResult matchAndRewriteRegionInitializedGlobal(
      cir::GlobalOp op, mlir::Attribute init,
      mlir::ConversionPatternRewriter &rewriter) const;

  void setupRegionInitializedLLVMGlobalOp(
      cir::GlobalOp op, mlir::ConversionPatternRewriter &rewriter) const;
};

class CIRToLLVMBrOpLowering : public mlir::OpConversionPattern<cir::BrOp> {
public:
  using mlir::OpConversionPattern<cir::BrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMTrapOpLowering : public mlir::OpConversionPattern<cir::TrapOp> {
public:
  using mlir::OpConversionPattern<cir::TrapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TrapOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H
