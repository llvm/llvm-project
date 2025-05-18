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

/// Convert a CIR attribute to an LLVM attribute. May use the datalayout for
/// lowering attributes to-be-stored in memory.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::Attribute attr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter);

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage);

class CIRToLLVMBrCondOpLowering
    : public mlir::OpConversionPattern<cir::BrCondOp> {
public:
  using mlir::OpConversionPattern<cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrCondOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCastOpLowering : public mlir::OpConversionPattern<cir::CastOp> {
  mlir::DataLayout const &dataLayout;

  mlir::Type convertTy(mlir::Type ty) const;

public:
  CIRToLLVMCastOpLowering(const mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context,
                          mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(cir::CastOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMReturnOpLowering
    : public mlir::OpConversionPattern<cir::ReturnOp> {
public:
  using mlir::OpConversionPattern<cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ReturnOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCallOpLowering : public mlir::OpConversionPattern<cir::CallOp> {
public:
  using mlir::OpConversionPattern<cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
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
public:
  CIRToLLVMConstantOpLowering(const mlir::TypeConverter &typeConverter,
                              mlir::MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {
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

class CIRToLLVMSwitchFlatOpLowering
    : public mlir::OpConversionPattern<cir::SwitchFlatOp> {
public:
  using mlir::OpConversionPattern<cir::SwitchFlatOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SwitchFlatOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGetGlobalOpLowering
    : public mlir::OpConversionPattern<cir::GetGlobalOp> {
public:
  using mlir::OpConversionPattern<cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetGlobalOp op, OpAdaptor,
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

class CIRToLLVMUnaryOpLowering
    : public mlir::OpConversionPattern<cir::UnaryOp> {
public:
  using mlir::OpConversionPattern<cir::UnaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::UnaryOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBinOpLowering : public mlir::OpConversionPattern<cir::BinOp> {
  mlir::LLVM::IntegerOverflowFlags getIntOverflowFlag(cir::BinOp op) const;

public:
  using mlir::OpConversionPattern<cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BinOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMCmpOpLowering : public mlir::OpConversionPattern<cir::CmpOp> {
public:
  CIRToLLVMCmpOpLowering(const mlir::TypeConverter &typeConverter,
                         mlir::MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {
    setHasBoundedRewriteRecursion();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CmpOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMShiftOpLowering
    : public mlir::OpConversionPattern<cir::ShiftOp> {
public:
  using mlir::OpConversionPattern<cir::ShiftOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ShiftOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMSelectOpLowering
    : public mlir::OpConversionPattern<cir::SelectOp> {
public:
  using mlir::OpConversionPattern<cir::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SelectOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMBrOpLowering : public mlir::OpConversionPattern<cir::BrOp> {
public:
  using mlir::OpConversionPattern<cir::BrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMGetMemberOpLowering
    : public mlir::OpConversionPattern<cir::GetMemberOp> {
public:
  using mlir::OpConversionPattern<cir::GetMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetMemberOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMTrapOpLowering : public mlir::OpConversionPattern<cir::TrapOp> {
public:
  using mlir::OpConversionPattern<cir::TrapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TrapOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMPtrStrideOpLowering
    : public mlir::OpConversionPattern<cir::PtrStrideOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRToLLVMPtrStrideOpLowering(const mlir::TypeConverter &typeConverter,
                               mlir::MLIRContext *context,
                               mlir::DataLayout const &dataLayout)
      : OpConversionPattern(typeConverter, context), dataLayout(dataLayout) {}
  using mlir::OpConversionPattern<cir::PtrStrideOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::PtrStrideOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMStackSaveOpLowering
    : public mlir::OpConversionPattern<cir::StackSaveOp> {
public:
  using mlir::OpConversionPattern<cir::StackSaveOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::StackSaveOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMStackRestoreOpLowering
    : public mlir::OpConversionPattern<cir::StackRestoreOp> {
public:
  using OpConversionPattern<cir::StackRestoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::StackRestoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

class CIRToLLVMVecCreateOpLowering
    : public mlir::OpConversionPattern<cir::VecCreateOp> {
public:
  using mlir::OpConversionPattern<cir::VecCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCreateOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecExtractOpLowering
    : public mlir::OpConversionPattern<cir::VecExtractOp> {
public:
  using mlir::OpConversionPattern<cir::VecExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecExtractOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

class CIRToLLVMVecInsertOpLowering
    : public mlir::OpConversionPattern<cir::VecInsertOp> {
public:
  using mlir::OpConversionPattern<cir::VecInsertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecInsertOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &) const override;
};

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H
