//====- LowerCIRToMLIR.cpp - Lowering from CIR to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cir;
using namespace llvm;

namespace cir {

class CIRReturnLowering
    : public mlir::OpConversionPattern<mlir::cir::ReturnOp> {
public:
  using OpConversionPattern<mlir::cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::math::MathDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRCallOpLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
public:
  using OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, mlir::SymbolRefAttr::get(op), types, adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRAllocaOpLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
public:
  using OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getAllocaType();
    auto mlirType = getTypeConverter()->convertType(type);

    // FIXME: Some types can not be converted yet (e.g. struct)
    if (!mlirType)
      return mlir::LogicalResult::failure();

    auto memreftype = mlir::MemRefType::get({}, mlirType);
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memreftype,
                                                        op.getAlignmentAttr());
    return mlir::LogicalResult::success();
  }
};

class CIRLoadOpLowering : public mlir::OpConversionPattern<mlir::cir::LoadOp> {
public:
  using OpConversionPattern<mlir::cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRStoreOpLowering
    : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
public:
  using OpConversionPattern<mlir::cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, adaptor.getValue(),
                                                       adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRCosOpLowering : public mlir::OpConversionPattern<mlir::cir::CosOp> {
public:
  using OpConversionPattern<mlir::cir::CosOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CosOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::CosOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRConstantOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ConstantOp> {
public:
  using OpConversionPattern<mlir::cir::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = getTypeConverter()->convertType(op.getType());
    mlir::TypedAttr value;
    if (mlir::isa<mlir::cir::BoolType>(op.getType())) {
      auto boolValue = mlir::cast<mlir::cir::BoolAttr>(op.getValue());
      value = rewriter.getIntegerAttr(ty, boolValue.getValue());
    } else if (op.getType().isa<mlir::cir::CIRFPTypeInterface>()) {
      assert(ty.isF32() || ty.isF64() && "NYI");
      value = rewriter.getFloatAttr(
          typeConverter->convertType(op.getType()),
          op.getValue().cast<mlir::cir::FPAttr>().getValue());
    } else {
      auto cirIntAttr = mlir::dyn_cast<mlir::cir::IntAttr>(op.getValue());
      assert(cirIntAttr && "NYI non cir.int attr");
      value = rewriter.getIntegerAttr(
          ty, cast<mlir::cir::IntAttr>(op.getValue()).getValue());
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, ty, value);
    return mlir::LogicalResult::success();
  }
};

class CIRFuncOpLowering : public mlir::OpConversionPattern<mlir::cir::FuncOp> {
public:
  using OpConversionPattern<mlir::cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto fnType = op.getFunctionType();
    mlir::TypeConverter::SignatureConversion signatureConversion(
        fnType.getNumInputs());

    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = typeConverter->convertType(argType.value());
      if (!convertedType)
        return mlir::failure();
      signatureConversion.addInputs(argType.index(), convertedType);
    }

    mlir::Type resultType =
        getTypeConverter()->convertType(fnType.getReturnType());
    auto fn = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getName(),
        rewriter.getFunctionType(signatureConversion.getConvertedTypes(),
                                 resultType ? mlir::TypeRange(resultType)
                                            : mlir::TypeRange()));

    rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
    if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                           &signatureConversion)))
      return mlir::failure();

    rewriter.eraseOp(op);

    return mlir::LogicalResult::success();
  }
};

class CIRUnaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::UnaryOp> {
public:
  using OpConversionPattern<mlir::cir::UnaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto type = getTypeConverter()->convertType(op.getType());

    switch (op.getKind()) {
    case mlir::cir::UnaryOpKind::Inc: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, type, input, One);
      break;
    }
    case mlir::cir::UnaryOpKind::Dec: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, input, One);
      break;
    }
    case mlir::cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, op.getInput());
      break;
    }
    case mlir::cir::UnaryOpKind::Minus: {
      auto Zero = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 0));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, Zero, input);
      break;
    }
    case mlir::cir::UnaryOpKind::Not: {
      auto MinusOne = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, -1));
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, type, MinusOne,
                                                       input);
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBinOpLowering : public mlir::OpConversionPattern<mlir::cir::BinOp> {
public:
  using OpConversionPattern<mlir::cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert((adaptor.getLhs().getType() == adaptor.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
    assert((mlirType.isa<mlir::IntegerType>() ||
            mlirType.isa<mlir::FloatType>()) &&
           "operand type not supported yet");

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (mlirType.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Sub:
      if (mlirType.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Mul:
      if (mlirType.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Div:
      if (mlirType.isa<mlir::IntegerType>()) {
        if (mlirType.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          llvm_unreachable("integer mlirType not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Rem:
      if (mlirType.isa<mlir::IntegerType>()) {
        if (mlirType.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          llvm_unreachable("integer mlirType not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<mlir::cir::CmpOp> {
public:
  using OpConversionPattern<mlir::cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getLhs().getType();
    auto integerType =
        mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless);

    mlir::Value mlirResult;
    switch (op.getKind()) {
    case mlir::cir::CmpOpKind::gt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ugt;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGT),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ge: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::uge;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGE),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::lt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ult;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULT),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::le: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ule;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULE),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::eq: {
      if (type.isa<mlir::IntegerType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::eq),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UEQ),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ne: {
      if (type.isa<mlir::IntegerType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::ne),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UNE),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    }

    // MLIR comparison ops return i1, but cir::CmpOp returns the same type as
    // the LHS value. Since this return value can be used later, we need to
    // restore the type with the extension below.
    auto mlirResultTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, mlirResultTy,
                                                      mlirResult);

    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpRewritePattern<mlir::cir::BrOp> {
public:
  using OpRewritePattern<mlir::cir::BrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest());
    return mlir::LogicalResult::success();
  }
};

class CIRScopeOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ScopeOp> {
  using mlir::OpConversionPattern<mlir::cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ScopeOp scopeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Empty scope: just remove it.
    if (scopeOp.getRegion().empty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    for (auto &block : scopeOp.getRegion()) {
      rewriter.setInsertionPointToEnd(&block);
      auto *terminator = block.getTerminator();
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(
          terminator, terminator->getOperands());
    }

    SmallVector<mlir::Type> mlirResultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(scopeOp->getResultTypes(),
                                                      mlirResultTypes)))
      return mlir::LogicalResult::failure();

    rewriter.setInsertionPoint(scopeOp);
    auto newScopeOp = rewriter.create<mlir::memref::AllocaScopeOp>(
        scopeOp.getLoc(), mlirResultTypes);
    rewriter.inlineRegionBefore(scopeOp.getScopeRegion(),
                                newScopeOp.getBodyRegion(),
                                newScopeOp.getBodyRegion().end());
    rewriter.replaceOp(scopeOp, newScopeOp);

    return mlir::LogicalResult::success();
  }
};

struct CIRBrCondOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BrCondOp> {
  using mlir::OpConversionPattern<mlir::cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::arith::TruncIOp>(
        brOp.getLoc(), rewriter.getI1Type(), condition);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        brOp, i1Condition.getResult(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRTernaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::TernaryOp> {
public:
  using OpConversionPattern<mlir::cir::TernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::arith::TruncIOp>(
        op.getLoc(), rewriter.getI1Type(), condition);
    SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return mlir::failure();

    auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), resultTypes,
                                                 i1Condition.getResult(), true);
    auto *thenBlock = &ifOp.getThenRegion().front();
    auto *elseBlock = &ifOp.getElseRegion().front();
    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), thenBlock,
                               thenBlock->end());
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), elseBlock,
                               elseBlock->end());

    rewriter.replaceOp(op, ifOp);
    return mlir::success();
  }
};

class CIRYieldOpLowering
    : public mlir::OpConversionPattern<mlir::cir::YieldOp> {
public:
  using OpConversionPattern<mlir::cir::YieldOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::IfOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering, CIRBrOpLowering>(patterns.getContext());

  patterns.add<CIRCmpOpLowering, CIRCallOpLowering, CIRUnaryOpLowering,
               CIRBinOpLowering, CIRLoadOpLowering, CIRConstantOpLowering,
               CIRStoreOpLowering, CIRAllocaOpLowering, CIRFuncOpLowering,
               CIRScopeOpLowering, CIRBrCondOpLowering, CIRTernaryOpLowering,
               CIRYieldOpLowering, CIRCosOpLowering>(converter,
                                                     patterns.getContext());
}

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](mlir::cir::PointerType type) -> mlir::Type {
    auto ty = converter.convertType(type.getPointee());
    // FIXME: The pointee type might not be converted (e.g. struct)
    if (!ty)
      return nullptr;
    return mlir::MemRefType::get({}, ty);
  });
  converter.addConversion(
      [&](mlir::IntegerType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::FloatType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::cir::VoidType type) -> mlir::Type { return {}; });
  converter.addConversion([&](mlir::cir::IntType type) -> mlir::Type {
    // arith dialect ops doesn't take signed integer -- drop cir sign here
    return mlir::IntegerType::get(
        type.getContext(), type.getWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  });
  converter.addConversion([&](mlir::cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8);
  });
  converter.addConversion([&](mlir::cir::SingleType type) -> mlir::Type {
    return mlir::FloatType::getF32(type.getContext());
  });
  converter.addConversion([&](mlir::cir::DoubleType type) -> mlir::Type {
    return mlir::FloatType::getF64(type.getContext());
  });
  converter.addConversion([&](mlir::cir::ArrayType type) -> mlir::Type {
    auto elementType = converter.convertType(type.getEltType());
    return mlir::MemRefType::get(type.getSize(), elementType);
  });

  return converter;
}

void ConvertCIRToMLIRPass::runOnOperation() {
  auto module = getOperation();

  auto converter = prepareTypeConverter();

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRToMLIRConversionPatterns(patterns, converter);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                         mlir::math::MathDialect>();
  target.addIllegalDialect<mlir::cir::CIRDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<llvm::Module>
lowerFromCIRToMLIRToLLVMIR(mlir::ModuleOp theModule,
                           std::unique_ptr<mlir::MLIRContext> mlirCtx,
                           LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createConvertCIRToMLIRPass());
  pm.addPass(createConvertMLIRToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  mlir::registerOpenMPDialectTranslation(*mlirCtx);

  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}

std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass() {
  return std::make_unique<ConvertCIRToMLIRPass>();
}

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp theModule,
                                  mlir::MLIRContext *mlirCtx) {
  mlir::PassManager pm(mlirCtx);

  pm.addPass(createConvertCIRToMLIRPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  return theModule;
}

} // namespace cir
