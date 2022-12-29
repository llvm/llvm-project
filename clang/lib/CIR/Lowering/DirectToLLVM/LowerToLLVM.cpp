//====- LowerToLLVM.cpp - Lowering from CIR to LLVMIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to LLVMIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/Sequence.h"

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

class CIRBrCondOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BrCondOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::LLVM::TruncOp>(
        brOp.getLoc(), rewriter.getI1Type(), condition);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
        brOp, i1Condition.getResult(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRCastOpLowering : public mlir::OpConversionPattern<mlir::cir::CastOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::CastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp castOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    switch (castOp.getKind()) {
    case mlir::cir::CastKind::int_to_bool: {
      auto zero = rewriter.create<mlir::cir::ConstantOp>(
          src.getLoc(), src.getType(),
          mlir::IntegerAttr::get(src.getType(), 0));
      rewriter.replaceOpWithNewOp<mlir::cir::CmpOp>(
          castOp, src.getType(), mlir::cir::CmpOpKind::ne, src, zero);
      break;
    }
    default:
      llvm_unreachable("NYI");
    }

    return mlir::success();
  }
};

class CIRIfLowering : public mlir::OpConversionPattern<mlir::cir::IfOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::IfOp ifOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &thenRegion = ifOp.getThenRegion();
    auto &elseRegion = ifOp.getElseRegion();

    (void)thenRegion;
    (void)elseRegion;

    mlir::OpBuilder::InsertionGuard guard(rewriter);

    [[maybe_unused]] auto loc = ifOp.getLoc();

    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (ifOp->getResults().size() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region
    [[maybe_unused]] auto *thenBeforeBody = &ifOp.getThenRegion().front();
    [[maybe_unused]] auto *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<mlir::cir::YieldOp>(thenAfterBody->getTerminator())) {
      [[maybe_unused]] auto thenBranchOp =
          rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
              thenYieldOp, thenYieldOp.getArgs(), continueBlock);
    } else if (auto thenReturnOp = dyn_cast<mlir::cir::ReturnOp>(
                   thenAfterBody->getTerminator())) {
      ;
    } else {
      llvm_unreachable("what are we terminating with?");
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Inline then region
    [[maybe_unused]] auto *elseBeforeBody = &ifOp.getElseRegion().front();
    [[maybe_unused]] auto *elseAfterBody = &ifOp.getElseRegion().back();
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), thenAfterBody);

    rewriter.setInsertionPointToEnd(currentBlock);
    auto trunc = rewriter.create<mlir::LLVM::TruncOp>(loc, rewriter.getI1Type(),
                                                      adaptor.getCondition());
    rewriter.create<mlir::LLVM::CondBrOp>(loc, trunc.getRes(), thenBeforeBody,
                                          elseBeforeBody);

    rewriter.setInsertionPointToEnd(elseAfterBody);
    if (auto elseYieldOp =
            dyn_cast<mlir::cir::YieldOp>(elseAfterBody->getTerminator())) {
      [[maybe_unused]] auto elseBranchOp =
          rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
              elseYieldOp, elseYieldOp.getArgs(), continueBlock);
    } else if (auto elseReturnOp = dyn_cast<mlir::cir::ReturnOp>(
                   elseAfterBody->getTerminator())) {
      ;
    } else {
      llvm_unreachable("what are we terminating with?");
    }

    rewriter.setInsertionPoint(elseAfterBody->getTerminator());
    rewriter.replaceOp(ifOp, continueBlock->getArguments());

    return mlir::success();
  }
};

class CIRScopeOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ScopeOp> {
public:
  using OpConversionPattern<mlir::cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ScopeOp scopeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = scopeOp.getLoc();

    // Split the current block before the ScopeOp to create the inlining point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (scopeOp.getNumResults() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline body region.
    auto *beforeBody = &scopeOp.getRegion().front();
    auto *afterBody = &scopeOp.getRegion().back();
    rewriter.inlineRegionBefore(scopeOp.getRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    // TODO(CIR): stackSaveOp
    // auto stackSaveOp = rewriter.create<mlir::LLVM::StackSaveOp>(
    //     loc, mlir::LLVM::LLVMPointerType::get(
    //              mlir::IntegerType::get(scopeOp.getContext(), 8)));
    rewriter.create<mlir::cir::BrOp>(loc, mlir::ValueRange(), beforeBody);

    // Replace the scopeop return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    auto yieldOp = cast<mlir::cir::YieldOp>(afterBody->getTerminator());
    auto branchOp = rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        yieldOp, yieldOp.getArgs(), continueBlock);

    // // Insert stack restore before jumping out of the body of the region.
    rewriter.setInsertionPoint(branchOp);
    // TODO(CIR): stackrestore?
    // rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSaveOp);

    // Replace the op with values return from the body region.
    rewriter.replaceOp(scopeOp, continueBlock->getArguments());

    return mlir::success();
  }
};

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

struct ConvertCIRToLLVMPass
    : public mlir::PassWrapper<ConvertCIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect,
                    mlir::func::FuncDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-llvm"; }
};

class CIRCallLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
public:
  using OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, op.getResultTypes(), op.getCalleeAttr(), op.getArgOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRAllocaLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
public:
  using OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto elementTy = getTypeConverter()->convertType(op.getAllocaType());

    mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

    auto resultTy = mlir::LLVM::LLVMPointerType::get(getContext());

    rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
        op, resultTy, elementTy, one, op.getAlignmentAttr().getInt());
    return mlir::LogicalResult::success();
  }
};

class CIRLoadLowering : public mlir::OpConversionPattern<mlir::cir::LoadOp> {
public:
  using OpConversionPattern<mlir::cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto llvmTy =
        getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, llvmTy,
                                                    adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
public:
  using OpConversionPattern<mlir::cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(),
                                                     adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRConstantLowering
    : public mlir::OpConversionPattern<mlir::cir::ConstantOp> {
public:
  using OpConversionPattern<mlir::cir::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, getTypeConverter()->convertType(op.getType()), op.getValue());
    return mlir::LogicalResult::success();
  }
};

class CIRFuncLowering : public mlir::OpConversionPattern<mlir::cir::FuncOp> {
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

    mlir::Type resultType;
    if (fnType.getNumResults() == 1) {
      resultType = getTypeConverter()->convertType(fnType.getResult(0));
      if (!resultType)
        return mlir::failure();
    }

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
    mlir::Type type = op.getInput().getType();
    assert(type.isa<mlir::IntegerType>() && "operand type not supported yet");

    switch (op.getKind()) {
    case mlir::cir::UnaryOpKind::Inc: {
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, op.getType(),
                                                     op.getInput(), One);
      break;
    }
    case mlir::cir::UnaryOpKind::Dec: {
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, op.getType(),
                                                     op.getInput(), One);
      break;
    }
    case mlir::cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, op.getInput());
      break;
    }
    case mlir::cir::UnaryOpKind::Minus: {
      auto Zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, op.getType(), Zero,
                                                     op.getInput());
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
    assert((op.getLhs().getType() == op.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type type = op.getRhs().getType();
    assert((type.isa<mlir::IntegerType>() || type.isa<mlir::FloatType>()) &&
           "operand type not supported yet");

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Sub:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Mul:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::MulOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Div:
      if (type.isa<mlir::IntegerType>()) {
        if (type.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::LLVM::SDivOp>(
              op, op.getType(), op.getLhs(), op.getRhs());
        else
          llvm_unreachable("integer type not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FDivOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Rem:
      if (type.isa<mlir::IntegerType>()) {
        if (type.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::LLVM::SRemOp>(
              op, op.getType(), op.getLhs(), op.getRhs());
        else
          llvm_unreachable("integer type not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FRemOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(op, op.getType(),
                                                     op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, op.getType(),
                                                    op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, op.getType(),
                                                     op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Shl:
      rewriter.replaceOpWithNewOp<mlir::LLVM::ShlOp>(op, op.getType(),
                                                     op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Shr:
      if (type.isSignlessInteger())
        rewriter.replaceOpWithNewOp<mlir::LLVM::AShrOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        llvm_unreachable("integer type not supported in CIR yet");
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<mlir::cir::CmpOp> {
public:
  using OpConversionPattern<mlir::cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp cmpOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getLhs().getType();
    auto i1Type =
        mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless);
    auto i8Type =
        mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::Signless);

    switch (adaptor.getKind()) {
    case mlir::cir::CmpOpKind::gt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::ugt;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ugt),
            adaptor.getLhs(), adaptor.getRhs(),
            // TODO(CIR): These fastmath flags need to not be defaulted.
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ge: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::uge;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::uge),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::lt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::ult;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ult),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::le: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::ule;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ule),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::eq: {
      if (type.isa<mlir::IntegerType>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::ICmpPredicate::eq),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ueq),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ne: {
      if (type.isa<mlir::IntegerType>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::ICmpPredicate::ne),
            adaptor.getLhs(), adaptor.getRhs());

        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::une),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, i8Type,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpRewritePattern<mlir::cir::BrOp> {
public:
  using OpRewritePattern<mlir::cir::BrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, op.getDestOperands(),
                                                  op.getDest());
    return mlir::LogicalResult::success();
  }
};

void populateCIRToLLVMConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRBrOpLowering, CIRReturnLowering>(patterns.getContext());
  patterns.add<CIRCmpOpLowering, CIRBrCondOpLowering, CIRCallLowering,
               CIRUnaryOpLowering, CIRBinOpLowering, CIRLoadLowering,
               CIRConstantLowering, CIRStoreLowering, CIRAllocaLowering,
               CIRFuncLowering, CIRScopeOpLowering, CIRCastOpLowering,
               CIRIfLowering>(converter, patterns.getContext());
}

static void prepareTypeConverter(mlir::LLVMTypeConverter &converter) {
  converter.addConversion([&](mlir::cir::PointerType type) -> mlir::Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](mlir::cir::ArrayType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](mlir::cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8,
                                  mlir::IntegerType::Signless);
  });
}

void ConvertCIRToLLVMPass::runOnOperation() {
  auto module = getOperation();

  mlir::LLVMTypeConverter converter(&getContext());
  prepareTypeConverter(converter);

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRToLLVMConversionPatterns(patterns, converter);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::cir::CIRDialect, mlir::func::FuncDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule,
                             std::unique_ptr<mlir::MLIRContext> mlirCtx,
                             LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createConvertCIRToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  mlir::registerLLVMDialectTranslation(*mlirCtx);

  LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}
} // namespace direct
} // namespace cir
