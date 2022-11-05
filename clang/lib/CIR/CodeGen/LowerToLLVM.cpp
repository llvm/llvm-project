//====- LowerToLLVM.cpp - Lowering from CIR to LLVM -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of CIR operations to LLVMIR.
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

struct ConvertCIRToLLVMPass
    : public mlir::PassWrapper<ConvertCIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-llvm"; }
};

struct ConvertCIRToMemRefPass
    : public mlir::PassWrapper<ConvertCIRToMemRefPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-memref"; }
};

struct ConvertCIRToFuncPass
    : public mlir::PassWrapper<ConvertCIRToFuncPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::cir::CIRDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-func"; }
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

class CIRCallLowering : public mlir::OpRewritePattern<mlir::cir::CallOp> {
public:
  using OpRewritePattern<mlir::cir::CallOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, mlir::SymbolRefAttr::get(op), op.getResultTypes(),
        op.getArgOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRAllocaLowering : public mlir::OpRewritePattern<mlir::cir::AllocaOp> {
public:
  using OpRewritePattern<mlir::cir::AllocaOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto type = op.getAllocaType();
    mlir::MemRefType memreftype;

    if (type.isa<mlir::cir::BoolType>()) {
      auto integerType =
          mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::Signless);
      memreftype = mlir::MemRefType::get({}, integerType);
    } else if (type.isa<mlir::cir::ArrayType>()) {
      mlir::cir::ArrayType arraytype = type.dyn_cast<mlir::cir::ArrayType>();
      memreftype =
          mlir::MemRefType::get(arraytype.getSize(), arraytype.getEltType());
    } else if (type.isa<mlir::IntegerType>() || type.isa<mlir::FloatType>()) {
      memreftype = mlir::MemRefType::get({}, op.getAllocaType());
    } else {
      llvm_unreachable("type to be allocated not supported yet");
    }
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memreftype,
                                                        op.getAlignmentAttr());
    return mlir::LogicalResult::success();
  }
};

class CIRLoadLowering : public mlir::ConversionPattern {
public:
  CIRLoadLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(mlir::cir::LoadOp::getOperationName(), 1, ctx) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, operands[0]);
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::ConversionPattern {
public:
  CIRStoreLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(mlir::cir::StoreOp::getOperationName(), 1,
                                ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, operands[0],
                                                       operands[1]);
    return mlir::LogicalResult::success();
  }
};

class CIRConstantLowering
    : public mlir::OpRewritePattern<mlir::cir::ConstantOp> {
public:
  using OpRewritePattern<mlir::cir::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getType().isa<mlir::cir::BoolType>()) {
      mlir::Type type =
          mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::Signless);
      mlir::TypedAttr IntegerAttr;
      if (op.getValue() == mlir::BoolAttr::get(getContext(), true))
        IntegerAttr = mlir::IntegerAttr::get(type, 1);
      else
        IntegerAttr = mlir::IntegerAttr::get(type, 0);
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, type,
                                                           IntegerAttr);
    } else
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op.getType(),
                                                           op.getValue());
    return mlir::LogicalResult::success();
  }
};

class CIRFuncLowering : public mlir::OpConversionPattern<mlir::cir::FuncOp> {
public:
  using OpConversionPattern<mlir::cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto fn = rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(
        op, op.getName(), op.getFunctionType());
    auto &srcRegion = op.getBody();
    auto &dstRegion = fn.getBody();

    mlir::TypeConverter::SignatureConversion signatureConversion(
        op.front().getNumArguments());

    rewriter.inlineRegionBefore(srcRegion, dstRegion, fn.end());
    if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                           &signatureConversion)))
      return mlir::failure();

    return mlir::LogicalResult::success();
  }
};

class CIRUnaryOpLowering : public mlir::OpRewritePattern<mlir::cir::UnaryOp> {
public:
  using OpRewritePattern<mlir::cir::UnaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type type = op.getInput().getType();
    assert(type.isa<mlir::IntegerType>() && "operand type not supported yet");

    switch (op.getKind()) {
      llvm_unreachable("NYI");
    case mlir::cir::UnaryOpKind::Inc: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, op.getType(),
                                                       op.getInput(), One);
      break;
    }
    case mlir::cir::UnaryOpKind::Dec: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, op.getType(),
                                                       op.getInput(), One);
      break;
    }
    case mlir::cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, op.getInput());
      break;
    }
    case mlir::cir::UnaryOpKind::Minus: {
      auto Zero = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 0));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, op.getType(), Zero,
                                                       op.getInput());
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBinOpLowering : public mlir::OpRewritePattern<mlir::cir::BinOp> {
public:
  using OpRewritePattern<mlir::cir::BinOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BinOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert((op.getLhs().getType() == op.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type type = op.getRhs().getType();
    assert((type.isa<mlir::IntegerType>() || type.isa<mlir::FloatType>()) &&
           "operand type not supported yet");

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Sub:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Mul:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Div:
      if (type.isa<mlir::IntegerType>()) {
        if (type.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(
              op, op.getType(), op.getLhs(), op.getRhs());
        else
          llvm_unreachable("integer type not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Rem:
      if (type.isa<mlir::IntegerType>()) {
        if (type.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::RemSIOp>(
              op, op.getType(), op.getLhs(), op.getRhs());
        else
          llvm_unreachable("integer type not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(
          op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(op, op.getType(),
                                                      op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
          op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Shl:
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(
          op, op.getType(), op.getLhs(), op.getRhs());
      break;
    case mlir::cir::BinOpKind::Shr:
      if (type.isSignlessInteger())
        rewriter.replaceOpWithNewOp<mlir::arith::ShRSIOp>(
            op, op.getType(), op.getLhs(), op.getRhs());
      else
        llvm_unreachable("integer type not supported in CIR yet");
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpRewritePattern<mlir::cir::CmpOp> {
public:
  using OpRewritePattern<mlir::cir::CmpOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto type = op.getLhs().getType();
    auto integerType =
        mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless);

    switch (op.getKind()) {
    case mlir::cir::CmpOpKind::gt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ugt;
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGT),
            op.getLhs(), op.getRhs(),
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
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGE),
            op.getLhs(), op.getRhs(),
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
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULT),
            op.getLhs(), op.getRhs(),
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
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULE),
            op.getLhs(), op.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::eq: {
      if (type.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::eq),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UEQ),
            op.getLhs(), op.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ne: {
      if (type.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::ne),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UNE),
            op.getLhs(), op.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
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
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest());
    return mlir::LogicalResult::success();
  }
};

void populateCIRToMemRefConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<CIRAllocaLowering, CIRLoadLowering, CIRStoreLowering,
               CIRConstantLowering, CIRUnaryOpLowering, CIRBinOpLowering,
               CIRCmpOpLowering, CIRBrOpLowering>(patterns.getContext());
}

void ConvertCIRToLLVMPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void ConvertCIRToMemRefPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  // TODO: Should this be a wholesale conversion? It's a bit ambiguous on
  // whether we should have micro-conversions that do the minimal amount of work
  // or macro conversions that entiirely remove a dialect.
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();
  target
      .addIllegalOp<mlir::cir::BinOp, mlir::cir::ReturnOp, mlir::cir::AllocaOp,
                    mlir::cir::LoadOp, mlir::cir::StoreOp,
                    mlir::cir::ConstantOp, mlir::cir::CmpOp, mlir::cir::BrOp>();

  mlir::RewritePatternSet patterns(&getContext());
  populateCIRToMemRefConversionPatterns(patterns);
  // populateAffineToStdConversionPatterns(patterns);
  // populateLoopToStdConversionPatterns(patterns);

  auto module = getOperation();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void ConvertCIRToFuncPass::runOnOperation() {
  // End goal here is to legalize to builtin.func, func.return, func.call.
  // Given that children node are ignored, handle both return and call in
  // a subsequent conversion.

  // Convert cir.func to builtin.func
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addIllegalOp<mlir::cir::FuncOp, mlir::cir::ReturnOp,
                      mlir::cir::CallOp>();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::TypeConverter converter;
  patterns.add<CIRFuncLowering>(converter, patterns.getContext());
  patterns.add<CIRReturnLowering, CIRCallLowering>(patterns.getContext());

  auto module = getOperation();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<llvm::Module>
lowerFromCIRToLLVMIR(mlir::ModuleOp theModule,
                     std::unique_ptr<mlir::MLIRContext> mlirCtx,
                     LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createConvertCIRToFuncPass());
  pm.addPass(createConvertCIRToMemRefPass());
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

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

std::unique_ptr<mlir::Pass> createConvertCIRToMemRefPass() {
  return std::make_unique<ConvertCIRToMemRefPass>();
}

std::unique_ptr<mlir::Pass> createConvertCIRToFuncPass() {
  return std::make_unique<ConvertCIRToFuncPass>();
}

} // namespace cir
