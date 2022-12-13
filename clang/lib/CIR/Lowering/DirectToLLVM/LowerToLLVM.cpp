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

// class CIRCallLowering : public mlir::OpRewritePattern<mlir::cir::CallOp> {
// public:
//   using OpRewritePattern<mlir::cir::CallOp>::OpRewritePattern;

//   mlir::LogicalResult
//   matchAndRewrite(mlir::cir::CallOp op,
//                   mlir::PatternRewriter &rewriter) const override {
//     rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
//         op, mlir::SymbolRefAttr::get(op), op.getResultTypes(),
//         op.getArgOperands());
//     return mlir::LogicalResult::success();
//   }
// };

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
  matchAndRewrite(mlir::cir::CmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getLhs().getType();
    auto integerType =
        mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless);

    switch (op.getKind()) {
    case mlir::cir::CmpOpKind::gt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::ugt;
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
            op, integerType,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(
            op, integerType,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ugt),
            op.getLhs(), op.getRhs(),
            // TODO(CIR): These fastmath flags need to not be defaulted.
            mlir::LLVM::FastmathFlagsAttr::get(op.getContext(), {}));
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
            op, integerType,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(
            op, integerType,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::uge),
            op.getLhs(), op.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(op.getContext(), {}));
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
            op, integerType,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(
            op, integerType,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ult),
            op.getLhs(), op.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(op.getContext(), {}));
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
            op, integerType,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(
            op, integerType,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ule),
            op.getLhs(), op.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(op.getContext(), {}));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::eq: {
      if (type.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
            op, integerType,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::ICmpPredicate::eq),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(
            op, integerType,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ueq),
            op.getLhs(), op.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(op.getContext(), {}));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ne: {
      if (type.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
            op, integerType,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::ICmpPredicate::ne),
            op.getLhs(), op.getRhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(
            op, integerType,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::une),
            op.getLhs(), op.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(op.getContext(), {}));
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
  patterns.add<CIRBrOpLowering, /* CIRCallLowering, */
               CIRReturnLowering>(patterns.getContext());
  patterns.add<CIRCmpOpLowering, CIRUnaryOpLowering, CIRBinOpLowering,
               CIRLoadLowering, CIRConstantLowering, CIRStoreLowering,
               CIRAllocaLowering, CIRFuncLowering>(converter,
                                                   patterns.getContext());
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
