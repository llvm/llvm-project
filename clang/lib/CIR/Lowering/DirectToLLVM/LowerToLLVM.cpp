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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

//===----------------------------------------------------------------------===//
// Visitors for Lowering CIR Const Attributes
//===----------------------------------------------------------------------===//

/// Switches on the type of attribute and calls the appropriate conversion.
inline mlir::Value
lowerCirAttrAsValue(mlir::Attribute attr, mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter);

/// IntAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::cir::IntAttr intAttr, mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(intAttr.getType()), intAttr.getValue());
}

/// NullAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::cir::NullAttr nullAttr, mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  return rewriter.create<mlir::LLVM::ZeroOp>(
      loc, converter->convertType(nullAttr.getType()));
}

/// FloatAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::FloatAttr fltAttr, mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(fltAttr.getType()), fltAttr.getValue());
}

/// ConstStruct visitor.
mlir::Value lowerCirAttrAsValue(mlir::cir::ConstStructAttr constStruct,
                                mlir::Location loc,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constStruct.getType());
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  // Iteratively lower each constant element of the struct.
  for (auto [idx, elt] : llvm::enumerate(constStruct.getMembers())) {
    mlir::Value init = lowerCirAttrAsValue(elt, loc, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// ArrayAttr visitor.
mlir::Value lowerCirAttrAsValue(mlir::cir::ConstArrayAttr constArr,
                                mlir::Location loc,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constArr.getType());
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);
  auto arrayAttr = constArr.getElts().cast<mlir::ArrayAttr>();
  auto cirArrayType = constArr.getType().cast<mlir::cir::ArrayType>();
  assert(cirArrayType.getEltType().isa<mlir::cir::StructType>() &&
         "Types other than ConstArrayAttr are NYI");

  // Iteratively lower each constant element of the array.
  for (auto [idx, elt] : llvm::enumerate(arrayAttr)) {
    mlir::Value init = lowerCirAttrAsValue(elt, loc, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

/// Switches on the type of attribute and calls the appropriate conversion.
inline mlir::Value
lowerCirAttrAsValue(mlir::Attribute attr, mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  if (const auto intAttr = attr.dyn_cast<mlir::cir::IntAttr>())
    return lowerCirAttrAsValue(intAttr, loc, rewriter, converter);
  if (const auto fltAttr = attr.dyn_cast<mlir::FloatAttr>())
    return lowerCirAttrAsValue(fltAttr, loc, rewriter, converter);
  if (const auto nullAttr = attr.dyn_cast<mlir::cir::NullAttr>())
    return lowerCirAttrAsValue(nullAttr, loc, rewriter, converter);
  if (const auto constStruct = attr.dyn_cast<mlir::cir::ConstStructAttr>())
    return lowerCirAttrAsValue(constStruct, loc, rewriter, converter);
  if (const auto constArr = attr.dyn_cast<mlir::cir::ConstArrayAttr>())
    return lowerCirAttrAsValue(constArr, loc, rewriter, converter);
  if (const auto zeroAttr = attr.dyn_cast<mlir::cir::BoolAttr>())
    llvm_unreachable("bool attribute is NYI");
  if (const auto zeroAttr = attr.dyn_cast<mlir::cir::ZeroAttr>())
    llvm_unreachable("zero attribute is NYI");

  llvm_unreachable("unhandled attribute type");
}

//===----------------------------------------------------------------------===//

mlir::LLVM::Linkage convertLinkage(mlir::cir::GlobalLinkageKind linkage) {
  using CIR = mlir::cir::GlobalLinkageKind;
  using LLVM = mlir::LLVM::Linkage;

  switch (linkage) {
  case CIR::AvailableExternallyLinkage:
    return LLVM::AvailableExternally;
  case CIR::CommonLinkage:
    return LLVM::Common;
  case CIR::ExternalLinkage:
    return LLVM::External;
  case CIR::ExternalWeakLinkage:
    return LLVM::ExternWeak;
  case CIR::InternalLinkage:
    return LLVM::Internal;
  case CIR::LinkOnceAnyLinkage:
    return LLVM::Linkonce;
  case CIR::LinkOnceODRLinkage:
    return LLVM::LinkonceODR;
  case CIR::PrivateLinkage:
    return LLVM::Private;
  case CIR::WeakAnyLinkage:
    return LLVM::Weak;
  case CIR::WeakODRLinkage:
    return LLVM::WeakODR;
  };
}

class CIRPtrStrideOpLowering
    : public mlir::OpConversionPattern<mlir::cir::PtrStrideOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::PtrStrideOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::PtrStrideOp ptrStrideOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *tc = getTypeConverter();
    const auto resultTy = tc->convertType(ptrStrideOp.getType());
    const auto elementTy = tc->convertType(ptrStrideOp.getElementTy());
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(ptrStrideOp, resultTy,
                                                   elementTy, adaptor.getBase(),
                                                   adaptor.getStride());

    return mlir::success();
  }
};

class CIRLoopOpLowering : public mlir::OpConversionPattern<mlir::cir::LoopOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::LoopOp>::OpConversionPattern;
  using LoopKind = mlir::cir::LoopOpKind;

  mlir::LogicalResult
  fetchCondRegionYields(mlir::Region &condRegion,
                        mlir::cir::YieldOp &yieldToBody,
                        mlir::cir::YieldOp &yieldToCont) const {
    for (auto &bb : condRegion) {
      if (auto yieldOp = dyn_cast<mlir::cir::YieldOp>(bb.getTerminator())) {
        if (!yieldOp.getKind().has_value())
          yieldToCont = yieldOp;
        else if (yieldOp.getKind() == mlir::cir::YieldOpKind::Continue)
          yieldToBody = yieldOp;
        else
          return mlir::failure();
      }
    }

    // Succeed only if both yields are found.
    if (!yieldToBody || !yieldToCont)
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOp loopOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto kind = loopOp.getKind();
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Fetch required info from the condition region.
    auto &condRegion = loopOp.getCond();
    auto &condFrontBlock = condRegion.front();
    mlir::cir::YieldOp yieldToBody, yieldToCont;
    if (fetchCondRegionYields(condRegion, yieldToBody, yieldToCont).failed())
      return loopOp.emitError("failed to fetch yields in cond region");

    // Fetch required info from the body region.
    auto &bodyRegion = loopOp.getBody();
    auto &bodyFrontBlock = bodyRegion.front();
    auto bodyYield =
        dyn_cast<mlir::cir::YieldOp>(bodyRegion.back().getTerminator());
    assert(bodyYield && "unstructured while loops are NYI");

    // Fetch required info from the step region.
    auto &stepRegion = loopOp.getStep();
    auto &stepFrontBlock = stepRegion.front();
    auto stepYield =
        dyn_cast<mlir::cir::YieldOp>(stepRegion.back().getTerminator());

    // Move loop op region contents to current CFG.
    rewriter.inlineRegionBefore(condRegion, continueBlock);
    rewriter.inlineRegionBefore(bodyRegion, continueBlock);
    if (kind == LoopKind::For) // Ignore step if not a for-loop.
      rewriter.inlineRegionBefore(stepRegion, continueBlock);

    // Set loop entry point to condition or to body in do-while cases.
    rewriter.setInsertionPointToEnd(currentBlock);
    auto &entry = (kind != LoopKind::DoWhile ? condFrontBlock : bodyFrontBlock);
    rewriter.create<mlir::cir::BrOp>(loopOp.getLoc(), &entry);

    // Set loop exit point to continue block.
    rewriter.setInsertionPoint(yieldToCont);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldToCont, continueBlock);

    // Branch from condition to body.
    rewriter.setInsertionPoint(yieldToBody);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldToBody, &bodyFrontBlock);

    // Branch from body to condition or to step on for-loop cases.
    rewriter.setInsertionPoint(bodyYield);
    auto &bodyExit = (kind == LoopKind::For ? stepFrontBlock : condFrontBlock);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(bodyYield, &bodyExit);

    // Is a for loop: branch from step to condition.
    if (kind == LoopKind::For) {
      rewriter.setInsertionPoint(stepYield);
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(stepYield, &condFrontBlock);
    }

    // Remove the loop op.
    rewriter.eraseOp(loopOp);
    return mlir::success();
  }
};

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

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp castOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    switch (castOp.getKind()) {
    case mlir::cir::CastKind::array_to_ptrdecay: {
      const auto ptrTy = castOp.getType().cast<mlir::cir::PointerType>();
      auto sourceValue = adaptor.getOperands().front();
      auto targetType =
          getTypeConverter()->convertType(castOp->getResult(0).getType());
      auto elementTy = convertTy(ptrTy.getPointee());
      auto offset = llvm::SmallVector<mlir::LLVM::GEPArg>{0};
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
          castOp, targetType, elementTy, sourceValue, offset);
      break;
    }
    case mlir::cir::CastKind::int_to_bool: {
      auto zero = rewriter.create<mlir::cir::ConstantOp>(
          src.getLoc(), castOp.getSrc().getType(),
          mlir::cir::IntAttr::get(castOp.getSrc().getType(), 0));
      rewriter.replaceOpWithNewOp<mlir::cir::CmpOp>(
          castOp, mlir::cir::BoolType::get(getContext()),
          mlir::cir::CmpOpKind::ne, castOp.getSrc(), zero);
      break;
    }
    case mlir::cir::CastKind::integral: {
      auto dstType = castOp.getResult().getType().cast<mlir::cir::IntType>();
      auto srcType = castOp.getSrc().getType().dyn_cast<mlir::cir::IntType>();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy =
          getTypeConverter()->convertType(dstType).cast<mlir::IntegerType>();

      // Target integer is smaller: truncate source value.
      if (dstType.getWidth() < srcType.getWidth()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
      }
      // Target integer is larger: sign extend or zero extend.
      else if (dstType.getWidth() > srcType.getWidth()) {
        if (srcType.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      } else { // Target integer is of the same size: do nothing.
        rewriter.replaceOp(castOp, llvmSrcVal);
      }
      break;
    }
    case mlir::cir::CastKind::floating: {
      auto dstTy = castOp.getResult().getType().cast<mlir::FloatType>();
      auto srcTy = castOp.getSrc().getType();
      auto llvmSrcVal = adaptor.getOperands().front();

      if (auto fpSrcTy = srcTy.dyn_cast<mlir::FloatType>()) {
        if (fpSrcTy.getWidth() > dstTy.getWidth())
          rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(castOp, dstTy,
                                                             llvmSrcVal);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::FPExtOp>(castOp, dstTy,
                                                           llvmSrcVal);
        return mlir::success();
      }

      return castOp.emitError() << "NYI cast from " << srcTy << " to " << dstTy;
    }
    case mlir::cir::CastKind::int_to_ptr: {
      auto dstTy = castOp.getType().cast<mlir::cir::PointerType>();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::ptr_to_int: {
      auto dstTy = castOp.getType().cast<mlir::cir::IntType>();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::float_to_bool: {
      auto dstTy = castOp.getType().cast<mlir::cir::BoolType>();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      auto kind = mlir::LLVM::FCmpPredicate::une;

      // Check if float is not equal to zero.
      auto zeroFloat = rewriter.create<mlir::LLVM::ConstantOp>(
          castOp.getLoc(), llvmSrcVal.getType(),
          mlir::FloatAttr::get(llvmSrcVal.getType(), 0.0));

      // Extend comparison result to either bool (C++) or int (C).
      mlir::Value cmpResult = rewriter.create<mlir::LLVM::FCmpOp>(
          castOp.getLoc(), kind, llvmSrcVal, zeroFloat);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                      cmpResult);
      return mlir::success();
    }
    case mlir::cir::CastKind::bool_to_int: {
      auto dstTy = castOp.getType().cast<mlir::cir::IntType>();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::int_to_float: {
      auto dstTy = castOp.getType();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      if (castOp.getSrc().getType().cast<mlir::cir::IntType>().isSigned())
        rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::float_to_int: {
      auto dstTy = castOp.getType();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      if (castOp.getResult().getType().cast<mlir::cir::IntType>().isSigned())
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPToUIOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::bitcast: {
      auto dstTy = castOp.getType();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::ptr_to_bool: {
      auto null = rewriter.create<mlir::cir::ConstantOp>(
          src.getLoc(), castOp.getSrc().getType(),
          mlir::cir::NullAttr::get(castOp.getSrc().getType()));
      rewriter.replaceOpWithNewOp<mlir::cir::CmpOp>(
          castOp, mlir::cir::BoolType::get(getContext()),
          mlir::cir::CmpOpKind::ne, castOp.getSrc(), null);
      break;
    }
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
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = ifOp.getLoc();
    auto emptyElse = ifOp.getElseRegion().empty();

    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (ifOp->getResults().size() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region
    auto *thenBeforeBody = &ifOp.getThenRegion().front();
    auto *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<mlir::cir::YieldOp>(thenAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
          thenYieldOp, thenYieldOp.getArgs(), continueBlock);
    } else if (!dyn_cast<mlir::cir::ReturnOp>(thenAfterBody->getTerminator())) {
      llvm_unreachable("what are we terminating with?");
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    mlir::Block *elseBeforeBody = nullptr;
    mlir::Block *elseAfterBody = nullptr;
    if (!emptyElse) {
      elseBeforeBody = &ifOp.getElseRegion().front();
      elseAfterBody = &ifOp.getElseRegion().back();
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), thenAfterBody);
    } else {
      elseBeforeBody = elseAfterBody = continueBlock;
    }

    rewriter.setInsertionPointToEnd(currentBlock);
    auto trunc = rewriter.create<mlir::LLVM::TruncOp>(loc, rewriter.getI1Type(),
                                                      adaptor.getCondition());
    rewriter.create<mlir::LLVM::CondBrOp>(loc, trunc.getRes(), thenBeforeBody,
                                          elseBeforeBody);

    if (!emptyElse) {
      rewriter.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOp =
              dyn_cast<mlir::cir::YieldOp>(elseAfterBody->getTerminator())) {
        rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
            elseYieldOp, elseYieldOp.getArgs(), continueBlock);
      } else if (!dyn_cast<mlir::cir::ReturnOp>(
                     elseAfterBody->getTerminator())) {
        llvm_unreachable("what are we terminating with?");
      }
    }

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

    // Empty scope: just remove it.
    if (scopeOp.getRegion().empty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Split the current block before the ScopeOp to create the inlining
    // point.
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
    registry.insert<mlir::BuiltinDialect, mlir::DLTIDialect,
                    mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();
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
    llvm::SmallVector<mlir::Type, 8> llvmResults;
    auto cirResults = op.getResultTypes();

    if (getTypeConverter()->convertTypes(cirResults, llvmResults).failed())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, llvmResults, op.getCalleeAttr(), adaptor.getOperands());
    return mlir::success();
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

mlir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(mlir::cir::ConstArrayAttr attr,
                                     mlir::Type type) {
  auto values = llvm::SmallVector<mlir::APInt, 8>{};
  auto stringAttr = attr.getElts().dyn_cast<mlir::StringAttr>();
  assert(stringAttr && "expected string attribute here");
  for (auto element : stringAttr)
    values.push_back({8, (uint64_t)element});
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({(int64_t)values.size()}, type),
      llvm::ArrayRef(values));
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr
convertToDenseElementsAttr(mlir::cir::ConstArrayAttr attr, mlir::Type type) {
  auto values = llvm::SmallVector<StorageTy, 8>{};
  auto arrayAttr = attr.getElts().dyn_cast<mlir::ArrayAttr>();
  assert(arrayAttr && "expected array here");
  for (auto element : arrayAttr)
    values.push_back(element.cast<AttrTy>().getValue());
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({(int64_t)values.size()}, type),
      llvm::ArrayRef(values));
}

std::optional<mlir::Attribute>
lowerConstArrayAttr(mlir::cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter) {

  // Ensure ConstArrayAttr has a type.
  auto typedConstArr = constArr.dyn_cast<mlir::TypedAttr>();
  assert(typedConstArr && "cir::ConstArrayAttr is not a mlir::TypedAttr");

  // Ensure ConstArrayAttr type is a ArrayType.
  auto cirArrayType = typedConstArr.getType().dyn_cast<mlir::cir::ArrayType>();
  assert(cirArrayType && "cir::ConstArrayAttr is not a cir::ArrayType");

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  auto type = cirArrayType.getEltType();

  // Convert array attr to LLVM compatible dense elements attr.
  if (constArr.getElts().isa<mlir::StringAttr>())
    return convertStringAttrToDenseElementsAttr(constArr,
                                                converter->convertType(type));
  if (type.isa<mlir::cir::IntType>())
    return convertToDenseElementsAttr<mlir::cir::IntAttr, mlir::APInt>(
        constArr, converter->convertType(type));
  if (type.isa<mlir::FloatType>())
    return convertToDenseElementsAttr<mlir::FloatAttr, mlir::APFloat>(
        constArr, converter->convertType(type));

  return std::nullopt;
}

class CIRConstantLowering
    : public mlir::OpConversionPattern<mlir::cir::ConstantOp> {
public:
  using OpConversionPattern<mlir::cir::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Attribute attr = op.getValue();

    if (op.getType().isa<mlir::cir::BoolType>()) {
      int value =
          (op.getValue() ==
           mlir::cir::BoolAttr::get(
               getContext(), ::mlir::cir::BoolType::get(getContext()), true));
      attr = rewriter.getIntegerAttr(typeConverter->convertType(op.getType()),
                                     value);
    } else if (op.getType().isa<mlir::cir::IntType>()) {
      attr = rewriter.getIntegerAttr(
          typeConverter->convertType(op.getType()),
          op.getValue().cast<mlir::cir::IntAttr>().getValue());
    } else if (op.getType().isa<mlir::FloatType>()) {
      attr = op.getValue();
    } else if (op.getType().isa<mlir::cir::PointerType>()) {
      // Optimize with dedicated LLVM op for null pointers.
      if (op.getValue().isa<mlir::cir::NullAttr>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
            op, typeConverter->convertType(op.getType()));
        return mlir::success();
      }
      attr = op.getValue();
    }
    // TODO(cir): constant arrays are currently just pushed into the stack using
    // the store instruction, instead of being stored as global variables and
    // then memcopyied into the stack (as done in Clang).
    else if (auto arrTy = op.getType().dyn_cast<mlir::cir::ArrayType>()) {
      // Fetch operation constant array initializer.
      auto constArr = op.getValue().dyn_cast<mlir::cir::ConstArrayAttr>();
      if (!constArr)
        return op.emitError() << "array does not have a constant initializer";

      // Lower constant array initializer.
      auto denseAttr = lowerConstArrayAttr(constArr, typeConverter);
      if (!denseAttr.has_value()) {
        op.emitError()
            << "unsupported lowering for #cir.const_array with element type "
            << arrTy.getEltType();
        return mlir::failure();
      }
      attr = denseAttr.value();
    } else if (const auto structAttr =
                   op.getValue().dyn_cast<mlir::cir::ConstStructAttr>()) {
      // TODO(cir): this diverges from traditional lowering. Normally the
      // initializer would be a global constant that is memcopied. Here we just
      // define a local constant with llvm.undef that will be stored into the
      // stack.
      auto initVal =
          lowerCirAttrAsValue(structAttr, op.getLoc(), rewriter, typeConverter);
      rewriter.replaceAllUsesWith(op, initVal);
      rewriter.eraseOp(op);
      return mlir::success();
    } else
      return op.emitError() << "unsupported constant type " << op.getType();

    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, getTypeConverter()->convertType(op.getType()), attr);

    return mlir::success();
  }
};

class CIRVAStartLowering
    : public mlir::OpConversionPattern<mlir::cir::VAStartOp> {
public:
  using OpConversionPattern<mlir::cir::VAStartOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VAStartOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(getContext());
    auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), i8PtrTy, adaptor.getOperands().front());
    rewriter.replaceOpWithNewOp<mlir::LLVM::VaStartOp>(op, vaList);
    return mlir::success();
  }
};

class CIRVAEndLowering : public mlir::OpConversionPattern<mlir::cir::VAEndOp> {
public:
  using OpConversionPattern<mlir::cir::VAEndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VAEndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(getContext());
    auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), i8PtrTy, adaptor.getOperands().front());
    rewriter.replaceOpWithNewOp<mlir::LLVM::VaEndOp>(op, vaList);
    return mlir::success();
  }
};

class CIRVACopyLowering
    : public mlir::OpConversionPattern<mlir::cir::VACopyOp> {
public:
  using OpConversionPattern<mlir::cir::VACopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VACopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(getContext());
    auto dstList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), i8PtrTy, adaptor.getOperands().front());
    auto srcList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), i8PtrTy, adaptor.getOperands().back());
    rewriter.replaceOpWithNewOp<mlir::LLVM::VaCopyOp>(op, dstList, srcList);
    return mlir::success();
  }
};

class CIRVAArgLowering : public mlir::OpConversionPattern<mlir::cir::VAArgOp> {
public:
  using OpConversionPattern<mlir::cir::VAArgOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VAArgOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return op.emitError("cir.vaarg lowering is NYI");
  }
};

class CIRFuncLowering : public mlir::OpConversionPattern<mlir::cir::FuncOp> {
public:
  using OpConversionPattern<mlir::cir::FuncOp>::OpConversionPattern;

  /// Returns the name used for the linkage attribute. This *must* correspond
  /// to the name of the attribute in ODS.
  static StringRef getLinkageAttrNameString() { return "linkage"; }

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out
  /// argument attributes.
  void
  filterFuncAttributes(mlir::cir::FuncOp func, bool filterArgAndResAttrs,
                       SmallVectorImpl<mlir::NamedAttribute> &result) const {
    for (auto attr : func->getAttrs()) {
      if (attr.getName() == mlir::SymbolTable::getSymbolAttrName() ||
          attr.getName() == func.getFunctionTypeAttrName() ||
          attr.getName() == getLinkageAttrNameString() ||
          (filterArgAndResAttrs &&
           (attr.getName() == func.getArgAttrsAttrName() ||
            attr.getName() == func.getResAttrsAttrName())))
        continue;

      // `CIRDialectLLVMIRTranslationInterface` requires "cir." prefix for
      // dialect specific attributes, rename them.
      if (attr.getName() == func.getExtraAttrsAttrName()) {
        std::string cirName = "cir." + func.getExtraAttrsAttrName().str();
        attr.setName(mlir::StringAttr::get(getContext(), cirName));
      }
      result.push_back(attr);
    }
  }

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

    // Create the LLVM function operation.
    auto llvmFnTy = mlir::LLVM::LLVMFunctionType::get(
        resultType ? resultType : mlir::LLVM::LLVMVoidType::get(getContext()),
        signatureConversion.getConvertedTypes(),
        /*isVarArg=*/fnType.isVarArg());
    // LLVMFuncOp expects a single FileLine Location instead of a fused
    // location.
    auto Loc = op.getLoc();
    if (Loc.isa<mlir::FusedLoc>()) {
      auto FusedLoc = Loc.cast<mlir::FusedLoc>();
      Loc = FusedLoc.getLocations()[0];
    }
    assert(Loc.isa<mlir::FileLineColLoc>() && "expected single location here");

    auto linkage = convertLinkage(op.getLinkage());
    SmallVector<mlir::NamedAttribute, 4> attributes;
    filterFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

    auto fn = rewriter.create<mlir::LLVM::LLVMFuncOp>(
        Loc, op.getName(), llvmFnTy, linkage, false, mlir::LLVM::CConv::C,
        mlir::SymbolRefAttr(), attributes);

    rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
    if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                           &signatureConversion)))
      return mlir::failure();

    rewriter.eraseOp(op);

    return mlir::LogicalResult::success();
  }
};

class CIRGetGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GetGlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
    // CIRGen should mitigate this and not emit the get_global.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    auto type = getTypeConverter()->convertType(op.getType());
    auto symbol = op.getName();
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, type, symbol);
    return mlir::success();
  }
};

class CIRSwitchOpLowering
    : public mlir::OpConversionPattern<mlir::cir::SwitchOp> {
public:
  using OpConversionPattern<mlir::cir::SwitchOp>::OpConversionPattern;

  inline void rewriteYieldOp(mlir::ConversionPatternRewriter &rewriter,
                             mlir::cir::YieldOp yieldOp,
                             mlir::Block *destination) const {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, yieldOp.getOperands(),
                                                 destination);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SwitchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Empty switch statement: just erase it.
    if (!op.getCases().has_value() || op.getCases()->empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Create exit block.
    rewriter.setInsertionPointAfter(op);
    auto *exitBlock =
        rewriter.splitBlock(rewriter.getBlock(), rewriter.getInsertionPoint());

    // Allocate required data structures (disconsider default case in
    // vectors).
    llvm::SmallVector<mlir::APInt, 8> caseValues;
    llvm::SmallVector<mlir::Block *, 8> caseDestinations;
    llvm::SmallVector<mlir::ValueRange, 8> caseOperands;

    // Initialize default case as optional.
    mlir::Block *defaultDestination = exitBlock;
    mlir::ValueRange defaultOperands = exitBlock->getArguments();

    // Track fallthrough between cases.
    mlir::cir::YieldOp fallthroughYieldOp = nullptr;

    // Digest the case statements values and bodies.
    for (size_t i = 0; i < op.getCases()->size(); ++i) {
      auto &region = op.getRegion(i);
      auto caseAttr = op.getCases()->getValue()[i].cast<mlir::cir::CaseAttr>();

      // Found default case: save destination and operands.
      if (caseAttr.getKind().getValue() == mlir::cir::CaseOpKind::Default) {
        defaultDestination = &region.front();
        defaultOperands = region.getArguments();
      } else {
        // AnyOf cases kind can have multiple values, hence the loop below.
        for (auto &value : caseAttr.getValue()) {
          caseValues.push_back(value.cast<mlir::cir::IntAttr>().getValue());
          caseOperands.push_back(region.getArguments());
          caseDestinations.push_back(&region.front());
        }
      }

      // Previous case is a fallthrough: branch it to this case.
      if (fallthroughYieldOp) {
        rewriteYieldOp(rewriter, fallthroughYieldOp, &region.front());
        fallthroughYieldOp = nullptr;
      }

      // TODO(cir): Handle multi-block case statements.
      if (region.getBlocks().size() != 1)
        return op->emitError("multi-block case statement is NYI");

      // Handle switch-case yields.
      auto *terminator = region.front().getTerminator();
      if (auto yieldOp = dyn_cast<mlir::cir::YieldOp>(terminator)) {
        // TODO(cir): Ensure every yield instead of dealing with optional
        // values.
        assert(yieldOp.getKind().has_value() && "switch yield has no kind");

        switch (yieldOp.getKind().value()) {
        // Fallthrough to next case: track it for the next case to handle.
        case mlir::cir::YieldOpKind::Fallthrough:
          fallthroughYieldOp = yieldOp;
          break;
        // Break out of switch: branch to exit block.
        case mlir::cir::YieldOpKind::Break:
          rewriteYieldOp(rewriter, yieldOp, exitBlock);
          break;
        default:
          return op->emitError("invalid yield kind in case statement");
        }
      }

      // Extract region contents before erasing the switch op.
      rewriter.inlineRegionBefore(region, exitBlock);
    }

    // Last case is a fallthrough: branch it to exit.
    if (fallthroughYieldOp) {
      rewriteYieldOp(rewriter, fallthroughYieldOp, exitBlock);
      fallthroughYieldOp = nullptr;
    }

    // Set switch op to branch to the newly created blocks.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SwitchOp>(
        op, adaptor.getCondition(), defaultDestination, defaultOperands,
        caseValues, caseDestinations, caseOperands);
    return mlir::success();
  }
};

class CIRGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GlobalOp>::OpConversionPattern;

  /// Replace CIR global with a region initialized LLVM global and update
  /// insertion point to the end of the initializer block.
  inline void setupRegionInitializedLLVMGlobalOp(
      mlir::cir::GlobalOp op, mlir::ConversionPatternRewriter &rewriter) const {
    const auto llvmType = getTypeConverter()->convertType(op.getSymType());
    auto newGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, llvmType, op.getConstant(), convertLinkage(op.getLinkage()),
        op.getSymName(), nullptr);
    newGlobalOp.getRegion().push_back(new mlir::Block());
    rewriter.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    // Fetch required values to create LLVM op.
    const auto llvmType = getTypeConverter()->convertType(op.getSymType());
    const auto isConst = op.getConstant();
    const auto linkage = convertLinkage(op.getLinkage());
    const auto symbol = op.getSymName();
    const auto loc = op.getLoc();
    std::optional<mlir::Attribute> init = op.getInitialValue();

    // Check for missing funcionalities.
    if (!init.has_value()) {
      op.emitError() << "uninitialized globals are not yet supported.";
      return mlir::failure();
    }

    // Initializer is a constant array: convert it to a compatible llvm init.
    if (auto constArr = init.value().dyn_cast<mlir::cir::ConstArrayAttr>()) {
      if (auto attr = constArr.getElts().dyn_cast<mlir::StringAttr>()) {
        init = rewriter.getStringAttr(attr.getValue());
      } else if (auto attr = constArr.getElts().dyn_cast<mlir::ArrayAttr>()) {
        auto eltTy =
            constArr.getType().cast<mlir::cir::ArrayType>().getEltType();
        if (eltTy.isa<mlir::cir::StructType>()) {
          setupRegionInitializedLLVMGlobalOp(op, rewriter);
          rewriter.create<mlir::LLVM::ReturnOp>(
              op->getLoc(), lowerCirAttrAsValue(constArr, op->getLoc(),
                                                rewriter, typeConverter));
          return mlir::success();
        }
        if (!(init = lowerConstArrayAttr(constArr, getTypeConverter()))) {
          op.emitError() << "unsupported lowering for #cir.const_array with "
                            "element type "
                         << op.getSymType();
          return mlir::failure();
        }
      } else {
        op.emitError()
            << "unsupported lowering for #cir.const_array with value "
            << constArr.getElts();
        return mlir::failure();
      }
    } else if (llvm::isa<mlir::FloatAttr>(init.value())) {
      // Nothing to do since LLVM already supports these types as
      // initializers.
    }
    // Initializer is a constant integer: convert to MLIR builtin constant.
    else if (auto intAttr = init.value().dyn_cast<mlir::cir::IntAttr>()) {
      init = rewriter.getIntegerAttr(llvmType, intAttr.getValue());
    }
    // Initializer is a global: load global value in initializer block.
    else if (auto attr = init.value().dyn_cast<mlir::FlatSymbolRefAttr>()) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);

      // Fetch global used as initializer.
      auto sourceSymbol =
          dyn_cast<mlir::LLVM::GlobalOp>(mlir::SymbolTable::lookupSymbolIn(
              op->getParentOfType<mlir::ModuleOp>(), attr.getValue()));

      // Load and return the initializer value.
      auto addressOfOp = rewriter.create<mlir::LLVM::AddressOfOp>(
          loc, mlir::LLVM::LLVMPointerType::get(getContext()),
          sourceSymbol.getSymName());
      llvm::SmallVector<mlir::LLVM::GEPArg> offset{0};
      auto gepOp = rewriter.create<mlir::LLVM::GEPOp>(
          loc, llvmType, sourceSymbol.getType(),
          addressOfOp.getResult(), offset);
      rewriter.create<mlir::LLVM::ReturnOp>(loc, gepOp.getResult());
      return mlir::success();
    } else if (isa<mlir::cir::ZeroAttr, mlir::cir::NullAttr>(init.value())) {
      // TODO(cir): once LLVM's dialect has a proper zeroinitializer attribute
      // this should be updated. For now, we tag the LLVM global with a
      // cir.zero attribute that is later replaced with a zeroinitializer.
      // Null pointers also use this path for simplicity, as we would
      // otherwise require a region-based initialization for the global op.
      auto llvmGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, llvmType, isConst, linkage, symbol, nullptr);
      auto cirZeroAttr = mlir::cir::ZeroAttr::get(getContext(), llvmType);
      llvmGlobalOp->setAttr("cir.initial_value", cirZeroAttr);
      return mlir::success();
    } else if (const auto structAttr =
                   init.value().dyn_cast<mlir::cir::ConstStructAttr>()) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          op->getLoc(), lowerCirAttrAsValue(structAttr, op->getLoc(), rewriter,
                                            typeConverter));
      return mlir::success();
    } else if (auto attr = init.value().dyn_cast<mlir::cir::GlobalViewAttr>()) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);

      // Return the address of the global symbol.
      auto elementType = typeConverter->convertType(attr.getType());
      auto addrOfOp = rewriter.create<mlir::LLVM::AddressOfOp>(
          op->getLoc(), elementType, attr.getSymbol());
      rewriter.create<mlir::LLVM::ReturnOp>(op->getLoc(), addrOfOp.getResult());
      return mlir::success();
    } else {
      op.emitError() << "usupported initializer '" << init.value() << "'";
      return mlir::failure();
    }

    // Rewrite op.
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, llvmType, isConst, linkage, symbol, init.value());
    return mlir::success();
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

    auto llvmInType = adaptor.getInput().getType();
    auto llvmType = getTypeConverter()->convertType(op.getType());

    // Integer unary operations.
    if (type.isa<mlir::cir::IntType>()) {
      switch (op.getKind()) {
      case mlir::cir::UnaryOpKind::Inc: {
        auto One = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), llvmInType, mlir::IntegerAttr::get(llvmInType, 1));
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmType,
                                                       adaptor.getInput(), One);
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Dec: {
        auto One = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), llvmInType, mlir::IntegerAttr::get(llvmInType, 1));
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType,
                                                       adaptor.getInput(), One);
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Plus: {
        rewriter.replaceOp(op, adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Minus: {
        auto Zero = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), llvmInType, mlir::IntegerAttr::get(llvmInType, 0));
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType, Zero,
                                                       adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Not: {
        auto MinusOne = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), llvmType, mlir::IntegerAttr::get(llvmType, -1));
        rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, llvmType, MinusOne,
                                                       adaptor.getInput());
        return mlir::success();
      }
      }
    }

    // Floating point unary operations.
    if (type.isa<mlir::FloatType>()) {
      switch (op.getKind()) {
      case mlir::cir::UnaryOpKind::Inc: {
        auto oneAttr = rewriter.getFloatAttr(llvmInType, 1.0);
        auto oneConst = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), llvmInType, oneAttr);
        rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, oneConst,
                                                        adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Dec: {
        auto negOneAttr = rewriter.getFloatAttr(llvmInType, -1.0);
        auto negOneConst = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), llvmInType, negOneAttr);
        rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(
            op, llvmType, negOneConst, adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Plus:
        rewriter.replaceOp(op, adaptor.getInput());
        return mlir::success();
      case mlir::cir::UnaryOpKind::Minus: {
        auto negOneAttr = mlir::FloatAttr::get(llvmInType, -1.0);
        auto negOneConst = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), llvmInType, negOneAttr);
        rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(
            op, llvmType, negOneConst, adaptor.getInput());
        return mlir::success();
      }
      default:
        op.emitError() << "Floating point unary lowering ot implemented";
        return mlir::failure();
      }
    }

    // Boolean unary operations.
    if (type.isa<mlir::cir::BoolType>()) {
      switch (op.getKind()) {
      case mlir::cir::UnaryOpKind::Not:
        rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(
            op, llvmType, adaptor.getInput(),
            rewriter.create<mlir::LLVM::ConstantOp>(
                op.getLoc(), llvmType, mlir::IntegerAttr::get(llvmType, 1)));
        return mlir::success();
      default:
        op.emitError() << "Unary operator not implemented for bool type";
        return mlir::failure();
      }
    }

    return op.emitError() << "Unary operation has unsupported type: " << type;
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
    assert((type.isa<mlir::cir::IntType, mlir::FloatType>()) &&
           "operand type not supported yet");

    auto llvmTy = getTypeConverter()->convertType(op.getType());
    auto rhs = adaptor.getRhs();
    auto lhs = adaptor.getLhs();

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (type.isa<mlir::cir::IntType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Sub:
      if (type.isa<mlir::cir::IntType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Mul:
      if (type.isa<mlir::cir::IntType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::MulOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Div:
      if (auto ty = type.dyn_cast<mlir::cir::IntType>()) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::UDivOp>(op, llvmTy, lhs, rhs);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::SDivOp>(op, llvmTy, lhs, rhs);
      } else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FDivOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Rem:
      if (auto ty = type.dyn_cast<mlir::cir::IntType>()) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::URemOp>(op, llvmTy, lhs, rhs);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::SRemOp>(op, llvmTy, lhs, rhs);
      } else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FRemOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, llvmTy, lhs, rhs);
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRShiftOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ShiftOp> {
public:
  using OpConversionPattern<mlir::cir::ShiftOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ShiftOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cirAmtTy = op.getAmount().getType().dyn_cast<mlir::cir::IntType>();
    auto cirValTy = op.getValue().getType().dyn_cast<mlir::cir::IntType>();
    auto llvmTy = getTypeConverter()->convertType(op.getType());
    auto loc = op.getLoc();
    mlir::Value amt = adaptor.getAmount();
    mlir::Value val = adaptor.getValue();

    assert(cirValTy && cirAmtTy && "non-integer shift is NYI");
    assert(cirValTy == op.getType() && "inconsistent operands' types NYI");

    // Ensure shift amount is the same type as the value. Some undefined
    // behavior might occur in the casts below as per [C99 6.5.7.3].
    if (cirAmtTy.getWidth() > cirValTy.getWidth()) {
      amt = rewriter.create<mlir::LLVM::TruncOp>(loc, llvmTy, amt);
    } else if (cirAmtTy.getWidth() < cirValTy.getWidth()) {
      if (cirAmtTy.isSigned())
        amt = rewriter.create<mlir::LLVM::SExtOp>(loc, llvmTy, amt);
      else
        amt = rewriter.create<mlir::LLVM::ZExtOp>(loc, llvmTy, amt);
    }

    // Lower to the proper LLVM shift operation.
    if (op.getIsShiftleft())
      rewriter.replaceOpWithNewOp<mlir::LLVM::ShlOp>(op, llvmTy, val, amt);
    else {
      if (cirValTy.isUnsigned())
        rewriter.replaceOpWithNewOp<mlir::LLVM::LShrOp>(op, llvmTy, val, amt);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::AShrOp>(op, llvmTy, val, amt);
    }

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
    auto loc = op->getLoc();
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    auto *continueBlock = rewriter.createBlock(
        remainingOpsBlock, op->getResultTypes(),
        SmallVector<mlir::Location>(/* result number always 1 */ 1, loc));
    rewriter.create<mlir::cir::BrOp>(loc, remainingOpsBlock);

    auto &trueRegion = op.getTrueRegion();
    auto *trueBlock = &trueRegion.front();
    mlir::Operation *trueTerminator = trueRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&trueRegion.back());
    auto trueYieldOp = dyn_cast<mlir::cir::YieldOp>(trueTerminator);

    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        trueYieldOp, trueYieldOp.getArgs(), continueBlock);
    rewriter.inlineRegionBefore(trueRegion, continueBlock);

    auto *falseBlock = continueBlock;
    auto &falseRegion = op.getFalseRegion();

    falseBlock = &falseRegion.front();
    mlir::Operation *falseTerminator = falseRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&falseRegion.back());
    auto falseYieldOp = dyn_cast<mlir::cir::YieldOp>(falseTerminator);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        falseYieldOp, falseYieldOp.getArgs(), continueBlock);
    rewriter.inlineRegionBefore(falseRegion, continueBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::LLVM::TruncOp>(
        op.getLoc(), rewriter.getI1Type(), condition);
    rewriter.create<mlir::LLVM::CondBrOp>(loc, i1Condition.getResult(),
                                          trueBlock, falseBlock);

    rewriter.replaceOp(op, continueBlock->getArguments());

    // Ok, we're done!
    return mlir::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<mlir::cir::CmpOp> {
public:
  using OpConversionPattern<mlir::cir::CmpOp>::OpConversionPattern;

  mlir::LLVM::ICmpPredicate convertToICmpPredicate(mlir::cir::CmpOpKind kind,
                                                   bool isSigned) const {
    using CIR = mlir::cir::CmpOpKind;
    using LLVMICmp = mlir::LLVM::ICmpPredicate;

    switch (kind) {
    case CIR::eq:
      return LLVMICmp::eq;
    case CIR::ne:
      return LLVMICmp::ne;
    case CIR::lt:
      return (isSigned ? LLVMICmp::slt : LLVMICmp::ult);
    case CIR::le:
      return (isSigned ? LLVMICmp::sle : LLVMICmp::ule);
    case CIR::gt:
      return (isSigned ? LLVMICmp::sgt : LLVMICmp::ugt);
    case CIR::ge:
      return (isSigned ? LLVMICmp::sge : LLVMICmp::uge);
    }
    llvm_unreachable("Unknown CmpOpKind");
  }

  mlir::LLVM::FCmpPredicate
  convertToFCmpPredicate(mlir::cir::CmpOpKind kind) const {
    using CIR = mlir::cir::CmpOpKind;
    using LLVMFCmp = mlir::LLVM::FCmpPredicate;

    switch (kind) {
    case CIR::eq:
      return LLVMFCmp::ueq;
    case CIR::ne:
      return LLVMFCmp::une;
    case CIR::lt:
      return LLVMFCmp::ult;
    case CIR::le:
      return LLVMFCmp::ule;
    case CIR::gt:
      return LLVMFCmp::ugt;
    case CIR::ge:
      return LLVMFCmp::uge;
    }
    llvm_unreachable("Unknown CmpOpKind");
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp cmpOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = cmpOp.getLhs().getType();
    mlir::Value llResult;

    // Lower to LLVM comparison op.
    if (auto intTy = type.dyn_cast<mlir::cir::IntType>()) {
      auto kind = convertToICmpPredicate(cmpOp.getKind(), intTy.isSigned());
      llResult = rewriter.create<mlir::LLVM::ICmpOp>(
          cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ptrTy = type.dyn_cast<mlir::cir::PointerType>()) {
      auto kind = convertToICmpPredicate(cmpOp.getKind(), /* isSigned=*/false);
      llResult = rewriter.create<mlir::LLVM::ICmpOp>(
          cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (type.isa<mlir::FloatType>()) {
      auto kind = convertToFCmpPredicate(cmpOp.getKind());
      llResult = rewriter.create<mlir::LLVM::FCmpOp>(
          cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else {
      return cmpOp.emitError() << "unsupported type for CmpOp: " << type;
    }

    // LLVM comparison ops return i1, but cir::CmpOp returns the same type as
    // the LHS value. Since this return value can be used later, we need to
    // restore the type with the extension below.
    auto llResultTy = getTypeConverter()->convertType(cmpOp.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, llResultTy,
                                                    llResult);

    return mlir::success();
  }
};

class CIRBrOpLowering : public mlir::OpConversionPattern<mlir::cir::BrOp> {
public:
  using OpConversionPattern<mlir::cir::BrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, adaptor.getOperands(),
                                                  op.getDest());
    return mlir::LogicalResult::success();
  }
};

class CIRStructElementAddrOpLowering
    : public mlir::OpConversionPattern<mlir::cir::StructElementAddr> {
public:
  using mlir::OpConversionPattern<
      mlir::cir::StructElementAddr>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StructElementAddr op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto llResTy = getTypeConverter()->convertType(op.getType());
    // Since the base address is a pointer to structs, the first offset is
    // always zero. The second offset tell us which member it will access.
    llvm::SmallVector<mlir::LLVM::GEPArg> offset{0, op.getIndex()};
    const auto elementTy = getTypeConverter()->convertType(
        op.getStructAddr().getType().getPointee());
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, llResTy, elementTy, adaptor.getStructAddr(), offset);
    return mlir::success();
  }
};

class CIRPtrDiffOpLowering
    : public mlir::OpConversionPattern<mlir::cir::PtrDiffOp> {
public:
  using OpConversionPattern<mlir::cir::PtrDiffOp>::OpConversionPattern;

  uint64_t getTypeSize(mlir::Type type, mlir::Operation &op) const {
    mlir::DataLayout layout(op.getParentOfType<mlir::ModuleOp>());
    return llvm::divideCeil(layout.getTypeSizeInBits(type), 8);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::PtrDiffOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstTy = op.getType().cast<mlir::cir::IntType>();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);

    auto lhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                       adaptor.getLhs());
    auto rhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                       adaptor.getRhs());

    auto diff =
        rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), llvmDstTy, lhs, rhs);

    auto ptrTy = op.getLhs().getType().cast<mlir::cir::PointerType>();
    auto typeSize = getTypeSize(ptrTy.getPointee(), *op);
    auto typeSizeVal = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), llvmDstTy, mlir::IntegerAttr::get(llvmDstTy, typeSize));

    if (dstTy.isUnsigned())
      rewriter.replaceOpWithNewOp<mlir::LLVM::UDivOp>(op, llvmDstTy, diff,
                                                      typeSizeVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::SDivOp>(op, llvmDstTy, diff,
                                                      typeSizeVal);

    return mlir::success();
  }
};

void populateCIRToLLVMConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering>(patterns.getContext());
  patterns.add<CIRCmpOpLowering, CIRLoopOpLowering, CIRBrCondOpLowering,
               CIRPtrStrideOpLowering, CIRCallLowering, CIRUnaryOpLowering,
               CIRBinOpLowering, CIRShiftOpLowering, CIRLoadLowering,
               CIRConstantLowering, CIRStoreLowering, CIRAllocaLowering,
               CIRFuncLowering, CIRScopeOpLowering, CIRCastOpLowering,
               CIRIfLowering, CIRGlobalOpLowering, CIRGetGlobalOpLowering,
               CIRVAStartLowering, CIRVAEndLowering, CIRVACopyLowering,
               CIRVAArgLowering, CIRBrOpLowering, CIRTernaryOpLowering,
               CIRStructElementAddrOpLowering, CIRSwitchOpLowering,
               CIRPtrDiffOpLowering>(
      converter, patterns.getContext());
}

namespace {
void prepareTypeConverter(mlir::LLVMTypeConverter &converter) {
  converter.addConversion([&](mlir::cir::PointerType type) -> mlir::Type {
    return mlir::LLVM::LLVMPointerType::get(&converter.getContext());
  });
  converter.addConversion([&](mlir::cir::ArrayType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](mlir::cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8,
                                  mlir::IntegerType::Signless);
  });
  converter.addConversion([&](mlir::cir::IntType type) -> mlir::Type {
    // LLVM doesn't work with signed types, so we drop the CIR signs here.
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([&](mlir::cir::FuncType type) -> mlir::Type {
    auto result = converter.convertType(type.getReturnType());
    llvm::SmallVector<mlir::Type> arguments;
    if (converter.convertTypes(type.getInputs(), arguments).failed())
      llvm_unreachable("Failed to convert function type parameters");
    auto varArg = type.isVarArg();
    return mlir::LLVM::LLVMFunctionType::get(result, arguments, varArg);
  });
  converter.addConversion([&](mlir::cir::StructType type) -> mlir::Type {
    llvm::SmallVector<mlir::Type> llvmMembers;
    for (auto ty : type.getMembers())
      llvmMembers.push_back(converter.convertType(ty));

    // Struct has a name: lower as an identified struct.
    mlir::LLVM::LLVMStructType llvmStruct;
    if (type.getTypeName().size() != 0) {
      llvmStruct = mlir::LLVM::LLVMStructType::getIdentified(
          type.getContext(), type.getTypeName());
      if (llvmStruct.setBody(llvmMembers, /*isPacked=*/type.getPacked())
              .failed())
        llvm_unreachable("Failed to set body of struct");
    } else { // Struct has no name: lower as literal struct.
      llvmStruct = mlir::LLVM::LLVMStructType::getLiteral(
          type.getContext(), llvmMembers, /*isPacked=*/type.getPacked());
    }

    return llvmStruct;
  });
  converter.addConversion([&](mlir::cir::VoidType type) -> mlir::Type {
    return mlir::LLVM::LLVMVoidType::get(type.getContext());
  });
}
} // namespace

void ConvertCIRToLLVMPass::runOnOperation() {
  auto module = getOperation();

  mlir::LLVMTypeConverter converter(&getContext());
  prepareTypeConverter(converter);

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRToLLVMConversionPatterns(patterns, converter);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(getContext());
  using namespace mlir::cir;
  // clang-format off
  target.addLegalOp<mlir::ModuleOp
                    // ,AllocaOp
                    // ,BrCondOp
                    // ,BrOp
                    // ,CallOp
                    // ,CastOp
                    // ,CmpOp
                    // ,ConstantOp
                    // ,FuncOp
                    // ,LoadOp
                    // ,LoopOp
                    // ,ReturnOp
                    // ,StoreOp
                    // ,YieldOp
                    >();
  // clang-format on
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::BuiltinDialect, mlir::cir::CIRDialect,
                           mlir::func::FuncDialect>();

  getOperation()->removeAttr("cir.sob");
  getOperation()->removeAttr("cir.lang");

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

extern void registerCIRDialectTranslation(mlir::MLIRContext &context);

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule, LLVMContext &llvmCtx) {
  mlir::MLIRContext *mlirCtx = theModule.getContext();
  mlir::PassManager pm(mlirCtx);

  pm.addPass(createConvertCIRToLLVMPass());

  // This is necessary to have line tables emitted and basic
  // debugger working. In the future we will add proper debug information
  // emission directly from our frontend.
  pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

  // FIXME(cir): this shouldn't be necessary. It's meant to be a temporary
  // workaround until we understand why some unrealized casts are being
  // emmited and how to properly avoid them.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  (void)mlir::applyPassManagerCLOptions(pm);

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  registerCIRDialectTranslation(*mlirCtx);

  auto ModuleName = theModule.getName();
  auto llvmModule = mlir::translateModuleToLLVMIR(
      theModule, llvmCtx, ModuleName ? *ModuleName : "CIRToLLVMModule");

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}
} // namespace direct
} // namespace cir
