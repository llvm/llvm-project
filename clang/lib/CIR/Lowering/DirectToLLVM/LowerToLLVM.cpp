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
#include "LoweringHelpers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
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
// Helper Methods
//===----------------------------------------------------------------------===//

namespace {

/// Lowers operations with the terminator trait that have a single successor.
void lowerTerminator(mlir::Operation *op, mlir::Block *dest,
                     mlir::ConversionPatternRewriter &rewriter) {
  assert(op->hasTrait<mlir::OpTrait::IsTerminator>() && "not a terminator");
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(op, dest);
}

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(mlir::Region &region,
                        mlir::function_ref<void(mlir::Operation *)> callback) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<Ops...>(op))
      return mlir::WalkResult::skip();
    callback(op);
    return mlir::WalkResult::advance();
  });
}

/// Convert from a CIR comparison kind to an LLVM IR integral comparison kind.
mlir::LLVM::ICmpPredicate
convertCmpKindToICmpPredicate(mlir::cir::CmpOpKind kind, bool isSigned) {
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

/// Convert from a CIR comparison kind to an LLVM IR floating-point comparison
/// kind.
mlir::LLVM::FCmpPredicate
convertCmpKindToFCmpPredicate(mlir::cir::CmpOpKind kind) {
  using CIR = mlir::cir::CmpOpKind;
  using LLVMFCmp = mlir::LLVM::FCmpPredicate;
  switch (kind) {
  case CIR::eq:
    return LLVMFCmp::oeq;
  case CIR::ne:
    return LLVMFCmp::une;
  case CIR::lt:
    return LLVMFCmp::olt;
  case CIR::le:
    return LLVMFCmp::ole;
  case CIR::gt:
    return LLVMFCmp::ogt;
  case CIR::ge:
    return LLVMFCmp::oge;
  }
  llvm_unreachable("Unknown CmpOpKind");
}

/// If the given type is a vector type, return the vector's element type.
/// Otherwise return the given type unchanged.
mlir::Type elementTypeIfVector(mlir::Type type) {
  if (auto VecType = type.dyn_cast<mlir::cir::VectorType>()) {
    return VecType.getEltType();
  }
  return type;
}

} // namespace

//===----------------------------------------------------------------------===//
// Visitors for Lowering CIR Const Attributes
//===----------------------------------------------------------------------===//

/// Switches on the type of attribute and calls the appropriate conversion.
inline mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::Attribute attr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter);

/// IntAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::cir::IntAttr intAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(intAttr.getType()), intAttr.getValue());
}

/// BoolAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::cir::BoolAttr boolAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(boolAttr.getType()), boolAttr.getValue());
}

/// ConstPtrAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::cir::ConstPtrAttr ptrAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  if (ptrAttr.isNullValue()) {
    return rewriter.create<mlir::LLVM::ZeroOp>(
        loc, converter->convertType(ptrAttr.getType()));
  }
  mlir::Value ptrVal = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), ptrAttr.getValue());
  return rewriter.create<mlir::LLVM::IntToPtrOp>(
      loc, converter->convertType(ptrAttr.getType()), ptrVal);
}

/// FPAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::cir::FPAttr fltAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(fltAttr.getType()), fltAttr.getValue());
}

/// ZeroAttr visitor.
inline mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::cir::ZeroAttr zeroAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::cir::ZeroInitConstOp>(
      loc, converter->convertType(zeroAttr.getType()));
}

/// ConstStruct visitor.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                mlir::cir::ConstStructAttr constStruct,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constStruct.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  // Iteratively lower each constant element of the struct.
  for (auto [idx, elt] : llvm::enumerate(constStruct.getMembers())) {
    mlir::Value init = lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// VTableAttr visitor.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                mlir::cir::VTableAttr vtableArr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(vtableArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(vtableArr.getVtableData())) {
    mlir::Value init = lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// TypeInfoAttr visitor.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                mlir::cir::TypeInfoAttr typeinfoArr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(typeinfoArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(typeinfoArr.getData())) {
    mlir::Value init = lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// ConstArrayAttr visitor
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                mlir::cir::ConstArrayAttr constArr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result;

  if (auto zeros = constArr.getTrailingZerosNum()) {
    auto arrayTy = constArr.getType();
    result = rewriter.create<mlir::cir::ZeroInitConstOp>(
        loc, converter->convertType(arrayTy));
  } else {
    result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);
  }

  // Iteratively lower each constant element of the array.
  if (auto arrayAttr = constArr.getElts().dyn_cast<mlir::ArrayAttr>()) {
    for (auto [idx, elt] : llvm::enumerate(arrayAttr)) {
      mlir::Value init =
          lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
      result =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
    }
  }
  // TODO(cir): this diverges from traditional lowering. Normally the string
  // would be a global constant that is memcopied.
  else if (auto strAttr = constArr.getElts().dyn_cast<mlir::StringAttr>()) {
    auto arrayTy = strAttr.getType().dyn_cast<mlir::cir::ArrayType>();
    assert(arrayTy && "String attribute must have an array type");
    auto eltTy = arrayTy.getEltType();
    for (auto [idx, elt] : llvm::enumerate(strAttr)) {
      auto init = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, converter->convertType(eltTy), elt);
      result =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
    }
  } else {
    llvm_unreachable("unexpected ConstArrayAttr elements");
  }

  return result;
}

// GlobalViewAttr visitor.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                mlir::cir::GlobalViewAttr globalAttr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto module = parentOp->getParentOfType<mlir::ModuleOp>();
  mlir::Type sourceType;
  llvm::StringRef symName;
  auto *sourceSymbol =
      mlir::SymbolTable::lookupSymbolIn(module, globalAttr.getSymbol());
  if (auto llvmSymbol = dyn_cast<mlir::LLVM::GlobalOp>(sourceSymbol)) {
    sourceType = llvmSymbol.getType();
    symName = llvmSymbol.getSymName();
  } else if (auto cirSymbol = dyn_cast<mlir::cir::GlobalOp>(sourceSymbol)) {
    sourceType = converter->convertType(cirSymbol.getSymType());
    symName = cirSymbol.getSymName();
  } else if (auto llvmFun = dyn_cast<mlir::LLVM::LLVMFuncOp>(sourceSymbol)) {
    sourceType = llvmFun.getFunctionType();
    symName = llvmFun.getSymName();
  } else if (auto fun = dyn_cast<mlir::cir::FuncOp>(sourceSymbol)) {
    sourceType = converter->convertType(fun.getFunctionType());
    symName = fun.getSymName();
  } else {
    llvm_unreachable("Unexpected GlobalOp type");
  }

  auto loc = parentOp->getLoc();
  mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
      loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), symName);

  if (globalAttr.getIndices()) {
    llvm::SmallVector<mlir::LLVM::GEPArg> indices;
    for (auto idx : globalAttr.getIndices()) {
      auto intAttr = dyn_cast<mlir::IntegerAttr>(idx);
      assert(intAttr && "index must be integers");
      indices.push_back(intAttr.getValue().getSExtValue());
    }
    auto resTy = addrOp.getType();
    auto eltTy = converter->convertType(sourceType);
    addrOp = rewriter.create<mlir::LLVM::GEPOp>(loc, resTy, eltTy, addrOp,
                                                indices, true);
  }

  auto ptrTy = globalAttr.getType().dyn_cast<mlir::cir::PointerType>();
  assert(ptrTy && "Expecting pointer type in GlobalViewAttr");
  auto llvmEltTy = converter->convertType(ptrTy.getPointee());

  if (llvmEltTy == sourceType)
    return addrOp;

  auto llvmDstTy = converter->convertType(globalAttr.getType());
  return rewriter.create<mlir::LLVM::BitcastOp>(parentOp->getLoc(), llvmDstTy,
                                                addrOp);
}

/// Switches on the type of attribute and calls the appropriate conversion.
inline mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::Attribute attr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  if (const auto intAttr = attr.dyn_cast<mlir::cir::IntAttr>())
    return lowerCirAttrAsValue(parentOp, intAttr, rewriter, converter);
  if (const auto fltAttr = attr.dyn_cast<mlir::cir::FPAttr>())
    return lowerCirAttrAsValue(parentOp, fltAttr, rewriter, converter);
  if (const auto ptrAttr = attr.dyn_cast<mlir::cir::ConstPtrAttr>())
    return lowerCirAttrAsValue(parentOp, ptrAttr, rewriter, converter);
  if (const auto constStruct = attr.dyn_cast<mlir::cir::ConstStructAttr>())
    return lowerCirAttrAsValue(parentOp, constStruct, rewriter, converter);
  if (const auto constArr = attr.dyn_cast<mlir::cir::ConstArrayAttr>())
    return lowerCirAttrAsValue(parentOp, constArr, rewriter, converter);
  if (const auto boolAttr = attr.dyn_cast<mlir::cir::BoolAttr>())
    return lowerCirAttrAsValue(parentOp, boolAttr, rewriter, converter);
  if (const auto zeroAttr = attr.dyn_cast<mlir::cir::ZeroAttr>())
    return lowerCirAttrAsValue(parentOp, zeroAttr, rewriter, converter);
  if (const auto globalAttr = attr.dyn_cast<mlir::cir::GlobalViewAttr>())
    return lowerCirAttrAsValue(parentOp, globalAttr, rewriter, converter);
  if (const auto vtableAttr = attr.dyn_cast<mlir::cir::VTableAttr>())
    return lowerCirAttrAsValue(parentOp, vtableAttr, rewriter, converter);
  if (const auto typeinfoAttr = attr.dyn_cast<mlir::cir::TypeInfoAttr>())
    return lowerCirAttrAsValue(parentOp, typeinfoAttr, rewriter, converter);

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

class CIRCopyOpLowering : public mlir::OpConversionPattern<mlir::cir::CopyOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::CopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::Value length = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), op.getLength());
    rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyOp>(
        op, adaptor.getDst(), adaptor.getSrc(), length, /*isVolatile=*/false);
    return mlir::success();
  }
};

class CIRMemCpyOpLowering
    : public mlir::OpConversionPattern<mlir::cir::MemCpyOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::MemCpyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::MemCpyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyOp>(
        op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
        /*isVolatile=*/false);
    return mlir::success();
  }
};

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

class CIRLoopOpInterfaceLowering
    : public mlir::OpInterfaceConversionPattern<mlir::cir::LoopOpInterface> {
public:
  using mlir::OpInterfaceConversionPattern<
      mlir::cir::LoopOpInterface>::OpInterfaceConversionPattern;

  inline void
  lowerConditionOp(mlir::cir::ConditionOp op, mlir::Block *body,
                   mlir::Block *exit,
                   mlir::ConversionPatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::cir::BrCondOp>(op, op.getCondition(),
                                                     body, exit);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOpInterface op,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Setup CFG blocks.
    auto *entry = rewriter.getInsertionBlock();
    auto *exit = rewriter.splitBlock(entry, rewriter.getInsertionPoint());
    auto *cond = &op.getCond().front();
    auto *body = &op.getBody().front();
    auto *step = (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);

    // Setup loop entry branch.
    rewriter.setInsertionPointToEnd(entry);
    rewriter.create<mlir::LLVM::BrOp>(op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit.
    auto conditionOp = cast<mlir::cir::ConditionOp>(cond->getTerminator());
    lowerConditionOp(conditionOp, body, exit, rewriter);

    // TODO(cir): Remove the walks below. It visits operations unnecessarily,
    // however, to solve this we would likely need a custom DialecConversion
    // driver to customize the order that operations are visited.

    // Lower continue statements.
    mlir::Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](mlir::Operation *op) {
      if (isa<mlir::cir::ContinueOp>(op))
        lowerTerminator(op, dest, rewriter);
    });

    // Lower break statements.
    walkRegionSkipping<mlir::cir::LoopOpInterface, mlir::cir::SwitchOp>(
        op.getBody(), [&](mlir::Operation *op) {
          if (isa<mlir::cir::BreakOp>(op))
            lowerTerminator(op, exit, rewriter);
        });

    // Lower optional body region yield.
    auto bodyYield = dyn_cast<mlir::cir::YieldOp>(body->getTerminator());
    if (bodyYield)
      lowerTerminator(bodyYield, (step ? step : cond), rewriter);

    // Lower mandatory step region yield.
    if (step)
      lowerTerminator(cast<mlir::cir::YieldOp>(step->getTerminator()), cond,
                      rewriter);

    // Move region contents out of the loop op.
    rewriter.inlineRegionBefore(op.getCond(), exit);
    rewriter.inlineRegionBefore(op.getBody(), exit);
    if (step)
      rewriter.inlineRegionBefore(*op.maybeGetStep(), exit);

    rewriter.eraseOp(op);
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
    // For arithmetic conversions, LLVM IR uses the same instruction to convert
    // both individual scalars and entire vectors. This lowering pass handles
    // both situations.

    auto src = adaptor.getSrc();

    switch (castOp.getKind()) {
    case mlir::cir::CastKind::array_to_ptrdecay: {
      const auto ptrTy = castOp.getType().cast<mlir::cir::PointerType>();
      auto sourceValue = adaptor.getOperands().front();
      auto targetType = convertTy(ptrTy);
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
      auto srcType = castOp.getSrc().getType();
      auto dstType = castOp.getResult().getType();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstType = getTypeConverter()->convertType(dstType);
      mlir::cir::IntType srcIntType =
          elementTypeIfVector(srcType).cast<mlir::cir::IntType>();
      mlir::cir::IntType dstIntType =
          elementTypeIfVector(dstType).cast<mlir::cir::IntType>();

      if (dstIntType.getWidth() < srcIntType.getWidth()) {
        // Bigger to smaller. Truncate.
        rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(castOp, llvmDstType,
                                                         llvmSrcVal);
      } else if (dstIntType.getWidth() > srcIntType.getWidth()) {
        // Smaller to bigger. Zero extend or sign extend based on signedness.
        if (srcIntType.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstType,
                                                          llvmSrcVal);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(castOp, llvmDstType,
                                                          llvmSrcVal);
      } else {
        // Same size. Signedness changes doesn't matter to LLVM. Do nothing.
        rewriter.replaceOp(castOp, llvmSrcVal);
      }
      break;
    }
    case mlir::cir::CastKind::floating: {
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy =
          getTypeConverter()->convertType(castOp.getResult().getType());

      auto srcTy = elementTypeIfVector(castOp.getSrc().getType());
      auto dstTy = elementTypeIfVector(castOp.getResult().getType());

      if (!dstTy.isa<mlir::cir::CIRFPTypeInterface>() ||
          !srcTy.isa<mlir::cir::CIRFPTypeInterface>())
        return castOp.emitError()
               << "NYI cast from " << srcTy << " to " << dstTy;

      auto getFloatWidth = [](mlir::Type ty) -> unsigned {
        return ty.cast<mlir::cir::CIRFPTypeInterface>().getWidth();
      };

      if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(castOp, llvmDstTy,
                                                           llvmSrcVal);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPExtOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
      return mlir::success();
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
      auto llvmSrcTy = llvmSrcVal.getType().cast<mlir::IntegerType>();
      auto llvmDstTy =
          getTypeConverter()->convertType(dstTy).cast<mlir::IntegerType>();
      if (llvmSrcTy.getWidth() == llvmDstTy.getWidth())
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                           llvmSrcVal);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::bool_to_float: {
      auto dstTy = castOp.getType();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::int_to_float: {
      auto dstTy = castOp.getType();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      if (elementTypeIfVector(castOp.getSrc().getType())
              .cast<mlir::cir::IntType>()
              .isSigned())
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
      if (elementTypeIfVector(castOp.getResult().getType())
              .cast<mlir::cir::IntType>()
              .isSigned())
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
          mlir::cir::ConstPtrAttr::get(getContext(), castOp.getSrc().getType(),
                                       0));
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

    // FIXME: CIR always lowers !cir.bool to i8 type.
    // In this reason CIR CodeGen often emits the redundant zext + trunc
    // sequence that prevents lowering of llvm.expect in
    // LowerExpectIntrinsicPass.
    // We should fix that in a more appropriate way. But as a temporary solution
    // just avoid the redundant casts here.
    mlir::Value condition;
    auto zext =
        dyn_cast<mlir::LLVM::ZExtOp>(adaptor.getCondition().getDefiningOp());
    if (zext && zext->getOperand(0).getType() == rewriter.getI1Type()) {
      condition = zext->getOperand(0);
      if (zext->use_empty())
        rewriter.eraseOp(zext);
    } else {
      auto trunc = rewriter.create<mlir::LLVM::TruncOp>(
          loc, rewriter.getI1Type(), adaptor.getCondition());
      condition = trunc.getRes();
    }

    rewriter.create<mlir::LLVM::CondBrOp>(loc, condition, thenBeforeBody,
                                          elseBeforeBody);

    if (!emptyElse) {
      rewriter.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOp =
              dyn_cast<mlir::cir::YieldOp>(elseAfterBody->getTerminator())) {
        rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
            elseYieldOp, elseYieldOp.getArgs(), continueBlock);
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
    if (auto yieldOp =
            dyn_cast<mlir::cir::YieldOp>(afterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, yieldOp.getArgs(),
                                                   continueBlock);
    }

    // TODO(cir): stackrestore?

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

  virtual StringRef getArgument() const override {
    return "cir-to-llvm-internal";
  }
};

class CIRCallLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
public:
  using OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 8> llvmResults;
    auto cirResults = op.getResultTypes();
    auto *converter = getTypeConverter();

    if (converter->convertTypes(cirResults, llvmResults).failed())
      return mlir::failure();

    if (auto callee = op.getCalleeAttr()) { // direct call
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, llvmResults, op.getCalleeAttr(), adaptor.getOperands());
    } else { // indirect call
      assert(op.getOperands().size() &&
             "operands list must no be empty for the indirect call");
      auto typ = op.getOperands().front().getType();
      assert(isa<mlir::cir::PointerType>(typ) && "expected pointer type");
      auto ptyp = dyn_cast<mlir::cir::PointerType>(typ);
      auto ftyp = dyn_cast<mlir::cir::FuncType>(ptyp.getPointee());
      assert(ftyp && "expected a pointer to a function as the first operand");

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op,
          dyn_cast<mlir::LLVM::LLVMFunctionType>(converter->convertType(ftyp)),
          adaptor.getOperands());
    }
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
    mlir::Value size =
        op.isDynamic()
            ? adaptor.getDynAllocSize()
            : rewriter.create<mlir::LLVM::ConstantOp>(
                  op.getLoc(),
                  typeConverter->convertType(rewriter.getIndexType()),
                  rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    auto elementTy = getTypeConverter()->convertType(op.getAllocaType());
    auto resultTy = mlir::LLVM::LLVMPointerType::get(getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
        op, resultTy, elementTy, size, op.getAlignmentAttr().getInt());
    return mlir::success();
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
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
        op, llvmTy, adaptor.getAddr(), /* alignment */ 0,
        /* volatile */ op.getIsVolatile());
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
public:
  using OpConversionPattern<mlir::cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
        op, adaptor.getValue(), adaptor.getAddr(),
        /* alignment */ 0, /* volatile */ op.getIsVolatile());
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

template <typename StorageTy> StorageTy getZeroInitFromType(mlir::Type Ty);

template <> mlir::APInt getZeroInitFromType(mlir::Type Ty) {
  assert(Ty.isa<mlir::cir::IntType>() && "expected int type");
  auto IntTy = Ty.cast<mlir::cir::IntType>();
  return mlir::APInt::getZero(IntTy.getWidth());
}

template <> mlir::APFloat getZeroInitFromType(mlir::Type Ty) {
  assert((Ty.isa<mlir::cir::SingleType, mlir::cir::DoubleType>()) &&
         "only float and double supported");
  if (Ty.isF32() || Ty.isa<mlir::cir::SingleType>())
    return mlir::APFloat(0.f);
  if (Ty.isF64() || Ty.isa<mlir::cir::DoubleType>())
    return mlir::APFloat(0.0);
  llvm_unreachable("NYI");
}

// return the nested type and quiantity of elements for cir.array type.
// e.g: for !cir.array<!cir.array<!s32i x 3> x 1>
// it returns !s32i as return value and stores 3 to elemQuantity.
mlir::Type getNestedTypeAndElemQuantity(mlir::Type Ty, unsigned &elemQuantity) {
  assert(Ty.isa<mlir::cir::ArrayType>() && "expected ArrayType");

  elemQuantity = 1;
  mlir::Type nestTy = Ty;
  while (auto ArrTy = nestTy.dyn_cast<mlir::cir::ArrayType>()) {
    nestTy = ArrTy.getEltType();
    elemQuantity *= ArrTy.getSize();
  }

  return nestTy;
}

template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(mlir::cir::ConstArrayAttr attr,
                                    llvm::SmallVectorImpl<StorageTy> &values) {
  auto arrayAttr = attr.getElts().cast<mlir::ArrayAttr>();
  for (auto eltAttr : arrayAttr) {
    if (auto valueAttr = eltAttr.dyn_cast<AttrTy>()) {
      values.push_back(valueAttr.getValue());
    } else if (auto subArrayAttr =
                   eltAttr.dyn_cast<mlir::cir::ConstArrayAttr>()) {
      convertToDenseElementsAttrImpl<AttrTy>(subArrayAttr, values);
    } else if (auto zeroAttr = eltAttr.dyn_cast<mlir::cir::ZeroAttr>()) {
      unsigned numStoredZeros = 0;
      auto nestTy =
          getNestedTypeAndElemQuantity(zeroAttr.getType(), numStoredZeros);
      values.insert(values.end(), numStoredZeros,
                    getZeroInitFromType<StorageTy>(nestTy));
    } else {
      llvm_unreachable("unknown element in ConstArrayAttr");
    }
  }
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr
convertToDenseElementsAttr(mlir::cir::ConstArrayAttr attr,
                           const llvm::SmallVectorImpl<int64_t> &dims,
                           mlir::Type type) {
  auto values = llvm::SmallVector<StorageTy, 8>{};
  convertToDenseElementsAttrImpl<AttrTy>(attr, values);
  return mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(dims, type),
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
  mlir::Type type = cirArrayType;
  auto dims = llvm::SmallVector<int64_t, 2>{};
  while (auto arrayType = type.dyn_cast<mlir::cir::ArrayType>()) {
    dims.push_back(arrayType.getSize());
    type = arrayType.getEltType();
  }

  // Convert array attr to LLVM compatible dense elements attr.
  if (constArr.getElts().isa<mlir::StringAttr>())
    return convertStringAttrToDenseElementsAttr(constArr,
                                                converter->convertType(type));
  if (type.isa<mlir::cir::IntType>())
    return convertToDenseElementsAttr<mlir::cir::IntAttr, mlir::APInt>(
        constArr, dims, converter->convertType(type));
  if (type.isa<mlir::cir::CIRFPTypeInterface>())
    return convertToDenseElementsAttr<mlir::cir::FPAttr, mlir::APFloat>(
        constArr, dims, converter->convertType(type));

  return std::nullopt;
}

bool hasTrailingZeros(mlir::cir::ConstArrayAttr attr) {
  auto array = attr.getElts().dyn_cast<mlir::ArrayAttr>();
  return attr.hasTrailingZeros() ||
         (array && std::count_if(array.begin(), array.end(), [](auto elt) {
            auto ar = dyn_cast<mlir::cir::ConstArrayAttr>(elt);
            return ar && hasTrailingZeros(ar);
          }));
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
    } else if (op.getType().isa<mlir::cir::CIRFPTypeInterface>()) {
      attr = rewriter.getFloatAttr(
          typeConverter->convertType(op.getType()),
          op.getValue().cast<mlir::cir::FPAttr>().getValue());
    } else if (op.getType().isa<mlir::cir::PointerType>()) {
      // Optimize with dedicated LLVM op for null pointers.
      if (op.getValue().isa<mlir::cir::ConstPtrAttr>()) {
        if (op.getValue().cast<mlir::cir::ConstPtrAttr>().isNullValue()) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
              op, typeConverter->convertType(op.getType()));
          return mlir::success();
        }
      }
      // Lower GlobalViewAttr to llvm.mlir.addressof
      if (auto gv = op.getValue().dyn_cast<mlir::cir::GlobalViewAttr>()) {
        auto newOp = lowerCirAttrAsValue(op, gv, rewriter, getTypeConverter());
        rewriter.replaceOp(op, newOp);
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
      if (!constArr && !isa<mlir::cir::ZeroAttr>(op.getValue()))
        return op.emitError() << "array does not have a constant initializer";

      std::optional<mlir::Attribute> denseAttr;
      if (constArr && hasTrailingZeros(constArr)) {
        auto newOp =
            lowerCirAttrAsValue(op, constArr, rewriter, getTypeConverter());
        rewriter.replaceOp(op, newOp);
        return mlir::success();
      } else if (constArr &&
                 (denseAttr = lowerConstArrayAttr(constArr, typeConverter))) {
        attr = denseAttr.value();
      } else {
        auto initVal =
            lowerCirAttrAsValue(op, op.getValue(), rewriter, typeConverter);
        rewriter.replaceAllUsesWith(op, initVal);
        rewriter.eraseOp(op);
        return mlir::success();
      }
    } else if (const auto structAttr =
                   op.getValue().dyn_cast<mlir::cir::ConstStructAttr>()) {
      // TODO(cir): this diverges from traditional lowering. Normally the
      // initializer would be a global constant that is memcopied. Here we just
      // define a local constant with llvm.undef that will be stored into the
      // stack.
      auto initVal =
          lowerCirAttrAsValue(op, structAttr, rewriter, typeConverter);
      rewriter.replaceAllUsesWith(op, initVal);
      rewriter.eraseOp(op);
      return mlir::success();
    } else if (auto strTy = op.getType().dyn_cast<mlir::cir::StructType>()) {
      if (auto zero = op.getValue().dyn_cast<mlir::cir::ZeroAttr>()) {
        auto initVal = lowerCirAttrAsValue(op, zero, rewriter, typeConverter);
        rewriter.replaceAllUsesWith(op, initVal);
        rewriter.eraseOp(op);
        return mlir::success();
      }

      return op.emitError() << "unsupported lowering for struct constant type "
                            << op.getType();
    } else
      return op.emitError() << "unsupported constant type " << op.getType();

    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, getTypeConverter()->convertType(op.getType()), attr);

    return mlir::success();
  }
};

class CIRVectorCreateLowering
    : public mlir::OpConversionPattern<mlir::cir::VecCreateOp> {
public:
  using OpConversionPattern<mlir::cir::VecCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Start with an 'undef' value for the vector.  Then 'insertelement' for
    // each of the vector elements.
    auto vecTy = op.getType().dyn_cast<mlir::cir::VectorType>();
    assert(vecTy && "result type of cir.vec.create op is not VectorType");
    auto llvmTy = typeConverter->convertType(vecTy);
    auto loc = op.getLoc();
    mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);
    assert(vecTy.getSize() == op.getElements().size() &&
           "cir.vec.create op count doesn't match vector type elements count");
    for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
      mlir::Value indexValue = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), i);
      result = rewriter.create<mlir::LLVM::InsertElementOp>(
          loc, result, adaptor.getElements()[i], indexValue);
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class CIRVectorInsertLowering
    : public mlir::OpConversionPattern<mlir::cir::VecInsertOp> {
public:
  using OpConversionPattern<mlir::cir::VecInsertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecInsertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertElementOp>(
        op, adaptor.getVec(), adaptor.getValue(), adaptor.getIndex());
    return mlir::success();
  }
};

class CIRVectorExtractLowering
    : public mlir::OpConversionPattern<mlir::cir::VecExtractOp> {
public:
  using OpConversionPattern<mlir::cir::VecExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecExtractOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractElementOp>(
        op, adaptor.getVec(), adaptor.getIndex());
    return mlir::success();
  }
};

class CIRVectorCmpOpLowering
    : public mlir::OpConversionPattern<mlir::cir::VecCmpOp> {
public:
  using OpConversionPattern<mlir::cir::VecCmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(op.getType().isa<mlir::cir::VectorType>() &&
           op.getLhs().getType().isa<mlir::cir::VectorType>() &&
           op.getRhs().getType().isa<mlir::cir::VectorType>() &&
           "Vector compare with non-vector type");
    // LLVM IR vector comparison returns a vector of i1.  This one-bit vector
    // must be sign-extended to the correct result type.
    auto elementType = elementTypeIfVector(op.getLhs().getType());
    mlir::Value bitResult;
    if (auto intType = elementType.dyn_cast<mlir::cir::IntType>()) {
      bitResult = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(),
          convertCmpKindToICmpPredicate(op.getKind(), intType.isSigned()),
          adaptor.getLhs(), adaptor.getRhs());
    } else if (elementType.isa<mlir::cir::CIRFPTypeInterface>()) {
      bitResult = rewriter.create<mlir::LLVM::FCmpOp>(
          op.getLoc(), convertCmpKindToFCmpPredicate(op.getKind()),
          adaptor.getLhs(), adaptor.getRhs());
    } else {
      return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(
        op, typeConverter->convertType(op.getType()), bitResult);
    return mlir::success();
  }
};

class CIRVectorSplatLowering
    : public mlir::OpConversionPattern<mlir::cir::VecSplatOp> {
public:
  using OpConversionPattern<mlir::cir::VecSplatOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecSplatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Vector splat can be implemented with an `insertelement` and a
    // `shufflevector`, which is better than an `insertelement` for each
    // element in the vector. Start with an undef vector. Insert the value into
    // the first element. Then use a `shufflevector` with a mask of all 0 to
    // fill out the entire vector with that value.
    auto vecTy = op.getType().dyn_cast<mlir::cir::VectorType>();
    assert(vecTy && "result type of cir.vec.splat op is not VectorType");
    auto llvmTy = typeConverter->convertType(vecTy);
    auto loc = op.getLoc();
    mlir::Value undef = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);
    mlir::Value indexValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    mlir::Value elementValue = adaptor.getValue();
    mlir::Value oneElement = rewriter.create<mlir::LLVM::InsertElementOp>(
        loc, undef, elementValue, indexValue);
    SmallVector<int32_t> zeroValues(vecTy.getSize(), 0);
    mlir::Value shuffled = rewriter.create<mlir::LLVM::ShuffleVectorOp>(
        loc, oneElement, undef, zeroValues);
    rewriter.replaceOp(op, shuffled);
    return mlir::success();
  }
};

class CIRVectorTernaryLowering
    : public mlir::OpConversionPattern<mlir::cir::VecTernaryOp> {
public:
  using OpConversionPattern<mlir::cir::VecTernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecTernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(op.getType().isa<mlir::cir::VectorType>() &&
           op.getCond().getType().isa<mlir::cir::VectorType>() &&
           op.getVec1().getType().isa<mlir::cir::VectorType>() &&
           op.getVec2().getType().isa<mlir::cir::VectorType>() &&
           "Vector ternary op with non-vector type");
    // Convert `cond` into a vector of i1, then use that in a `select` op.
    mlir::Value bitVec = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::ne, adaptor.getCond(),
        rewriter.create<mlir::LLVM::ZeroOp>(
            op.getCond().getLoc(),
            typeConverter->convertType(op.getCond().getType())));
    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
        op, bitVec, adaptor.getVec1(), adaptor.getVec2());
    return mlir::success();
  }
};

class CIRVectorShuffleIntsLowering
    : public mlir::OpConversionPattern<mlir::cir::VecShuffleOp> {
public:
  using OpConversionPattern<mlir::cir::VecShuffleOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecShuffleOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // LLVM::ShuffleVectorOp takes an ArrayRef of int for the list of indices.
    // Convert the ClangIR ArrayAttr of IntAttr constants into a
    // SmallVector<int>.
    SmallVector<int, 8> indices;
    std::transform(
        op.getIndices().begin(), op.getIndices().end(),
        std::back_inserter(indices), [](mlir::Attribute intAttr) {
          return intAttr.cast<mlir::cir::IntAttr>().getValue().getSExtValue();
        });
    rewriter.replaceOpWithNewOp<mlir::LLVM::ShuffleVectorOp>(
        op, adaptor.getVec1(), adaptor.getVec2(), indices);
    return mlir::success();
  }
};

class CIRVectorShuffleVecLowering
    : public mlir::OpConversionPattern<mlir::cir::VecShuffleDynamicOp> {
public:
  using OpConversionPattern<
      mlir::cir::VecShuffleDynamicOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecShuffleDynamicOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // LLVM IR does not have an operation that corresponds to this form of
    // the built-in.
    //     __builtin_shufflevector(V, I)
    // is implemented as this pseudocode, where the for loop is unrolled
    // and N is the number of elements:
    //     masked = I & (N-1)
    //     for (i in 0 <= i < N)
    //       result[i] = V[masked[i]]
    auto loc = op.getLoc();
    mlir::Value input = adaptor.getVec();
    mlir::Type llvmIndexVecType =
        getTypeConverter()->convertType(op.getIndices().getType());
    mlir::Type llvmIndexType = getTypeConverter()->convertType(
        elementTypeIfVector(op.getIndices().getType()));
    uint64_t numElements =
        op.getVec().getType().cast<mlir::cir::VectorType>().getSize();
    mlir::Value maskValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType,
        mlir::IntegerAttr::get(llvmIndexType, numElements - 1));
    mlir::Value maskVector =
        rewriter.create<mlir::LLVM::UndefOp>(loc, llvmIndexVecType);
    for (uint64_t i = 0; i < numElements; ++i) {
      mlir::Value iValue = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), i);
      maskVector = rewriter.create<mlir::LLVM::InsertElementOp>(
          loc, maskVector, maskValue, iValue);
    }
    mlir::Value maskedIndices = rewriter.create<mlir::LLVM::AndOp>(
        loc, llvmIndexVecType, adaptor.getIndices(), maskVector);
    mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(
        loc, getTypeConverter()->convertType(op.getVec().getType()));
    for (uint64_t i = 0; i < numElements; ++i) {
      mlir::Value iValue = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), i);
      mlir::Value indexValue = rewriter.create<mlir::LLVM::ExtractElementOp>(
          loc, maskedIndices, iValue);
      mlir::Value valueAtIndex =
          rewriter.create<mlir::LLVM::ExtractElementOp>(loc, input, indexValue);
      result = rewriter.create<mlir::LLVM::InsertElementOp>(
          loc, result, valueAtIndex, iValue);
    }
    rewriter.replaceOp(op, result);
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
    auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
    auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), opaquePtr, adaptor.getOperands().front());
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
    auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
    auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), opaquePtr, adaptor.getOperands().front());
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
    auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
    auto dstList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), opaquePtr, adaptor.getOperands().front());
    auto srcList = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), opaquePtr, adaptor.getOperands().back());
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
    assert((Loc.isa<mlir::FileLineColLoc>() || Loc.isa<mlir::UnknownLoc>()) &&
           "expected single location or unknown location here");

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

      for (auto &blk : region.getBlocks()) {
        if (blk.getNumSuccessors())
          continue;

        // Handle switch-case yields.
        if (auto yieldOp = dyn_cast<mlir::cir::YieldOp>(blk.getTerminator()))
          fallthroughYieldOp = yieldOp;
      }

      // Handle break statements.
      walkRegionSkipping<mlir::cir::LoopOpInterface, mlir::cir::SwitchOp>(
          region, [&](mlir::Operation *op) {
            if (isa<mlir::cir::BreakOp>(op))
              lowerTerminator(op, exitBlock, rewriter);
          });

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
    std::optional<mlir::StringRef> section = op.getSection();
    std::optional<mlir::Attribute> init = op.getInitialValue();

    SmallVector<mlir::NamedAttribute> attributes;
    if (section.has_value())
      attributes.push_back(rewriter.getNamedAttr(
          "section", rewriter.getStringAttr(section.value())));

    // Check for missing funcionalities.
    if (!init.has_value()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, llvmType, isConst, linkage, symbol, mlir::Attribute(),
          /*alignment*/ 0, /*addrSpace*/ 0,
          /*dsoLocal*/ false, /*threadLocal*/ false,
          /*comdat*/ mlir::SymbolRefAttr(), attributes);
      return mlir::success();
    }

    // Initializer is a constant array: convert it to a compatible llvm init.
    if (auto constArr = init.value().dyn_cast<mlir::cir::ConstArrayAttr>()) {
      if (auto attr = constArr.getElts().dyn_cast<mlir::StringAttr>()) {
        init = rewriter.getStringAttr(attr.getValue());
      } else if (auto attr = constArr.getElts().dyn_cast<mlir::ArrayAttr>()) {
        // Failed to use a compact attribute as an initializer:
        // initialize elements individually.
        if (!(init = lowerConstArrayAttr(constArr, getTypeConverter()))) {
          setupRegionInitializedLLVMGlobalOp(op, rewriter);
          rewriter.create<mlir::LLVM::ReturnOp>(
              op->getLoc(),
              lowerCirAttrAsValue(op, constArr, rewriter, typeConverter));
          return mlir::success();
        }
      } else {
        op.emitError()
            << "unsupported lowering for #cir.const_array with value "
            << constArr.getElts();
        return mlir::failure();
      }
    } else if (auto fltAttr = init.value().dyn_cast<mlir::cir::FPAttr>()) {
      // Initializer is a constant floating-point number: convert to MLIR
      // builtin constant.
      init = rewriter.getFloatAttr(llvmType, fltAttr.getValue());
    }
    // Initializer is a constant integer: convert to MLIR builtin constant.
    else if (auto intAttr = init.value().dyn_cast<mlir::cir::IntAttr>()) {
      init = rewriter.getIntegerAttr(llvmType, intAttr.getValue());
    } else if (auto boolAttr = init.value().dyn_cast<mlir::cir::BoolAttr>()) {
      init = rewriter.getBoolAttr(boolAttr.getValue());
    } else if (isa<mlir::cir::ZeroAttr, mlir::cir::ConstPtrAttr>(
                   init.value())) {
      // TODO(cir): once LLVM's dialect has a proper zeroinitializer attribute
      // this should be updated. For now, we use a custom op to initialize
      // globals to zero.
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      auto value =
          lowerCirAttrAsValue(op, init.value(), rewriter, typeConverter);
      rewriter.create<mlir::LLVM::ReturnOp>(loc, value);
      return mlir::success();
    } else if (const auto structAttr =
                   init.value().dyn_cast<mlir::cir::ConstStructAttr>()) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          op->getLoc(),
          lowerCirAttrAsValue(op, structAttr, rewriter, typeConverter));
      return mlir::success();
    } else if (auto attr = init.value().dyn_cast<mlir::cir::GlobalViewAttr>()) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          loc, lowerCirAttrAsValue(op, attr, rewriter, typeConverter));
      return mlir::success();
    } else if (const auto vtableAttr =
                   init.value().dyn_cast<mlir::cir::VTableAttr>()) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          op->getLoc(),
          lowerCirAttrAsValue(op, vtableAttr, rewriter, typeConverter));
      return mlir::success();
    } else if (const auto typeinfoAttr =
                   init.value().dyn_cast<mlir::cir::TypeInfoAttr>()) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          op->getLoc(),
          lowerCirAttrAsValue(op, typeinfoAttr, rewriter, typeConverter));
      return mlir::success();
    } else {
      op.emitError() << "usupported initializer '" << init.value() << "'";
      return mlir::failure();
    }

    // Rewrite op.
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, llvmType, isConst, linkage, symbol, init.value(),
        /*alignment*/ 0, /*addrSpace*/ 0,
        /*dsoLocal*/ false, /*threadLocal*/ false,
        /*comdat*/ mlir::SymbolRefAttr(), attributes);
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
    assert(op.getType() == op.getInput().getType() &&
           "Unary operation's operand type and result type are different");
    mlir::Type type = op.getType();
    mlir::Type elementType = elementTypeIfVector(type);
    bool IsVector = type.isa<mlir::cir::VectorType>();
    auto llvmType = getTypeConverter()->convertType(type);
    auto loc = op.getLoc();

    // Integer unary operations: + - ~ ++ --
    if (elementType.isa<mlir::cir::IntType>()) {
      switch (op.getKind()) {
      case mlir::cir::UnaryOpKind::Inc: {
        assert(!IsVector && "++ not allowed on vector types");
        auto One = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, llvmType, mlir::IntegerAttr::get(llvmType, 1));
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmType,
                                                       adaptor.getInput(), One);
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Dec: {
        assert(!IsVector && "-- not allowed on vector types");
        auto One = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, llvmType, mlir::IntegerAttr::get(llvmType, 1));
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType,
                                                       adaptor.getInput(), One);
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Plus: {
        rewriter.replaceOp(op, adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Minus: {
        mlir::Value Zero;
        if (IsVector)
          Zero = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmType);
        else
          Zero = rewriter.create<mlir::LLVM::ConstantOp>(
              loc, llvmType, mlir::IntegerAttr::get(llvmType, 0));
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType, Zero,
                                                       adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Not: {
        // bit-wise compliment operator, implemented as an XOR with -1.
        mlir::Value MinusOne;
        if (IsVector) {
          // Creating a vector object with all -1 values is easier said than
          // done. It requires a series of insertelement ops.
          mlir::Type llvmElementType =
              getTypeConverter()->convertType(elementType);
          auto MinusOneInt = rewriter.create<mlir::LLVM::ConstantOp>(
              loc, llvmElementType,
              mlir::IntegerAttr::get(llvmElementType, -1));
          MinusOne = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmType);
          auto NumElements = type.dyn_cast<mlir::cir::VectorType>().getSize();
          for (uint64_t i = 0; i < NumElements; ++i) {
            mlir::Value indexValue = rewriter.create<mlir::LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), i);
            MinusOne = rewriter.create<mlir::LLVM::InsertElementOp>(
                loc, MinusOne, MinusOneInt, indexValue);
          }
        } else {
          MinusOne = rewriter.create<mlir::LLVM::ConstantOp>(
              loc, llvmType, mlir::IntegerAttr::get(llvmType, -1));
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, llvmType, MinusOne,
                                                       adaptor.getInput());
        return mlir::success();
      }
      }
    }

    // Floating point unary operations: + - ++ --
    if (elementType.isa<mlir::cir::CIRFPTypeInterface>()) {
      switch (op.getKind()) {
      case mlir::cir::UnaryOpKind::Inc: {
        assert(!IsVector && "++ not allowed on vector types");
        auto oneAttr = rewriter.getFloatAttr(llvmType, 1.0);
        auto oneConst =
            rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, oneAttr);
        rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, oneConst,
                                                        adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Dec: {
        assert(!IsVector && "-- not allowed on vector types");
        auto negOneAttr = rewriter.getFloatAttr(llvmType, -1.0);
        auto negOneConst =
            rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, negOneAttr);
        rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(
            op, llvmType, negOneConst, adaptor.getInput());
        return mlir::success();
      }
      case mlir::cir::UnaryOpKind::Plus:
        rewriter.replaceOp(op, adaptor.getInput());
        return mlir::success();
      case mlir::cir::UnaryOpKind::Minus: {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FNegOp>(op, llvmType,
                                                        adaptor.getInput());
        return mlir::success();
      }
      default:
        return op.emitError()
               << "Unknown floating-point unary operation during CIR lowering";
      }
    }

    // Boolean unary operations: ! only. (For all others, the operand has
    // already been promoted to int.)
    if (elementType.isa<mlir::cir::BoolType>()) {
      switch (op.getKind()) {
      case mlir::cir::UnaryOpKind::Not:
        assert(!IsVector && "NYI: op! on vector mask");
        rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(
            op, llvmType, adaptor.getInput(),
            rewriter.create<mlir::LLVM::ConstantOp>(
                loc, llvmType, mlir::IntegerAttr::get(llvmType, 1)));
        return mlir::success();
      default:
        return op.emitError()
               << "Unknown boolean unary operation during CIR lowering";
      }
    }

    // Pointer unary operations: + only.  (++ and -- of pointers are implemented
    // with cir.ptr_stride, not cir.unary.)
    if (elementType.isa<mlir::cir::PointerType>()) {
      switch (op.getKind()) {
      case mlir::cir::UnaryOpKind::Plus:
        rewriter.replaceOp(op, adaptor.getInput());
        return mlir::success();
      default:
        op.emitError() << "Unknown pointer unary operation during CIR lowering";
        return mlir::failure();
      }
    }

    return op.emitError() << "Unary operation has unsupported type: "
                          << elementType;
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
    assert((type.isa<mlir::cir::IntType, mlir::cir::CIRFPTypeInterface,
                     mlir::cir::VectorType>()) &&
           "operand type not supported yet");

    auto llvmTy = getTypeConverter()->convertType(op.getType());
    auto rhs = adaptor.getRhs();
    auto lhs = adaptor.getLhs();

    type = elementTypeIfVector(type);

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

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp cmpOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = cmpOp.getLhs().getType();
    mlir::Value llResult;

    // Lower to LLVM comparison op.
    if (auto intTy = type.dyn_cast<mlir::cir::IntType>()) {
      auto kind =
          convertCmpKindToICmpPredicate(cmpOp.getKind(), intTy.isSigned());
      llResult = rewriter.create<mlir::LLVM::ICmpOp>(
          cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ptrTy = type.dyn_cast<mlir::cir::PointerType>()) {
      auto kind = convertCmpKindToICmpPredicate(cmpOp.getKind(),
                                                /* isSigned=*/false);
      llResult = rewriter.create<mlir::LLVM::ICmpOp>(
          cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (type.isa<mlir::cir::CIRFPTypeInterface>()) {
      auto kind = convertCmpKindToFCmpPredicate(cmpOp.getKind());
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

static mlir::Value createLLVMBitOp(mlir::Location loc,
                                   const llvm::Twine &llvmIntrinBaseName,
                                   mlir::Type resultTy, mlir::Value operand,
                                   std::optional<bool> poisonZeroInputFlag,
                                   mlir::ConversionPatternRewriter &rewriter) {
  auto operandIntTy = operand.getType().cast<mlir::IntegerType>();
  auto resultIntTy = resultTy.cast<mlir::IntegerType>();

  std::string llvmIntrinName =
      llvmIntrinBaseName.concat(".i")
          .concat(std::to_string(operandIntTy.getWidth()))
          .str();
  auto llvmIntrinNameAttr =
      mlir::StringAttr::get(rewriter.getContext(), llvmIntrinName);

  // Note that LLVM intrinsic calls to bit intrinsics have the same type as the
  // operand.
  mlir::LLVM::CallIntrinsicOp op;
  if (poisonZeroInputFlag.has_value()) {
    auto poisonZeroInputValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), static_cast<int64_t>(*poisonZeroInputFlag));
    op = rewriter.create<mlir::LLVM::CallIntrinsicOp>(
        loc, operand.getType(), llvmIntrinNameAttr,
        mlir::ValueRange{operand, poisonZeroInputValue});
  } else {
    op = rewriter.create<mlir::LLVM::CallIntrinsicOp>(
        loc, operand.getType(), llvmIntrinNameAttr, operand);
  }

  mlir::Value result = op->getResult(0);
  if (operandIntTy.getWidth() > resultIntTy.getWidth()) {
    result = rewriter.create<mlir::LLVM::TruncOp>(loc, resultTy, result);
  } else if (operandIntTy.getWidth() < resultIntTy.getWidth()) {
    result = rewriter.create<mlir::LLVM::ZExtOp>(loc, resultTy, result);
  }

  return result;
}

class CIRBitClrsbOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitClrsbOp> {
public:
  using OpConversionPattern<mlir::cir::BitClrsbOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitClrsbOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), adaptor.getInput().getType(), 0);
    auto isNeg = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(),
        mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                           mlir::LLVM::ICmpPredicate::slt),
        adaptor.getInput(), zero);

    auto negOne = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), adaptor.getInput().getType(), -1);
    auto flipped = rewriter.create<mlir::LLVM::XOrOp>(
        op.getLoc(), adaptor.getInput(), negOne);

    auto select = rewriter.create<mlir::LLVM::SelectOp>(
        op.getLoc(), isNeg, flipped, adaptor.getInput());

    auto resTy = getTypeConverter()->convertType(op.getType());
    auto clz = createLLVMBitOp(op.getLoc(), "llvm.ctlz", resTy, select,
                               /*poisonZeroInputFlag=*/false, rewriter);

    auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
    auto res = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), clz, one);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitClzOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitClzOp> {
public:
  using OpConversionPattern<mlir::cir::BitClzOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitClzOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto llvmOp =
        createLLVMBitOp(op.getLoc(), "llvm.ctlz", resTy, adaptor.getInput(),
                        /*poisonZeroInputFlag=*/true, rewriter);
    rewriter.replaceOp(op, llvmOp);
    return mlir::LogicalResult::success();
  }
};

class CIRBitCtzOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitCtzOp> {
public:
  using OpConversionPattern<mlir::cir::BitCtzOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitCtzOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto llvmOp =
        createLLVMBitOp(op.getLoc(), "llvm.cttz", resTy, adaptor.getInput(),
                        /*poisonZeroInputFlag=*/true, rewriter);
    rewriter.replaceOp(op, llvmOp);
    return mlir::LogicalResult::success();
  }
};

class CIRBitFfsOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitFfsOp> {
public:
  using OpConversionPattern<mlir::cir::BitFfsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitFfsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto ctz =
        createLLVMBitOp(op.getLoc(), "llvm.cttz", resTy, adaptor.getInput(),
                        /*poisonZeroInputFlag=*/false, rewriter);

    auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
    auto ctzAddOne = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), ctz, one);

    auto zeroInputTy = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), adaptor.getInput().getType(), 0);
    auto isZero = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(),
        mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                           mlir::LLVM::ICmpPredicate::eq),
        adaptor.getInput(), zeroInputTy);

    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 0);
    auto res = rewriter.create<mlir::LLVM::SelectOp>(op.getLoc(), isZero, zero,
                                                     ctzAddOne);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitParityOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitParityOp> {
public:
  using OpConversionPattern<mlir::cir::BitParityOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitParityOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto popcnt =
        createLLVMBitOp(op.getLoc(), "llvm.ctpop", resTy, adaptor.getInput(),
                        /*poisonZeroInputFlag=*/std::nullopt, rewriter);

    auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
    auto popcntMod2 =
        rewriter.create<mlir::LLVM::AndOp>(op.getLoc(), popcnt, one);
    rewriter.replaceOp(op, popcntMod2);

    return mlir::LogicalResult::success();
  }
};

class CIRBitPopcountOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitPopcountOp> {
public:
  using OpConversionPattern<mlir::cir::BitPopcountOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitPopcountOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto llvmOp =
        createLLVMBitOp(op.getLoc(), "llvm.ctpop", resTy, adaptor.getInput(),
                        /*poisonZeroInputFlag=*/std::nullopt, rewriter);
    rewriter.replaceOp(op, llvmOp);
    return mlir::LogicalResult::success();
  }
};

class CIRAtomicFetchLowering
    : public mlir::OpConversionPattern<mlir::cir::AtomicFetch> {
public:
  using OpConversionPattern<mlir::cir::AtomicFetch>::OpConversionPattern;

  mlir::LLVM::AtomicOrdering
  getLLVMAtomicOrder(mlir::cir::MemOrder memo) const {
    switch (memo) {
    case mlir::cir::MemOrder::Relaxed:
      return mlir::LLVM::AtomicOrdering::monotonic;
    case mlir::cir::MemOrder::Consume:
    case mlir::cir::MemOrder::Acquire:
      return mlir::LLVM::AtomicOrdering::acquire;
    case mlir::cir::MemOrder::Release:
      return mlir::LLVM::AtomicOrdering::release;
    case mlir::cir::MemOrder::AcquireRelease:
      return mlir::LLVM::AtomicOrdering::acq_rel;
    case mlir::cir::MemOrder::SequentiallyConsistent:
      return mlir::LLVM::AtomicOrdering::seq_cst;
    }
    llvm_unreachable("shouldn't get here");
  }

  mlir::Value buildPostOp(mlir::cir::AtomicFetch op, OpAdaptor adaptor,
                          mlir::ConversionPatternRewriter &rewriter,
                          mlir::Value rmwVal, bool isInt) const {
    SmallVector<mlir::Value> atomicOperands = {rmwVal, adaptor.getVal()};
    SmallVector<mlir::Type> atomicResTys = {rmwVal.getType()};
    return rewriter
        .create(op.getLoc(),
                rewriter.getStringAttr(getLLVMBinop(op.getBinop(), isInt)),
                atomicOperands, atomicResTys, {})
        ->getResult(0);
  }

  llvm::StringLiteral getLLVMBinop(mlir::cir::AtomicFetchKind k,
                                   bool isInt) const {
    switch (k) {
    case mlir::cir::AtomicFetchKind::Add:
      return isInt ? mlir::LLVM::AddOp::getOperationName()
                   : mlir::LLVM::FAddOp::getOperationName();
    case mlir::cir::AtomicFetchKind::Sub:
      return isInt ? mlir::LLVM::SubOp::getOperationName()
                   : mlir::LLVM::FSubOp::getOperationName();
    case mlir::cir::AtomicFetchKind::And:
      return mlir::LLVM::AndOp::getOperationName();
    case mlir::cir::AtomicFetchKind::Xor:
      return mlir::LLVM::XOrOp::getOperationName();
    case mlir::cir::AtomicFetchKind::Or:
      return mlir::LLVM::OrOp::getOperationName();
    case mlir::cir::AtomicFetchKind::Nand:
      // There's no nand binop in LLVM, this is later fixed with a not.
      return mlir::LLVM::AndOp::getOperationName();
    }
    llvm_unreachable("Unknown atomic fetch opcode");
  }

  mlir::LLVM::AtomicBinOp getLLVMAtomicBinOp(mlir::cir::AtomicFetchKind k,
                                             bool isInt) const {
    switch (k) {
    case mlir::cir::AtomicFetchKind::Add:
      return isInt ? mlir::LLVM::AtomicBinOp::add
                   : mlir::LLVM::AtomicBinOp::fadd;
    case mlir::cir::AtomicFetchKind::Sub:
      return isInt ? mlir::LLVM::AtomicBinOp::sub
                   : mlir::LLVM::AtomicBinOp::fsub;
    case mlir::cir::AtomicFetchKind::And:
      return mlir::LLVM::AtomicBinOp::_and;
    case mlir::cir::AtomicFetchKind::Xor:
      return mlir::LLVM::AtomicBinOp::_xor;
    case mlir::cir::AtomicFetchKind::Or:
      return mlir::LLVM::AtomicBinOp::_or;
    case mlir::cir::AtomicFetchKind::Nand:
      return mlir::LLVM::AtomicBinOp::nand;
    }
    llvm_unreachable("Unknown atomic fetch opcode");
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AtomicFetch op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    bool isInt; // otherwise it's float.
    if (op.getVal().getType().isa<mlir::cir::IntType>())
      isInt = true;
    else if (op.getVal()
                 .getType()
                 .isa<mlir::cir::SingleType, mlir::cir::DoubleType>())
      isInt = false;
    else {
      return op.emitError()
             << "Unsupported type: " << adaptor.getVal().getType();
    }

    // FIXME: add syncscope.
    auto llvmOrder = getLLVMAtomicOrder(adaptor.getMemOrder());
    auto llvmBinOpc = getLLVMAtomicBinOp(op.getBinop(), isInt);
    auto rmwVal = rewriter.create<mlir::LLVM::AtomicRMWOp>(
        op.getLoc(), llvmBinOpc, adaptor.getPtr(), adaptor.getVal(), llvmOrder);

    mlir::Value result = rmwVal.getRes();
    if (!op.getFetchFirst()) {
      result = buildPostOp(op, adaptor, rewriter, rmwVal.getRes(), isInt);

      // Compensate lack of nand binop in LLVM IR.
      if (op.getBinop() == mlir::cir::AtomicFetchKind::Nand) {
        auto negOne = rewriter.create<mlir::LLVM::ConstantOp>(
            op.getLoc(), result.getType(), -1);
        result =
            rewriter.create<mlir::LLVM::XOrOp>(op.getLoc(), result, negOne);
      }
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class CIRByteswapOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ByteswapOp> {
public:
  using OpConversionPattern<mlir::cir::ByteswapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ByteswapOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Note that LLVM intrinsic calls to @llvm.bswap.i* have the same type as
    // the operand.

    auto resTy =
        getTypeConverter()->convertType(op.getType()).cast<mlir::IntegerType>();

    std::string llvmIntrinName = "llvm.bswap.i";
    llvmIntrinName.append(std::to_string(resTy.getWidth()));
    auto llvmIntrinNameAttr =
        mlir::StringAttr::get(rewriter.getContext(), llvmIntrinName);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, resTy, llvmIntrinNameAttr, adaptor.getInput());

    return mlir::LogicalResult::success();
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

class CIRGetMemberOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GetMemberOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::GetMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetMemberOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto llResTy = getTypeConverter()->convertType(op.getType());
    const auto structTy =
        op.getAddrTy().getPointee().cast<mlir::cir::StructType>();
    assert(structTy && "expected struct type");

    switch (structTy.getKind()) {
    case mlir::cir::StructType::Struct:
    case mlir::cir::StructType::Class: {
      // Since the base address is a pointer to an aggregate, the first offset
      // is always zero. The second offset tell us which member it will access.
      llvm::SmallVector<mlir::LLVM::GEPArg, 2> offset{0, op.getIndex()};
      const auto elementTy = getTypeConverter()->convertType(structTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, llResTy, elementTy,
                                                     adaptor.getAddr(), offset);
      return mlir::success();
    }
    case mlir::cir::StructType::Union:
      // Union members share the address space, so we just need a bitcast to
      // conform to type-checking.
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, llResTy,
                                                         adaptor.getAddr());
      return mlir::success();
    }
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

class CIRFAbsOpLowering : public mlir::OpConversionPattern<mlir::cir::FAbsOp> {
public:
  using OpConversionPattern<mlir::cir::FAbsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FAbsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::FAbsOp>(
        op, adaptor.getOperands().front());
    return mlir::success();
  }
};

class CIRExpectOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ExpectOp> {
public:
  using OpConversionPattern<mlir::cir::ExpectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ExpectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    std::optional<llvm::APFloat> prob = op.getProb();
    if (!prob)
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExpectOp>(op, adaptor.getVal(),
                                                        adaptor.getExpected());
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExpectWithProbabilityOp>(
          op, adaptor.getVal(), adaptor.getExpected(), prob.value());
    return mlir::success();
  }
};

class CIRVTableAddrPointOpLowering
    : public mlir::OpConversionPattern<mlir::cir::VTableAddrPointOp> {
public:
  using OpConversionPattern<mlir::cir::VTableAddrPointOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VTableAddrPointOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto *converter = getTypeConverter();
    auto targetType = converter->convertType(op.getType());
    mlir::Value symAddr = op.getSymAddr();

    mlir::Type eltType;
    if (!symAddr) {
      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto symbol = dyn_cast<mlir::LLVM::GlobalOp>(
          mlir::SymbolTable::lookupSymbolIn(module, op.getNameAttr()));
      symAddr = rewriter.create<mlir::LLVM::AddressOfOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()),
          *op.getName());
      eltType = converter->convertType(symbol.getType());
    }

    auto offsets = llvm::SmallVector<mlir::LLVM::GEPArg>{
        0, op.getVtableIndex(), op.getAddressPointIndex()};
    if (eltType)
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, targetType, eltType,
                                                     symAddr, offsets, true);
    else
      llvm_unreachable("Shouldn't ever be missing an eltType here");

    return mlir::success();
  }
};

class CIRStackSaveLowering
    : public mlir::OpConversionPattern<mlir::cir::StackSaveOp> {
public:
  using OpConversionPattern<mlir::cir::StackSaveOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StackSaveOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptrTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::StackSaveOp>(op, ptrTy);
    return mlir::success();
  }
};

class CIRStackRestoreLowering
    : public mlir::OpConversionPattern<mlir::cir::StackRestoreOp> {
public:
  using OpConversionPattern<mlir::cir::StackRestoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StackRestoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::StackRestoreOp>(op,
                                                            adaptor.getPtr());
    return mlir::success();
  }
};

class CIRUnreachableLowering
    : public mlir::OpConversionPattern<mlir::cir::UnreachableOp> {
public:
  using OpConversionPattern<mlir::cir::UnreachableOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnreachableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
    return mlir::success();
  }
};

class CIRTrapLowering : public mlir::OpConversionPattern<mlir::cir::TrapOp> {
public:
  using OpConversionPattern<mlir::cir::TrapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TrapOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    rewriter.eraseOp(op);

    auto llvmTrapIntrinsicType =
        mlir::LLVM::LLVMVoidType::get(op->getContext());
    rewriter.create<mlir::LLVM::CallIntrinsicOp>(
        loc, llvmTrapIntrinsicType, "llvm.trap", mlir::ValueRange{});

    // Note that the call to llvm.trap is not a terminator in LLVM dialect.
    // So we must emit an additional llvm.unreachable to terminate the current
    // block.
    rewriter.create<mlir::LLVM::UnreachableOp>(loc);

    return mlir::success();
  }
};

class CIRInlineAsmOpLowering
    : public mlir::OpConversionPattern<mlir::cir::InlineAsmOp> {

  using mlir::OpConversionPattern<mlir::cir::InlineAsmOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::InlineAsmOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Type llResTy;
    if (op.getNumResults())
      llResTy = getTypeConverter()->convertType(op.getType(0));

    auto dialect = op.getAsmFlavor();
    auto llDialect = dialect == mlir::cir::AsmFlavor::x86_att
                         ? mlir::LLVM::AsmDialect::AD_ATT
                         : mlir::LLVM::AsmDialect::AD_Intel;

    std::vector<mlir::Attribute> opAttrs;
    auto llvmAttrName = mlir::LLVM::InlineAsmOp::getElementTypeAttrName();

    if (auto operandAttrs = op.getOperandAttrs()) {
      for (auto attr : *operandAttrs) {
        if (isa<mlir::cir::OptNoneAttr>(attr)) {
          opAttrs.push_back(mlir::Attribute());
          continue;
        }

        mlir::TypeAttr tAttr = cast<mlir::TypeAttr>(attr);
        std::vector<mlir::NamedAttribute> attrs;
        auto typAttr = mlir::TypeAttr::get(
            getTypeConverter()->convertType(tAttr.getValue()));

        attrs.push_back(rewriter.getNamedAttr(llvmAttrName, typAttr));
        auto newDict = rewriter.getDictionaryAttr(attrs);
        opAttrs.push_back(newDict);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::InlineAsmOp>(
        op, llResTy, adaptor.getOperands(), op.getAsmStringAttr(),
        op.getConstraintsAttr(), op.getSideEffectsAttr(),
        /*is_align_stack*/ mlir::UnitAttr(),
        mlir::LLVM::AsmDialectAttr::get(getContext(), llDialect),
        rewriter.getArrayAttr(opAttrs));
    return mlir::success();
  }
};

class CIRPrefetchLowering
    : public mlir::OpConversionPattern<mlir::cir::PrefetchOp> {
public:
  using OpConversionPattern<mlir::cir::PrefetchOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::PrefetchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::Prefetch>(
        op, adaptor.getAddr(), adaptor.getIsWrite(), adaptor.getLocality(),
        /*DataCache*/ 1);
    return mlir::success();
  }
};

class CIRSetBitfieldLowering
    : public mlir::OpConversionPattern<mlir::cir::SetBitfieldOp> {
public:
  using OpConversionPattern<mlir::cir::SetBitfieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SetBitfieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto info = op.getBitfieldInfo();
    auto size = info.getSize();
    auto offset = info.getOffset();
    auto storageType = info.getStorageType();
    auto context = storageType.getContext();

    unsigned storageSize = 0;

    if (auto arTy = storageType.dyn_cast<mlir::cir::ArrayType>())
      storageSize = arTy.getSize() * 8;
    else if (auto intTy = storageType.dyn_cast<mlir::cir::IntType>())
      storageSize = intTy.getWidth();
    else
      llvm_unreachable(
          "Either ArrayType or IntType expected for bitfields storage");

    auto intType = mlir::IntegerType::get(context, storageSize);
    auto srcVal = createIntCast(rewriter, adaptor.getSrc(), intType);
    auto srcWidth = storageSize;
    auto resultVal = srcVal;

    if (storageSize != size) {
      assert(storageSize > size && "Invalid bitfield size.");

      mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
          op.getLoc(), intType, adaptor.getAddr(), /* alignment */ 0,
          op.getIsVolatile());

      srcVal = createAnd(rewriter, srcVal,
                         llvm::APInt::getLowBitsSet(srcWidth, size));
      resultVal = srcVal;
      srcVal = createShL(rewriter, srcVal, offset);

      // Mask out the original value.
      val =
          createAnd(rewriter, val,
                    ~llvm::APInt::getBitsSet(srcWidth, offset, offset + size));

      // Or together the unchanged values and the source value.
      srcVal = rewriter.create<mlir::LLVM::OrOp>(op.getLoc(), val, srcVal);
    }

    rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), srcVal, adaptor.getAddr(),
                                         /* alignment */ 0, op.getIsVolatile());

    auto resultTy = getTypeConverter()->convertType(op.getType());

    resultVal =
        createIntCast(rewriter, resultVal, resultTy.cast<mlir::IntegerType>());

    if (info.getIsSigned()) {
      assert(size <= storageSize);
      unsigned highBits = storageSize - size;

      if (highBits) {
        resultVal = createShL(rewriter, resultVal, highBits);
        resultVal = createAShR(rewriter, resultVal, highBits);
      }
    }

    rewriter.replaceOp(op, resultVal);
    return mlir::success();
  }
};

class CIRGetBitfieldLowering
    : public mlir::OpConversionPattern<mlir::cir::GetBitfieldOp> {
public:
  using OpConversionPattern<mlir::cir::GetBitfieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetBitfieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto info = op.getBitfieldInfo();
    auto size = info.getSize();
    auto offset = info.getOffset();
    auto storageType = info.getStorageType();
    auto context = storageType.getContext();
    unsigned storageSize = 0;

    if (auto arTy = storageType.dyn_cast<mlir::cir::ArrayType>())
      storageSize = arTy.getSize() * 8;
    else if (auto intTy = storageType.dyn_cast<mlir::cir::IntType>())
      storageSize = intTy.getWidth();
    else
      llvm_unreachable(
          "Either ArrayType or IntType expected for bitfields storage");

    auto intType = mlir::IntegerType::get(context, storageSize);

    mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), intType, adaptor.getAddr(), 0, op.getIsVolatile());
    val = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), intType, val);

    if (info.getIsSigned()) {
      assert(static_cast<unsigned>(offset + size) <= storageSize);
      unsigned highBits = storageSize - offset - size;
      val = createShL(rewriter, val, highBits);
      val = createAShR(rewriter, val, offset + highBits);
    } else {
      val = createLShR(rewriter, val, offset);

      if (static_cast<unsigned>(offset) + size < storageSize)
        val = createAnd(rewriter, val,
                        llvm::APInt::getLowBitsSet(storageSize, size));
    }

    auto resTy = getTypeConverter()->convertType(op.getType());
    auto newOp = createIntCast(rewriter, val, resTy.cast<mlir::IntegerType>(),
                               info.getIsSigned());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

class CIRIsConstantOpLowering
    : public mlir::OpConversionPattern<mlir::cir::IsConstantOp> {

  using mlir::OpConversionPattern<mlir::cir::IsConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::IsConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME(cir): llvm.intr.is.constant returns i1 value but the LLVM Lowering
    // expects that cir.bool type will be lowered as i8 type.
    // So we have to insert zext here.
    auto isConstantOP = rewriter.create<mlir::LLVM::IsConstantOp>(
        op.getLoc(), adaptor.getVal());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(op, rewriter.getI8Type(),
                                                    isConstantOP);
    return mlir::success();
  }
};

void populateCIRToLLVMConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering>(patterns.getContext());
  patterns.add<
      CIRCmpOpLowering, CIRBitClrsbOpLowering, CIRBitClzOpLowering,
      CIRBitCtzOpLowering, CIRBitFfsOpLowering, CIRBitParityOpLowering,
      CIRBitPopcountOpLowering, CIRAtomicFetchLowering, CIRByteswapOpLowering,
      CIRLoopOpInterfaceLowering, CIRBrCondOpLowering, CIRPtrStrideOpLowering,
      CIRCallLowering, CIRUnaryOpLowering, CIRBinOpLowering, CIRShiftOpLowering,
      CIRLoadLowering, CIRConstantLowering, CIRStoreLowering, CIRAllocaLowering,
      CIRFuncLowering, CIRScopeOpLowering, CIRCastOpLowering, CIRIfLowering,
      CIRGlobalOpLowering, CIRGetGlobalOpLowering, CIRVAStartLowering,
      CIRVAEndLowering, CIRVACopyLowering, CIRVAArgLowering, CIRBrOpLowering,
      CIRTernaryOpLowering, CIRGetMemberOpLowering, CIRSwitchOpLowering,
      CIRPtrDiffOpLowering, CIRCopyOpLowering, CIRMemCpyOpLowering,
      CIRFAbsOpLowering, CIRExpectOpLowering, CIRVTableAddrPointOpLowering,
      CIRVectorCreateLowering, CIRVectorInsertLowering,
      CIRVectorExtractLowering, CIRVectorCmpOpLowering, CIRVectorSplatLowering,
      CIRVectorTernaryLowering, CIRVectorShuffleIntsLowering,
      CIRVectorShuffleVecLowering, CIRStackSaveLowering,
      CIRStackRestoreLowering, CIRUnreachableLowering, CIRTrapLowering,
      CIRInlineAsmOpLowering, CIRSetBitfieldLowering, CIRGetBitfieldLowering,
      CIRPrefetchLowering, CIRIsConstantOpLowering>(converter,
                                                    patterns.getContext());
}

namespace {
void prepareTypeConverter(mlir::LLVMTypeConverter &converter,
                          mlir::DataLayout &dataLayout) {
  converter.addConversion([&](mlir::cir::PointerType type) -> mlir::Type {
    // Drop pointee type since LLVM dialect only allows opaque pointers.
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](mlir::cir::ArrayType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](mlir::cir::VectorType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::LLVM::getFixedVectorType(ty, type.getSize());
  });
  converter.addConversion([&](mlir::cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8,
                                  mlir::IntegerType::Signless);
  });
  converter.addConversion([&](mlir::cir::IntType type) -> mlir::Type {
    // LLVM doesn't work with signed types, so we drop the CIR signs here.
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([&](mlir::cir::SingleType type) -> mlir::Type {
    return mlir::FloatType::getF32(type.getContext());
  });
  converter.addConversion([&](mlir::cir::DoubleType type) -> mlir::Type {
    return mlir::FloatType::getF64(type.getContext());
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
    // FIXME(cir): create separate unions, struct, and classes types.
    // Convert struct members.
    llvm::SmallVector<mlir::Type> llvmMembers;
    switch (type.getKind()) {
    case mlir::cir::StructType::Class:
      // TODO(cir): This should be properly validated.
    case mlir::cir::StructType::Struct:
      for (auto ty : type.getMembers())
        llvmMembers.push_back(converter.convertType(ty));
      break;
    // Unions are lowered as only the largest member.
    case mlir::cir::StructType::Union: {
      auto largestMember = type.getLargestMember(dataLayout);
      if (largestMember)
        llvmMembers.push_back(converter.convertType(largestMember));
      break;
    }
    }

    // Struct has a name: lower as an identified struct.
    mlir::LLVM::LLVMStructType llvmStruct;
    if (type.getName()) {
      llvmStruct = mlir::LLVM::LLVMStructType::getIdentified(
          type.getContext(), type.getPrefixedName());
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

static void buildCtorList(mlir::ModuleOp module) {
  llvm::SmallVector<std::pair<StringRef, int>, 2> globalCtors;
  for (auto namedAttr : module->getAttrs()) {
    if (namedAttr.getName() == "cir.globalCtors") {
      for (auto attr : namedAttr.getValue().cast<mlir::ArrayAttr>()) {
        assert(attr.isa<mlir::cir::GlobalCtorAttr>() &&
               "must be a GlobalCtorAttr");
        if (auto ctorAttr = attr.cast<mlir::cir::GlobalCtorAttr>()) {
          // default priority is 65536
          int priority = 65536;
          if (ctorAttr.getPriority())
            priority = *ctorAttr.getPriority();
          globalCtors.emplace_back(ctorAttr.getName(), priority);
        }
      }
      break;
    }
  }

  if (globalCtors.empty())
    return;

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Create a global array llvm.global_ctors with element type of
  // struct { i32, ptr, ptr }
  auto CtorPFTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  llvm::SmallVector<mlir::Type> CtorStructFields;
  CtorStructFields.push_back(builder.getI32Type());
  CtorStructFields.push_back(CtorPFTy);
  CtorStructFields.push_back(CtorPFTy);

  auto CtorStructTy = mlir::LLVM::LLVMStructType::getLiteral(
      builder.getContext(), CtorStructFields);
  auto CtorStructArrayTy =
      mlir::LLVM::LLVMArrayType::get(CtorStructTy, globalCtors.size());

  auto loc = module.getLoc();
  auto newGlobalOp = builder.create<mlir::LLVM::GlobalOp>(
      loc, CtorStructArrayTy, true, mlir::LLVM::Linkage::Appending,
      "llvm.global_ctors", mlir::Attribute());

  newGlobalOp.getRegion().push_back(new mlir::Block());
  builder.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());

  mlir::Value result =
      builder.create<mlir::LLVM::UndefOp>(loc, CtorStructArrayTy);

  for (uint64_t I = 0; I < globalCtors.size(); I++) {
    auto fn = globalCtors[I];
    mlir::Value structInit =
        builder.create<mlir::LLVM::UndefOp>(loc, CtorStructTy);
    mlir::Value initPriority = builder.create<mlir::LLVM::ConstantOp>(
        loc, CtorStructFields[0], fn.second);
    mlir::Value initFuncAddr = builder.create<mlir::LLVM::AddressOfOp>(
        loc, CtorStructFields[1], fn.first);
    mlir::Value initAssociate =
        builder.create<mlir::LLVM::ZeroOp>(loc, CtorStructFields[2]);
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initPriority, 0);
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initFuncAddr, 1);
    // TODO: handle associated data for initializers.
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initAssociate, 2);
    result =
        builder.create<mlir::LLVM::InsertValueOp>(loc, result, structInit, I);
  }

  builder.create<mlir::LLVM::ReturnOp>(loc, result);
}

void ConvertCIRToLLVMPass::runOnOperation() {
  auto module = getOperation();
  mlir::DataLayout dataLayout(module);
  mlir::LLVMTypeConverter converter(&getContext());
  prepareTypeConverter(converter, dataLayout);

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
                    // ,ReturnOp
                    // ,StoreOp
                    // ,YieldOp
                    >();
  // clang-format on
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::BuiltinDialect, mlir::cir::CIRDialect,
                           mlir::func::FuncDialect>();

  // Allow operations that will be lowered directly to LLVM IR.
  target.addLegalOp<mlir::cir::ZeroInitConstOp>();

  getOperation()->removeAttr("cir.sob");
  getOperation()->removeAttr("cir.lang");

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();

  // Emit the llvm.global_ctors array.
  buildCtorList(module);
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

void populateCIRToLLVMPasses(mlir::OpPassManager &pm) {
  populateCIRPreLoweringPasses(pm);
  pm.addPass(createConvertCIRToLLVMPass());
}

extern void registerCIRDialectTranslation(mlir::MLIRContext &context);

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule, LLVMContext &llvmCtx,
                             bool disableVerifier) {
  mlir::MLIRContext *mlirCtx = theModule.getContext();
  mlir::PassManager pm(mlirCtx);
  populateCIRToLLVMPasses(pm);

  // This is necessary to have line tables emitted and basic
  // debugger working. In the future we will add proper debug information
  // emission directly from our frontend.
  pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

  // FIXME(cir): this shouldn't be necessary. It's meant to be a temporary
  // workaround until we understand why some unrealized casts are being
  // emmited and how to properly avoid them.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.enableVerifier(!disableVerifier);
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
  mlir::registerOpenMPDialectTranslation(*mlirCtx);
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
