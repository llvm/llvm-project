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
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <deque>
#include <optional>
#include <set>

#include "LowerModule.h"

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

namespace {

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
  if (auto VecType = mlir::dyn_cast<mlir::cir::VectorType>(type)) {
    return VecType.getEltType();
  }
  return type;
}

mlir::LLVM::Visibility
lowerCIRVisibilityToLLVMVisibility(mlir::cir::VisibilityKind visibilityKind) {
  switch (visibilityKind) {
  case mlir::cir::VisibilityKind::Default:
    return ::mlir::LLVM::Visibility::Default;
  case mlir::cir::VisibilityKind::Hidden:
    return ::mlir::LLVM::Visibility::Hidden;
  case mlir::cir::VisibilityKind::Protected:
    return ::mlir::LLVM::Visibility::Protected;
  }
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
  mlir::DataLayout layout(parentOp->getParentOfType<mlir::ModuleOp>());
  mlir::Value ptrVal = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getIntegerType(layout.getTypeSizeInBits(ptrAttr.getType())),
      ptrAttr.getValue().getInt());
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
  return rewriter.create<mlir::LLVM::ZeroOp>(
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
    result = rewriter.create<mlir::LLVM::ZeroOp>(
        loc, converter->convertType(arrayTy));
  } else {
    result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);
  }

  // Iteratively lower each constant element of the array.
  if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts())) {
    for (auto [idx, elt] : llvm::enumerate(arrayAttr)) {
      mlir::Value init =
          lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
      result =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
    }
  }
  // TODO(cir): this diverges from traditional lowering. Normally the string
  // would be a global constant that is memcopied.
  else if (auto strAttr =
               mlir::dyn_cast<mlir::StringAttr>(constArr.getElts())) {
    auto arrayTy = mlir::dyn_cast<mlir::cir::ArrayType>(strAttr.getType());
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

// ConstVectorAttr visitor.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                mlir::cir::ConstVectorAttr constVec,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constVec.getType());
  auto loc = parentOp->getLoc();
  SmallVector<mlir::Attribute> mlirValues;
  for (auto elementAttr : constVec.getElts()) {
    mlir::Attribute mlirAttr;
    if (auto intAttr = mlir::dyn_cast<mlir::cir::IntAttr>(elementAttr)) {
      mlirAttr = rewriter.getIntegerAttr(
          converter->convertType(intAttr.getType()), intAttr.getValue());
    } else if (auto floatAttr =
                   mlir::dyn_cast<mlir::cir::FPAttr>(elementAttr)) {
      mlirAttr = rewriter.getFloatAttr(
          converter->convertType(floatAttr.getType()), floatAttr.getValue());
    } else {
      llvm_unreachable(
          "vector constant with an element that is neither an int nor a float");
    }
    mlirValues.push_back(mlirAttr);
  }
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, llvmTy,
      mlir::DenseElementsAttr::get(mlir::cast<mlir::ShapedType>(llvmTy),
                                   mlirValues));
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

  auto ptrTy = mlir::dyn_cast<mlir::cir::PointerType>(globalAttr.getType());
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
  if (const auto intAttr = mlir::dyn_cast<mlir::cir::IntAttr>(attr))
    return lowerCirAttrAsValue(parentOp, intAttr, rewriter, converter);
  if (const auto fltAttr = mlir::dyn_cast<mlir::cir::FPAttr>(attr))
    return lowerCirAttrAsValue(parentOp, fltAttr, rewriter, converter);
  if (const auto ptrAttr = mlir::dyn_cast<mlir::cir::ConstPtrAttr>(attr))
    return lowerCirAttrAsValue(parentOp, ptrAttr, rewriter, converter);
  if (const auto constStruct = mlir::dyn_cast<mlir::cir::ConstStructAttr>(attr))
    return lowerCirAttrAsValue(parentOp, constStruct, rewriter, converter);
  if (const auto constArr = mlir::dyn_cast<mlir::cir::ConstArrayAttr>(attr))
    return lowerCirAttrAsValue(parentOp, constArr, rewriter, converter);
  if (const auto constVec = mlir::dyn_cast<mlir::cir::ConstVectorAttr>(attr))
    return lowerCirAttrAsValue(parentOp, constVec, rewriter, converter);
  if (const auto boolAttr = mlir::dyn_cast<mlir::cir::BoolAttr>(attr))
    return lowerCirAttrAsValue(parentOp, boolAttr, rewriter, converter);
  if (const auto zeroAttr = mlir::dyn_cast<mlir::cir::ZeroAttr>(attr))
    return lowerCirAttrAsValue(parentOp, zeroAttr, rewriter, converter);
  if (const auto globalAttr = mlir::dyn_cast<mlir::cir::GlobalViewAttr>(attr))
    return lowerCirAttrAsValue(parentOp, globalAttr, rewriter, converter);
  if (const auto vtableAttr = mlir::dyn_cast<mlir::cir::VTableAttr>(attr))
    return lowerCirAttrAsValue(parentOp, vtableAttr, rewriter, converter);
  if (const auto typeinfoAttr = mlir::dyn_cast<mlir::cir::TypeInfoAttr>(attr))
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

mlir::LLVM::CConv convertCallingConv(mlir::cir::CallingConv callinvConv) {
  using CIR = mlir::cir::CallingConv;
  using LLVM = mlir::LLVM::CConv;

  switch (callinvConv) {
  case CIR::C:
    return LLVM::C;
  case CIR::SpirKernel:
    return LLVM::SPIR_KERNEL;
  case CIR::SpirFunction:
    return LLVM::SPIR_FUNC;
  }
  llvm_unreachable("Unknown calling convention");
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
        op, adaptor.getDst(), adaptor.getSrc(), length, op.getIsVolatile());
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

static mlir::Value getLLVMIntCast(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Value llvmSrc,
                                  mlir::IntegerType llvmDstIntTy,
                                  bool isUnsigned, uint64_t cirDstIntWidth) {
  auto cirSrcWidth =
      mlir::cast<mlir::IntegerType>(llvmSrc.getType()).getWidth();
  if (cirSrcWidth == cirDstIntWidth)
    return llvmSrc;

  auto loc = llvmSrc.getLoc();
  if (cirSrcWidth < cirDstIntWidth) {
    if (isUnsigned)
      return rewriter.create<mlir::LLVM::ZExtOp>(loc, llvmDstIntTy, llvmSrc);
    return rewriter.create<mlir::LLVM::SExtOp>(loc, llvmDstIntTy, llvmSrc);
  }

  // Otherwise truncate
  return rewriter.create<mlir::LLVM::TruncOp>(loc, llvmDstIntTy, llvmSrc);
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
    auto elementTy = tc->convertType(ptrStrideOp.getElementTy());
    auto *ctx = elementTy.getContext();

    // void and function types doesn't really have a layout to use in GEPs,
    // make it i8 instead.
    if (mlir::isa<mlir::LLVM::LLVMVoidType>(elementTy) ||
        mlir::isa<mlir::LLVM::LLVMFunctionType>(elementTy))
      elementTy = mlir::IntegerType::get(elementTy.getContext(), 8,
                                         mlir::IntegerType::Signless);

    // Zero-extend, sign-extend or trunc the pointer value.
    auto index = adaptor.getStride();
    auto width = mlir::cast<mlir::IntegerType>(index.getType()).getWidth();
    mlir::DataLayout LLVMLayout(ptrStrideOp->getParentOfType<mlir::ModuleOp>());
    auto layoutWidth =
        LLVMLayout.getTypeIndexBitwidth(adaptor.getBase().getType());
    auto indexOp = index.getDefiningOp();
    if (indexOp && layoutWidth && width != *layoutWidth) {
      // If the index comes from a subtraction, make sure the extension happens
      // before it. To achieve that, look at unary minus, which already got
      // lowered to "sub 0, x".
      auto sub = dyn_cast<mlir::LLVM::SubOp>(indexOp);
      auto unary =
          dyn_cast<mlir::cir::UnaryOp>(ptrStrideOp.getStride().getDefiningOp());
      bool rewriteSub =
          unary && unary.getKind() == mlir::cir::UnaryOpKind::Minus && sub;
      if (rewriteSub)
        index = indexOp->getOperand(1);

      // Handle the cast
      auto llvmDstType = mlir::IntegerType::get(ctx, *layoutWidth);
      index = getLLVMIntCast(rewriter, index, llvmDstType,
                             ptrStrideOp.getStride().getType().isUnsigned(),
                             *layoutWidth);

      // Rewrite the sub in front of extensions/trunc
      if (rewriteSub) {
        index = rewriter.create<mlir::LLVM::SubOp>(
            index.getLoc(), index.getType(),
            rewriter.create<mlir::LLVM::ConstantOp>(
                index.getLoc(), index.getType(),
                mlir::IntegerAttr::get(index.getType(), 0)),
            index);
        sub->erase();
      }
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        ptrStrideOp, resultTy, elementTy, adaptor.getBase(), index);
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
    mlir::Value i1Condition;

    auto hasOneUse = false;

    if (auto defOp = brOp.getCond().getDefiningOp())
      hasOneUse = defOp->getResult(0).hasOneUse();

    if (auto defOp = adaptor.getCond().getDefiningOp()) {
      if (auto zext = dyn_cast<mlir::LLVM::ZExtOp>(defOp)) {
        if (zext->use_empty() &&
            zext->getOperand(0).getType() == rewriter.getI1Type()) {
          i1Condition = zext->getOperand(0);
          if (hasOneUse)
            rewriter.eraseOp(zext);
        }
      }
    }

    if (!i1Condition)
      i1Condition = rewriter.create<mlir::LLVM::TruncOp>(
          brOp.getLoc(), rewriter.getI1Type(), adaptor.getCond());

    rewriter.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
        brOp, i1Condition, brOp.getDestTrue(), adaptor.getDestOperandsTrue(),
        brOp.getDestFalse(), adaptor.getDestOperandsFalse());

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
      const auto ptrTy = mlir::cast<mlir::cir::PointerType>(castOp.getType());
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
          mlir::cast<mlir::cir::IntType>(elementTypeIfVector(srcType));
      mlir::cir::IntType dstIntType =
          mlir::cast<mlir::cir::IntType>(elementTypeIfVector(dstType));
      rewriter.replaceOp(
          castOp,
          getLLVMIntCast(rewriter, llvmSrcVal,
                         mlir::cast<mlir::IntegerType>(llvmDstType),
                         srcIntType.isUnsigned(), dstIntType.getWidth()));
      break;
    }
    case mlir::cir::CastKind::floating: {
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy =
          getTypeConverter()->convertType(castOp.getResult().getType());

      auto srcTy = elementTypeIfVector(castOp.getSrc().getType());
      auto dstTy = elementTypeIfVector(castOp.getResult().getType());

      if (!mlir::isa<mlir::cir::CIRFPTypeInterface>(dstTy) ||
          !mlir::isa<mlir::cir::CIRFPTypeInterface>(srcTy))
        return castOp.emitError()
               << "NYI cast from " << srcTy << " to " << dstTy;

      auto getFloatWidth = [](mlir::Type ty) -> unsigned {
        return mlir::cast<mlir::cir::CIRFPTypeInterface>(ty).getWidth();
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
      auto dstTy = mlir::cast<mlir::cir::PointerType>(castOp.getType());
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::ptr_to_int: {
      auto dstTy = mlir::cast<mlir::cir::IntType>(castOp.getType());
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      return mlir::success();
    }
    case mlir::cir::CastKind::float_to_bool: {
      auto dstTy = mlir::cast<mlir::cir::BoolType>(castOp.getType());
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
      auto dstTy = mlir::cast<mlir::cir::IntType>(castOp.getType());
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmSrcTy = mlir::cast<mlir::IntegerType>(llvmSrcVal.getType());
      auto llvmDstTy =
          mlir::cast<mlir::IntegerType>(getTypeConverter()->convertType(dstTy));
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
      if (mlir::cast<mlir::cir::IntType>(
              elementTypeIfVector(castOp.getSrc().getType()))
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
      if (mlir::cast<mlir::cir::IntType>(
              elementTypeIfVector(castOp.getResult().getType()))
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
      auto zero =
          mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 64), 0);
      auto null = rewriter.create<mlir::cir::ConstantOp>(
          src.getLoc(), castOp.getSrc().getType(),
          mlir::cir::ConstPtrAttr::get(getContext(), castOp.getSrc().getType(),
                                       zero));
      rewriter.replaceOpWithNewOp<mlir::cir::CmpOp>(
          castOp, mlir::cir::BoolType::get(getContext()),
          mlir::cir::CmpOpKind::ne, castOp.getSrc(), null);
      break;
    }
    case mlir::cir::CastKind::address_space: {
      auto dstTy = castOp.getType();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy = getTypeConverter()->convertType(dstTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddrSpaceCastOp>(
          castOp, llvmDstTy, llvmSrcVal);
      break;
    }
    default: {
      return castOp.emitError("Unhandled cast kind: ")
             << castOp.getKindAttrName();
    }
    }

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

  void buildGlobalAnnotationsVar();

  virtual StringRef getArgument() const override { return "cir-flat-to-llvm"; }
  static constexpr StringRef annotationSection = "llvm.metadata";
};

mlir::LogicalResult
rewriteToCallOrInvoke(mlir::Operation *op, mlir::ValueRange callOperands,
                      mlir::ConversionPatternRewriter &rewriter,
                      const mlir::TypeConverter *converter,
                      mlir::FlatSymbolRefAttr calleeAttr,
                      mlir::Block *continueBlock = nullptr,
                      mlir::Block *landingPadBlock = nullptr) {
  llvm::SmallVector<mlir::Type, 8> llvmResults;
  auto cirResults = op->getResultTypes();
  auto callIf = cast<mlir::cir::CIRCallOpInterface>(op);

  if (converter->convertTypes(cirResults, llvmResults).failed())
    return mlir::failure();

  auto cconv = convertCallingConv(callIf.getCallingConv());

  if (calleeAttr) { // direct call
    if (landingPadBlock) {
      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
          op, llvmResults, calleeAttr, callOperands, continueBlock,
          mlir::ValueRange{}, landingPadBlock, mlir::ValueRange{});
      newOp.setCConv(cconv);
    } else {
      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, llvmResults, calleeAttr, callOperands);
      newOp.setCConv(cconv);
    }
  } else { // indirect call
    assert(op->getOperands().size() &&
           "operands list must no be empty for the indirect call");
    auto typ = op->getOperands().front().getType();
    assert(isa<mlir::cir::PointerType>(typ) && "expected pointer type");
    auto ptyp = dyn_cast<mlir::cir::PointerType>(typ);
    auto ftyp = dyn_cast<mlir::cir::FuncType>(ptyp.getPointee());
    assert(ftyp && "expected a pointer to a function as the first operand");

    if (landingPadBlock) {
      auto llvmFnTy =
          dyn_cast<mlir::LLVM::LLVMFunctionType>(converter->convertType(ftyp));
      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
          op, llvmFnTy, mlir::FlatSymbolRefAttr{}, callOperands, continueBlock,
          mlir::ValueRange{}, landingPadBlock, mlir::ValueRange{});
      newOp.setCConv(cconv);
    } else {
      auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op,
          dyn_cast<mlir::LLVM::LLVMFunctionType>(converter->convertType(ftyp)),
          callOperands);
      newOp.setCConv(cconv);
    }
  }
  return mlir::success();
}

class CIRCallLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
public:
  using OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriteToCallOrInvoke(op.getOperation(), adaptor.getOperands(),
                                 rewriter, getTypeConverter(),
                                 op.getCalleeAttr());
  }
};

class CIRTryCallLowering
    : public mlir::OpConversionPattern<mlir::cir::TryCallOp> {
public:
  using OpConversionPattern<mlir::cir::TryCallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TryCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getCallingConv() != mlir::cir::CallingConv::C) {
      return op.emitError(
          "non-C calling convention is not implemented for try_call");
    }
    return rewriteToCallOrInvoke(
        op.getOperation(), adaptor.getOperands(), rewriter, getTypeConverter(),
        op.getCalleeAttr(), op.getCont(), op.getLandingPad());
  }
};

static mlir::LLVM::LLVMStructType
getLLVMLandingPadStructTy(mlir::ConversionPatternRewriter &rewriter) {
  // Create the landing pad type: struct { ptr, i32 }
  mlir::MLIRContext *ctx = rewriter.getContext();
  auto llvmPtr = mlir::LLVM::LLVMPointerType::get(ctx);
  llvm::SmallVector<mlir::Type> structFields;
  structFields.push_back(llvmPtr);
  structFields.push_back(rewriter.getI32Type());

  return mlir::LLVM::LLVMStructType::getLiteral(ctx, structFields);
}

class CIREhInflightOpLowering
    : public mlir::OpConversionPattern<mlir::cir::EhInflightOp> {
public:
  using OpConversionPattern<mlir::cir::EhInflightOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::EhInflightOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto llvmLandingPadStructTy = getLLVMLandingPadStructTy(rewriter);
    mlir::ArrayAttr symListAttr = op.getSymTypeListAttr();
    mlir::SmallVector<mlir::Value, 4> symAddrs;

    auto llvmFn = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    assert(llvmFn && "expected LLVM function parent");
    mlir::Block *entryBlock = &llvmFn.getRegion().front();
    assert(entryBlock->isEntryBlock());

    // %x = landingpad { ptr, i32 }
    // Note that since llvm.landingpad has to be the first operation on the
    // block, any needed value for its operands has to be added somewhere else.
    if (symListAttr) {
      //   catch ptr @_ZTIi
      //   catch ptr @_ZTIPKc
      for (mlir::Attribute attr : op.getSymTypeListAttr()) {
        auto symAttr = cast<mlir::FlatSymbolRefAttr>(attr);
        // Generate `llvm.mlir.addressof` for each symbol, and place those
        // operations in the LLVM function entry basic block.
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);
        mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
            loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
            symAttr.getValue());
        symAddrs.push_back(addrOp);
      }
    } else {
      if (!op.getCleanup()) {
        //   catch ptr null
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);
        mlir::Value nullOp = rewriter.create<mlir::LLVM::ZeroOp>(
            loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
        symAddrs.push_back(nullOp);
      }
    }

    // %slot = extractvalue { ptr, i32 } %x, 0
    // %selector = extractvalue { ptr, i32 } %x, 1
    auto padOp = rewriter.create<mlir::LLVM::LandingpadOp>(
        loc, llvmLandingPadStructTy, symAddrs);
    SmallVector<int64_t> slotIdx = {0};
    SmallVector<int64_t> selectorIdx = {1};

    if (op.getCleanup())
      padOp.setCleanup(true);

    mlir::Value slot =
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, padOp, slotIdx);
    mlir::Value selector =
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, padOp, selectorIdx);

    rewriter.replaceOp(op, mlir::ValueRange{slot, selector});

    // Landing pads are required to be in LLVM functions with personality
    // attribute. FIXME: for now hardcode personality creation in order to start
    // adding exception tests, once we annotate CIR with such information,
    // change it to be in FuncOp lowering instead.
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      // Insert personality decl before the current function.
      rewriter.setInsertionPoint(llvmFn);
      auto personalityFnTy =
          mlir::LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {},
                                            /*isVarArg=*/true);
      auto personalityFn = rewriter.create<mlir::LLVM::LLVMFuncOp>(
          loc, "__gxx_personality_v0", personalityFnTy);
      llvmFn.setPersonality(personalityFn.getName());
    }
    return mlir::success();
  }
};

class CIRAllocaLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
  mlir::DataLayout const &dataLayout;

public:
  CIRAllocaLowering(mlir::TypeConverter const &typeConverter,
                    mlir::DataLayout const &dataLayout,
                    mlir::MLIRContext *context)
      : OpConversionPattern<mlir::cir::AllocaOp>(typeConverter, context),
        dataLayout(dataLayout) {}

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
    auto resultTy = getTypeConverter()->convertType(op.getResult().getType());
    // Verification between the CIR alloca AS and the one from data layout.
    {
      auto resPtrTy = mlir::cast<mlir::LLVM::LLVMPointerType>(resultTy);
      auto dlAllocaASAttr = mlir::cast_if_present<mlir::IntegerAttr>(
          dataLayout.getAllocaMemorySpace());
      // Absence means 0
      // TODO: The query for the alloca AS should be done through CIRDataLayout
      // instead to reuse the logic of interpret null attr as 0.
      auto dlAllocaAS = dlAllocaASAttr ? dlAllocaASAttr.getInt() : 0;
      if (dlAllocaAS != resPtrTy.getAddressSpace()) {
        return op.emitError() << "alloca address space doesn't match the one "
                                 "from the target data layout: "
                              << dlAllocaAS;
      }
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
        op, resultTy, elementTy, size, op.getAlignmentAttr().getInt());
    return mlir::success();
  }
};

static mlir::LLVM::AtomicOrdering
getLLVMMemOrder(std::optional<mlir::cir::MemOrder> &memorder) {
  if (!memorder)
    return mlir::LLVM::AtomicOrdering::not_atomic;
  switch (*memorder) {
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
  llvm_unreachable("unknown memory order");
}

class CIRLoadLowering : public mlir::OpConversionPattern<mlir::cir::LoadOp> {
public:
  using OpConversionPattern<mlir::cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto llvmTy =
        getTypeConverter()->convertType(op.getResult().getType());
    auto memorder = op.getMemOrder();
    auto ordering = getLLVMMemOrder(memorder);
    auto alignOpt = op.getAlignment();
    unsigned alignment = 0;
    if (!alignOpt) {
      mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
      alignment = (unsigned)layout.getTypeABIAlignment(llvmTy);
    } else {
      alignment = *alignOpt;
    }

    // TODO: nontemporal, invariant, syncscope.
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
        op, llvmTy, adaptor.getAddr(), /* alignment */ alignment,
        op.getIsVolatile(), /* nontemporal */ false,
        /* invariant */ false, ordering);
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
public:
  using OpConversionPattern<mlir::cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memorder = op.getMemOrder();
    auto ordering = getLLVMMemOrder(memorder);
    auto alignOpt = op.getAlignment();
    unsigned alignment = 0;
    if (!alignOpt) {
      const auto llvmTy =
          getTypeConverter()->convertType(op.getValue().getType());
      mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
      alignment = (unsigned)layout.getTypeABIAlignment(llvmTy);
    } else {
      alignment = *alignOpt;
    }

    // TODO: nontemporal, syncscope.
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
        op, adaptor.getValue(), adaptor.getAddr(), alignment,
        op.getIsVolatile(), /* nontemporal */ false, ordering);
    return mlir::LogicalResult::success();
  }
};

bool hasTrailingZeros(mlir::cir::ConstArrayAttr attr) {
  auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr.getElts());
  return attr.hasTrailingZeros() ||
         (array && std::count_if(array.begin(), array.end(), [](auto elt) {
            auto ar = dyn_cast<mlir::cir::ConstArrayAttr>(elt);
            return ar && hasTrailingZeros(ar);
          }));
}

static mlir::Attribute
lowerDataMemberAttr(mlir::ModuleOp moduleOp, mlir::cir::DataMemberAttr attr,
                    const mlir::TypeConverter &typeConverter) {
  mlir::DataLayout layout{moduleOp};

  uint64_t memberOffset;
  if (attr.isNullPtr()) {
    // TODO(cir): the numerical value of a null data member pointer is
    // ABI-specific and should be queried through ABI.
    assert(!MissingFeatures::targetCodeGenInfoGetNullPointer());
    memberOffset = -1ull;
  } else {
    auto memberIndex = attr.getMemberIndex().value();
    memberOffset =
        attr.getType().getClsTy().getElementOffset(layout, memberIndex);
  }

  auto underlyingIntTy = mlir::IntegerType::get(
      moduleOp->getContext(), layout.getTypeSizeInBits(attr.getType()));
  return mlir::IntegerAttr::get(underlyingIntTy, memberOffset);
}

class CIRConstantLowering
    : public mlir::OpConversionPattern<mlir::cir::ConstantOp> {
public:
  using OpConversionPattern<mlir::cir::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Attribute attr = op.getValue();

    if (mlir::isa<mlir::cir::BoolType>(op.getType())) {
      int value =
          (op.getValue() ==
           mlir::cir::BoolAttr::get(
               getContext(), ::mlir::cir::BoolType::get(getContext()), true));
      attr = rewriter.getIntegerAttr(typeConverter->convertType(op.getType()),
                                     value);
    } else if (mlir::isa<mlir::cir::IntType>(op.getType())) {
      attr = rewriter.getIntegerAttr(
          typeConverter->convertType(op.getType()),
          mlir::cast<mlir::cir::IntAttr>(op.getValue()).getValue());
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(op.getType())) {
      attr = rewriter.getFloatAttr(
          typeConverter->convertType(op.getType()),
          mlir::cast<mlir::cir::FPAttr>(op.getValue()).getValue());
    } else if (auto complexTy =
                   mlir::dyn_cast<mlir::cir::ComplexType>(op.getType())) {
      auto complexAttr = mlir::cast<mlir::cir::ComplexAttr>(op.getValue());
      auto complexElemTy = complexTy.getElementTy();
      auto complexElemLLVMTy = typeConverter->convertType(complexElemTy);

      mlir::Attribute components[2];
      if (mlir::isa<mlir::cir::IntType>(complexElemTy)) {
        components[0] = rewriter.getIntegerAttr(
            complexElemLLVMTy,
            mlir::cast<mlir::cir::IntAttr>(complexAttr.getReal()).getValue());
        components[1] = rewriter.getIntegerAttr(
            complexElemLLVMTy,
            mlir::cast<mlir::cir::IntAttr>(complexAttr.getImag()).getValue());
      } else {
        components[0] = rewriter.getFloatAttr(
            complexElemLLVMTy,
            mlir::cast<mlir::cir::FPAttr>(complexAttr.getReal()).getValue());
        components[1] = rewriter.getFloatAttr(
            complexElemLLVMTy,
            mlir::cast<mlir::cir::FPAttr>(complexAttr.getImag()).getValue());
      }

      attr = rewriter.getArrayAttr(components);
    } else if (mlir::isa<mlir::cir::PointerType>(op.getType())) {
      // Optimize with dedicated LLVM op for null pointers.
      if (mlir::isa<mlir::cir::ConstPtrAttr>(op.getValue())) {
        if (mlir::cast<mlir::cir::ConstPtrAttr>(op.getValue()).isNullValue()) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
              op, typeConverter->convertType(op.getType()));
          return mlir::success();
        }
      }
      // Lower GlobalViewAttr to llvm.mlir.addressof
      if (auto gv = mlir::dyn_cast<mlir::cir::GlobalViewAttr>(op.getValue())) {
        auto newOp = lowerCirAttrAsValue(op, gv, rewriter, getTypeConverter());
        rewriter.replaceOp(op, newOp);
        return mlir::success();
      }
      attr = op.getValue();
    } else if (mlir::isa<mlir::cir::DataMemberType>(op.getType())) {
      auto dataMember = mlir::cast<mlir::cir::DataMemberAttr>(op.getValue());
      attr = lowerDataMemberAttr(op->getParentOfType<mlir::ModuleOp>(),
                                 dataMember, *typeConverter);
    }
    // TODO(cir): constant arrays are currently just pushed into the stack using
    // the store instruction, instead of being stored as global variables and
    // then memcopyied into the stack (as done in Clang).
    else if (auto arrTy = mlir::dyn_cast<mlir::cir::ArrayType>(op.getType())) {
      // Fetch operation constant array initializer.

      auto constArr = mlir::dyn_cast<mlir::cir::ConstArrayAttr>(op.getValue());
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
                   mlir::dyn_cast<mlir::cir::ConstStructAttr>(op.getValue())) {
      // TODO(cir): this diverges from traditional lowering. Normally the
      // initializer would be a global constant that is memcopied. Here we just
      // define a local constant with llvm.undef that will be stored into the
      // stack.
      auto initVal =
          lowerCirAttrAsValue(op, structAttr, rewriter, typeConverter);
      rewriter.replaceAllUsesWith(op, initVal);
      rewriter.eraseOp(op);
      return mlir::success();
    } else if (auto strTy =
                   mlir::dyn_cast<mlir::cir::StructType>(op.getType())) {
      if (auto zero = mlir::dyn_cast<mlir::cir::ZeroAttr>(op.getValue())) {
        auto initVal = lowerCirAttrAsValue(op, zero, rewriter, typeConverter);
        rewriter.replaceAllUsesWith(op, initVal);
        rewriter.eraseOp(op);
        return mlir::success();
      }

      return op.emitError() << "unsupported lowering for struct constant type "
                            << op.getType();
    } else if (const auto vecTy =
                   mlir::dyn_cast<mlir::cir::VectorType>(op.getType())) {
      rewriter.replaceOp(op, lowerCirAttrAsValue(op, op.getValue(), rewriter,
                                                 getTypeConverter()));
      return mlir::success();
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
    auto vecTy = mlir::dyn_cast<mlir::cir::VectorType>(op.getType());
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

class CIRVectorCmpOpLowering
    : public mlir::OpConversionPattern<mlir::cir::VecCmpOp> {
public:
  using OpConversionPattern<mlir::cir::VecCmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(mlir::isa<mlir::cir::VectorType>(op.getType()) &&
           mlir::isa<mlir::cir::VectorType>(op.getLhs().getType()) &&
           mlir::isa<mlir::cir::VectorType>(op.getRhs().getType()) &&
           "Vector compare with non-vector type");
    // LLVM IR vector comparison returns a vector of i1.  This one-bit vector
    // must be sign-extended to the correct result type.
    auto elementType = elementTypeIfVector(op.getLhs().getType());
    mlir::Value bitResult;
    if (auto intType = mlir::dyn_cast<mlir::cir::IntType>(elementType)) {
      bitResult = rewriter.create<mlir::LLVM::ICmpOp>(
          op.getLoc(),
          convertCmpKindToICmpPredicate(op.getKind(), intType.isSigned()),
          adaptor.getLhs(), adaptor.getRhs());
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(elementType)) {
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
    auto vecTy = mlir::dyn_cast<mlir::cir::VectorType>(op.getType());
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
    assert(mlir::isa<mlir::cir::VectorType>(op.getType()) &&
           mlir::isa<mlir::cir::VectorType>(op.getCond().getType()) &&
           mlir::isa<mlir::cir::VectorType>(op.getVec1().getType()) &&
           mlir::isa<mlir::cir::VectorType>(op.getVec2().getType()) &&
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
    std::transform(op.getIndices().begin(), op.getIndices().end(),
                   std::back_inserter(indices), [](mlir::Attribute intAttr) {
                     return mlir::cast<mlir::cir::IntAttr>(intAttr)
                         .getValue()
                         .getSExtValue();
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
        mlir::cast<mlir::cir::VectorType>(op.getVec().getType()).getSize();
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

  /// Convert the `cir.func` attributes to `llvm.func` attributes.
  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out
  /// argument attributes.
  void
  lowerFuncAttributes(mlir::cir::FuncOp func, bool filterArgAndResAttrs,
                      SmallVectorImpl<mlir::NamedAttribute> &result) const {
    for (auto attr : func->getAttrs()) {
      if (attr.getName() == mlir::SymbolTable::getSymbolAttrName() ||
          attr.getName() == func.getFunctionTypeAttrName() ||
          attr.getName() == getLinkageAttrNameString() ||
          attr.getName() == func.getCallingConvAttrName() ||
          (filterArgAndResAttrs &&
           (attr.getName() == func.getArgAttrsAttrName() ||
            attr.getName() == func.getResAttrsAttrName())))
        continue;

      // `CIRDialectLLVMIRTranslationInterface` requires "cir." prefix for
      // dialect specific attributes, rename them.
      if (attr.getName() == func.getExtraAttrsAttrName()) {
        std::string cirName = "cir." + func.getExtraAttrsAttrName().str();
        attr.setName(mlir::StringAttr::get(getContext(), cirName));

        lowerFuncOpenCLKernelMetadata(attr);
      }
      result.push_back(attr);
    }
  }

  /// When do module translation, we can only translate LLVM-compatible types.
  /// Here we lower possible OpenCLKernelMetadataAttr to use the converted type.
  void
  lowerFuncOpenCLKernelMetadata(mlir::NamedAttribute &extraAttrsEntry) const {
    const auto attrKey = mlir::cir::OpenCLKernelMetadataAttr::getMnemonic();
    auto oldExtraAttrs =
        cast<mlir::cir::ExtraFuncAttributesAttr>(extraAttrsEntry.getValue());
    if (!oldExtraAttrs.getElements().contains(attrKey))
      return;

    mlir::NamedAttrList newExtraAttrs;
    for (auto entry : oldExtraAttrs.getElements()) {
      if (entry.getName() == attrKey) {
        auto clKernelMetadata =
            cast<mlir::cir::OpenCLKernelMetadataAttr>(entry.getValue());
        if (auto vecTypeHint = clKernelMetadata.getVecTypeHint()) {
          auto newType = typeConverter->convertType(vecTypeHint.getValue());
          auto newTypeHint = mlir::TypeAttr::get(newType);
          auto newCLKMAttr = mlir::cir::OpenCLKernelMetadataAttr::get(
              getContext(), clKernelMetadata.getWorkGroupSizeHint(),
              clKernelMetadata.getReqdWorkGroupSize(), newTypeHint,
              clKernelMetadata.getVecTypeHintSignedness(),
              clKernelMetadata.getIntelReqdSubGroupSize());
          entry.setValue(newCLKMAttr);
        }
      }
      newExtraAttrs.push_back(entry);
    }
    extraAttrsEntry.setValue(mlir::cir::ExtraFuncAttributesAttr::get(
        getContext(), newExtraAttrs.getDictionary(getContext())));
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto fnType = op.getFunctionType();
    auto isDsoLocal = op.getDsolocal();
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
    if (mlir::isa<mlir::FusedLoc>(Loc)) {
      auto FusedLoc = mlir::cast<mlir::FusedLoc>(Loc);
      Loc = FusedLoc.getLocations()[0];
    }
    assert((mlir::isa<mlir::FileLineColLoc>(Loc) ||
            mlir::isa<mlir::UnknownLoc>(Loc)) &&
           "expected single location or unknown location here");

    auto linkage = convertLinkage(op.getLinkage());
    auto cconv = convertCallingConv(op.getCallingConv());
    SmallVector<mlir::NamedAttribute, 4> attributes;
    lowerFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

    auto fn = rewriter.create<mlir::LLVM::LLVMFuncOp>(
        Loc, op.getName(), llvmFnTy, linkage, isDsoLocal, cconv,
        mlir::SymbolRefAttr(), attributes);

    fn.setVisibility_Attr(mlir::LLVM::VisibilityAttr::get(
        getContext(), lowerCIRVisibilityToLLVMVisibility(
                          op.getGlobalVisibilityAttr().getValue())));

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
    mlir::Operation *newop =
        rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), type, symbol);

    if (op.getTls()) {
      // Handle access to TLS via intrinsic.
      newop = rewriter.create<mlir::LLVM::ThreadlocalAddressOp>(
          op.getLoc(), type, newop->getResult(0));
    }

    rewriter.replaceOp(op, newop);
    return mlir::success();
  }
};

class CIRComplexCreateOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ComplexCreateOp> {
public:
  using OpConversionPattern<mlir::cir::ComplexCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ComplexCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto complexLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    auto initialComplex =
        rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), complexLLVMTy);

    int64_t position[1]{0};
    auto realComplex = rewriter.create<mlir::LLVM::InsertValueOp>(
        op->getLoc(), initialComplex, adaptor.getReal(), position);

    position[0] = 1;
    auto complex = rewriter.create<mlir::LLVM::InsertValueOp>(
        op->getLoc(), realComplex, adaptor.getImag(), position);

    rewriter.replaceOp(op, complex);
    return mlir::success();
  }
};

class CIRComplexRealOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ComplexRealOp> {
public:
  using OpConversionPattern<mlir::cir::ComplexRealOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ComplexRealOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, resultLLVMTy, adaptor.getOperand(),
        llvm::ArrayRef<std::int64_t>{0});
    return mlir::success();
  }
};

class CIRComplexImagOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ComplexImagOp> {
public:
  using OpConversionPattern<mlir::cir::ComplexImagOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ComplexImagOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, resultLLVMTy, adaptor.getOperand(),
        llvm::ArrayRef<std::int64_t>{1});
    return mlir::success();
  }
};

class CIRComplexRealPtrOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ComplexRealPtrOp> {
public:
  using OpConversionPattern<mlir::cir::ComplexRealPtrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ComplexRealPtrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operandTy =
        mlir::cast<mlir::cir::PointerType>(op.getOperand().getType());
    auto resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    auto elementLLVMTy =
        getTypeConverter()->convertType(operandTy.getPointee());

    mlir::LLVM::GEPArg gepIndices[2]{{0}, {0}};
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
        /*inbounds=*/true);

    return mlir::success();
  }
};

class CIRComplexImagPtrOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ComplexImagPtrOp> {
public:
  using OpConversionPattern<mlir::cir::ComplexImagPtrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ComplexImagPtrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operandTy =
        mlir::cast<mlir::cir::PointerType>(op.getOperand().getType());
    auto resultLLVMTy =
        getTypeConverter()->convertType(op.getResult().getType());
    auto elementLLVMTy =
        getTypeConverter()->convertType(operandTy.getPointee());

    mlir::LLVM::GEPArg gepIndices[2]{{0}, {1}};
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
        /*inbounds=*/true);

    return mlir::success();
  }
};

class CIRSwitchFlatOpLowering
    : public mlir::OpConversionPattern<mlir::cir::SwitchFlatOp> {
public:
  using OpConversionPattern<mlir::cir::SwitchFlatOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SwitchFlatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<mlir::APInt, 8> caseValues;
    if (op.getCaseValues()) {
      for (auto val : op.getCaseValues()) {
        auto intAttr = dyn_cast<mlir::cir::IntAttr>(val);
        caseValues.push_back(intAttr.getValue());
      }
    }

    llvm::SmallVector<mlir::Block *, 8> caseDestinations;
    llvm::SmallVector<mlir::ValueRange, 8> caseOperands;

    for (auto x : op.getCaseDestinations()) {
      caseDestinations.push_back(x);
    }

    for (auto x : op.getCaseOperands()) {
      caseOperands.push_back(x);
    }

    // Set switch op to branch to the newly created blocks.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SwitchOp>(
        op, adaptor.getCondition(), op.getDefaultDestination(),
        op.getDefaultOperands(), caseValues, caseDestinations, caseOperands);
    return mlir::success();
  }
};

class CIRGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GlobalOp>::OpConversionPattern;

  // Get addrspace by converting a pointer type.
  // TODO: The approach here is a little hacky. We should access the target info
  // directly to convert the address space of global op, similar to what we do
  // for type converter.
  unsigned getGlobalOpTargetAddrSpace(mlir::cir::GlobalOp op) const {
    auto tempPtrTy = mlir::cir::PointerType::get(getContext(), op.getSymType(),
                                                 op.getAddrSpaceAttr());
    return cast<mlir::LLVM::LLVMPointerType>(
               typeConverter->convertType(tempPtrTy))
        .getAddressSpace();
  }

  /// Replace CIR global with a region initialized LLVM global and update
  /// insertion point to the end of the initializer block.
  inline void setupRegionInitializedLLVMGlobalOp(
      mlir::cir::GlobalOp op, mlir::ConversionPatternRewriter &rewriter) const {
    const auto llvmType = getTypeConverter()->convertType(op.getSymType());
    SmallVector<mlir::NamedAttribute> attributes;
    auto newGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, llvmType, op.getConstant(), convertLinkage(op.getLinkage()),
        op.getSymName(), nullptr,
        /*alignment*/ op.getAlignment().value_or(0),
        /*addrSpace*/ getGlobalOpTargetAddrSpace(op),
        /*dsoLocal*/ false, /*threadLocal*/ (bool)op.getTlsModelAttr(),
        /*comdat*/ mlir::SymbolRefAttr(), attributes);
    newGlobalOp.getRegion().push_back(new mlir::Block());
    rewriter.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    // Fetch required values to create LLVM op.
    const auto llvmType = getTypeConverter()->convertType(op.getSymType());
    const auto isConst = op.getConstant();
    const auto isDsoLocal = op.getDsolocal();
    const auto linkage = convertLinkage(op.getLinkage());
    const auto symbol = op.getSymName();
    const auto loc = op.getLoc();
    std::optional<mlir::StringRef> section = op.getSection();
    std::optional<mlir::Attribute> init = op.getInitialValue();
    mlir::LLVM::VisibilityAttr visibility = mlir::LLVM::VisibilityAttr::get(
        getContext(), lowerCIRVisibilityToLLVMVisibility(
                          op.getGlobalVisibilityAttr().getValue()));

    SmallVector<mlir::NamedAttribute> attributes;
    if (section.has_value())
      attributes.push_back(rewriter.getNamedAttr(
          "section", rewriter.getStringAttr(section.value())));

    attributes.push_back(rewriter.getNamedAttr("visibility_", visibility));

    // Check for missing funcionalities.
    if (!init.has_value()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, llvmType, isConst, linkage, symbol, mlir::Attribute(),
          /*alignment*/ 0, /*addrSpace*/ getGlobalOpTargetAddrSpace(op),
          /*dsoLocal*/ isDsoLocal, /*threadLocal*/ (bool)op.getTlsModelAttr(),
          /*comdat*/ mlir::SymbolRefAttr(), attributes);
      return mlir::success();
    }

    // Initializer is a constant array: convert it to a compatible llvm init.
    if (auto constArr =
            mlir::dyn_cast<mlir::cir::ConstArrayAttr>(init.value())) {
      if (auto attr = mlir::dyn_cast<mlir::StringAttr>(constArr.getElts())) {
        init = rewriter.getStringAttr(attr.getValue());
      } else if (auto attr =
                     mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts())) {
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
    } else if (auto fltAttr = mlir::dyn_cast<mlir::cir::FPAttr>(init.value())) {
      // Initializer is a constant floating-point number: convert to MLIR
      // builtin constant.
      init = rewriter.getFloatAttr(llvmType, fltAttr.getValue());
    }
    // Initializer is a constant integer: convert to MLIR builtin constant.
    else if (auto intAttr = mlir::dyn_cast<mlir::cir::IntAttr>(init.value())) {
      init = rewriter.getIntegerAttr(llvmType, intAttr.getValue());
    } else if (auto boolAttr =
                   mlir::dyn_cast<mlir::cir::BoolAttr>(init.value())) {
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
    } else if (auto dataMemberAttr =
                   mlir::dyn_cast<mlir::cir::DataMemberAttr>(init.value())) {
      init = lowerDataMemberAttr(op->getParentOfType<mlir::ModuleOp>(),
                                 dataMemberAttr, *typeConverter);
    } else if (const auto structAttr =
                   mlir::dyn_cast<mlir::cir::ConstStructAttr>(init.value())) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          op->getLoc(),
          lowerCirAttrAsValue(op, structAttr, rewriter, typeConverter));
      return mlir::success();
    } else if (auto attr =
                   mlir::dyn_cast<mlir::cir::GlobalViewAttr>(init.value())) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          loc, lowerCirAttrAsValue(op, attr, rewriter, typeConverter));
      return mlir::success();
    } else if (const auto vtableAttr =
                   mlir::dyn_cast<mlir::cir::VTableAttr>(init.value())) {
      setupRegionInitializedLLVMGlobalOp(op, rewriter);
      rewriter.create<mlir::LLVM::ReturnOp>(
          op->getLoc(),
          lowerCirAttrAsValue(op, vtableAttr, rewriter, typeConverter));
      return mlir::success();
    } else if (const auto typeinfoAttr =
                   mlir::dyn_cast<mlir::cir::TypeInfoAttr>(init.value())) {
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
    auto llvmGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, llvmType, isConst, linkage, symbol, init.value(),
        /*alignment*/ op.getAlignment().value_or(0),
        /*addrSpace*/ getGlobalOpTargetAddrSpace(op),
        /*dsoLocal*/ false, /*threadLocal*/ (bool)op.getTlsModelAttr(),
        /*comdat*/ mlir::SymbolRefAttr(), attributes);

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (op.getComdat())
      addComdat(llvmGlobalOp, comdatOp, rewriter, mod);

    return mlir::success();
  }

private:
  mutable mlir::LLVM::ComdatOp comdatOp = nullptr;
  static void addComdat(mlir::LLVM::GlobalOp &op,
                        mlir::LLVM::ComdatOp &comdatOp,
                        mlir::OpBuilder &builder, mlir::ModuleOp &module) {
    StringRef comdatName("__llvm_comdat_globals");
    if (!comdatOp) {
      builder.setInsertionPointToStart(module.getBody());
      comdatOp =
          builder.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
    }
    builder.setInsertionPointToStart(&comdatOp.getBody().back());
    auto selectorOp = builder.create<mlir::LLVM::ComdatSelectorOp>(
        comdatOp.getLoc(), op.getSymName(), mlir::LLVM::comdat::Comdat::Any);
    op.setComdatAttr(mlir::SymbolRefAttr::get(
        builder.getContext(), comdatName,
        mlir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
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
    bool IsVector = mlir::isa<mlir::cir::VectorType>(type);
    auto llvmType = getTypeConverter()->convertType(type);
    auto loc = op.getLoc();

    // Integer unary operations: + - ~ ++ --
    if (mlir::isa<mlir::cir::IntType>(elementType)) {
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
          auto NumElements =
              mlir::dyn_cast<mlir::cir::VectorType>(type).getSize();
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
    if (mlir::isa<mlir::cir::CIRFPTypeInterface>(elementType)) {
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
        rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(
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
    if (mlir::isa<mlir::cir::BoolType>(elementType)) {
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
    if (mlir::isa<mlir::cir::PointerType>(elementType)) {
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

  mlir::LLVM::IntegerOverflowFlags
  getIntOverflowFlag(mlir::cir::BinOp op) const {
    if (op.getNoUnsignedWrap())
      return mlir::LLVM::IntegerOverflowFlags::nuw;

    if (op.getNoSignedWrap())
      return mlir::LLVM::IntegerOverflowFlags::nsw;

    return mlir::LLVM::IntegerOverflowFlags::none;
  }

public:
  using OpConversionPattern<mlir::cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert((op.getLhs().getType() == op.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type type = op.getRhs().getType();
    assert((mlir::isa<mlir::cir::IntType, mlir::cir::CIRFPTypeInterface,
                      mlir::cir::VectorType>(type)) &&
           "operand type not supported yet");

    auto llvmTy = getTypeConverter()->convertType(op.getType());
    auto rhs = adaptor.getRhs();
    auto lhs = adaptor.getLhs();

    type = elementTypeIfVector(type);

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (mlir::isa<mlir::cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmTy, lhs, rhs,
                                                       getIntOverflowFlag(op));
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Sub:
      if (mlir::isa<mlir::cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmTy, lhs, rhs,
                                                       getIntOverflowFlag(op));
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Mul:
      if (mlir::isa<mlir::cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::LLVM::MulOp>(op, llvmTy, lhs, rhs,
                                                       getIntOverflowFlag(op));
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Div:
      if (auto ty = mlir::dyn_cast<mlir::cir::IntType>(type)) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::UDivOp>(op, llvmTy, lhs, rhs);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::SDivOp>(op, llvmTy, lhs, rhs);
      } else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FDivOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Rem:
      if (auto ty = mlir::dyn_cast<mlir::cir::IntType>(type)) {
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

class CIRBinOpOverflowOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BinOpOverflowOp> {
public:
  using OpConversionPattern<mlir::cir::BinOpOverflowOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BinOpOverflowOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto arithKind = op.getKind();
    auto operandTy = op.getLhs().getType();
    auto resultTy = op.getResult().getType();

    auto encompassedTyInfo = computeEncompassedTypeWidth(operandTy, resultTy);
    auto encompassedLLVMTy = rewriter.getIntegerType(encompassedTyInfo.width);

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    if (operandTy.getWidth() < encompassedTyInfo.width) {
      if (operandTy.isSigned()) {
        lhs = rewriter.create<mlir::LLVM::SExtOp>(loc, encompassedLLVMTy, lhs);
        rhs = rewriter.create<mlir::LLVM::SExtOp>(loc, encompassedLLVMTy, rhs);
      } else {
        lhs = rewriter.create<mlir::LLVM::ZExtOp>(loc, encompassedLLVMTy, lhs);
        rhs = rewriter.create<mlir::LLVM::ZExtOp>(loc, encompassedLLVMTy, rhs);
      }
    }

    auto intrinName = getLLVMIntrinName(arithKind, encompassedTyInfo.sign,
                                        encompassedTyInfo.width);
    auto intrinNameAttr = mlir::StringAttr::get(op.getContext(), intrinName);

    auto overflowLLVMTy = rewriter.getI1Type();
    auto intrinRetTy = mlir::LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {encompassedLLVMTy, overflowLLVMTy});

    auto callLLVMIntrinOp = rewriter.create<mlir::LLVM::CallIntrinsicOp>(
        loc, intrinRetTy, intrinNameAttr, mlir::ValueRange{lhs, rhs});
    auto intrinRet = callLLVMIntrinOp.getResult(0);

    auto result = rewriter
                      .create<mlir::LLVM::ExtractValueOp>(loc, intrinRet,
                                                          ArrayRef<int64_t>{0})
                      .getResult();
    auto overflow = rewriter
                        .create<mlir::LLVM::ExtractValueOp>(
                            loc, intrinRet, ArrayRef<int64_t>{1})
                        .getResult();

    if (resultTy.getWidth() < encompassedTyInfo.width) {
      auto resultLLVMTy = getTypeConverter()->convertType(resultTy);
      auto truncResult =
          rewriter.create<mlir::LLVM::TruncOp>(loc, resultLLVMTy, result);

      // Extend the truncated result back to the encompassing type to check for
      // any overflows during the truncation.
      mlir::Value truncResultExt;
      if (resultTy.isSigned())
        truncResultExt = rewriter.create<mlir::LLVM::SExtOp>(
            loc, encompassedLLVMTy, truncResult);
      else
        truncResultExt = rewriter.create<mlir::LLVM::ZExtOp>(
            loc, encompassedLLVMTy, truncResult);
      auto truncOverflow = rewriter.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, truncResultExt, result);

      result = truncResult;
      overflow =
          rewriter.create<mlir::LLVM::OrOp>(loc, overflow, truncOverflow);
    }

    auto boolLLVMTy =
        getTypeConverter()->convertType(op.getOverflow().getType());
    if (boolLLVMTy != rewriter.getI1Type())
      overflow = rewriter.create<mlir::LLVM::ZExtOp>(loc, boolLLVMTy, overflow);

    rewriter.replaceOp(op, mlir::ValueRange{result, overflow});

    return mlir::success();
  }

private:
  static std::string getLLVMIntrinName(mlir::cir::BinOpOverflowKind opKind,
                                       bool isSigned, unsigned width) {
    // The intrinsic name is `@llvm.{s|u}{opKind}.with.overflow.i{width}`

    std::string name = "llvm.";

    if (isSigned)
      name.push_back('s');
    else
      name.push_back('u');

    switch (opKind) {
    case mlir::cir::BinOpOverflowKind::Add:
      name.append("add.");
      break;
    case mlir::cir::BinOpOverflowKind::Sub:
      name.append("sub.");
      break;
    case mlir::cir::BinOpOverflowKind::Mul:
      name.append("mul.");
      break;
    }

    name.append("with.overflow.i");
    name.append(std::to_string(width));

    return name;
  }

  struct EncompassedTypeInfo {
    bool sign;
    unsigned width;
  };

  static EncompassedTypeInfo
  computeEncompassedTypeWidth(mlir::cir::IntType operandTy,
                              mlir::cir::IntType resultTy) {
    auto sign = operandTy.getIsSigned() || resultTy.getIsSigned();
    auto width =
        std::max(operandTy.getWidth() + (sign && operandTy.isUnsigned()),
                 resultTy.getWidth() + (sign && resultTy.isUnsigned()));
    return {sign, width};
  }
};

class CIRShiftOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ShiftOp> {
public:
  using OpConversionPattern<mlir::cir::ShiftOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ShiftOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cirAmtTy =
        mlir::dyn_cast<mlir::cir::IntType>(op.getAmount().getType());
    auto cirValTy = mlir::dyn_cast<mlir::cir::IntType>(op.getValue().getType());
    auto llvmTy = getTypeConverter()->convertType(op.getType());
    mlir::Value amt = adaptor.getAmount();
    mlir::Value val = adaptor.getValue();

    assert(cirValTy && cirAmtTy && "non-integer shift is NYI");
    assert(cirValTy == op.getType() && "inconsistent operands' types NYI");

    // Ensure shift amount is the same type as the value. Some undefined
    // behavior might occur in the casts below as per [C99 6.5.7.3].
    amt = getLLVMIntCast(rewriter, amt, mlir::cast<mlir::IntegerType>(llvmTy),
                         !cirAmtTy.isSigned(), cirValTy.getWidth());

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

class CIRCmpOpLowering : public mlir::OpConversionPattern<mlir::cir::CmpOp> {
public:
  using OpConversionPattern<mlir::cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp cmpOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = cmpOp.getLhs().getType();
    mlir::Value llResult;

    // Lower to LLVM comparison op.
    if (auto intTy = mlir::dyn_cast<mlir::cir::IntType>(type)) {
      auto kind =
          convertCmpKindToICmpPredicate(cmpOp.getKind(), intTy.isSigned());
      llResult = rewriter.create<mlir::LLVM::ICmpOp>(
          cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ptrTy = mlir::dyn_cast<mlir::cir::PointerType>(type)) {
      auto kind = convertCmpKindToICmpPredicate(cmpOp.getKind(),
                                                /* isSigned=*/false);
      llResult = rewriter.create<mlir::LLVM::ICmpOp>(
          cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(type)) {
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

static mlir::LLVM::CallIntrinsicOp
createCallLLVMIntrinsicOp(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Location loc, const llvm::Twine &intrinsicName,
                          mlir::Type resultTy, mlir::ValueRange operands) {
  auto intrinsicNameAttr =
      mlir::StringAttr::get(rewriter.getContext(), intrinsicName);
  return rewriter.create<mlir::LLVM::CallIntrinsicOp>(
      loc, resultTy, intrinsicNameAttr, operands);
}

static mlir::LLVM::CallIntrinsicOp replaceOpWithCallLLVMIntrinsicOp(
    mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
    const llvm::Twine &intrinsicName, mlir::Type resultTy,
    mlir::ValueRange operands) {
  auto callIntrinOp = createCallLLVMIntrinsicOp(
      rewriter, op->getLoc(), intrinsicName, resultTy, operands);
  rewriter.replaceOp(op, callIntrinOp.getOperation());
  return callIntrinOp;
}

static mlir::Value createLLVMBitOp(mlir::Location loc,
                                   const llvm::Twine &llvmIntrinBaseName,
                                   mlir::Type resultTy, mlir::Value operand,
                                   std::optional<bool> poisonZeroInputFlag,
                                   mlir::ConversionPatternRewriter &rewriter) {
  auto operandIntTy = mlir::cast<mlir::IntegerType>(operand.getType());
  auto resultIntTy = mlir::cast<mlir::IntegerType>(resultTy);

  std::string llvmIntrinName =
      llvmIntrinBaseName.concat(".i")
          .concat(std::to_string(operandIntTy.getWidth()))
          .str();

  // Note that LLVM intrinsic calls to bit intrinsics have the same type as the
  // operand.
  mlir::LLVM::CallIntrinsicOp op;
  if (poisonZeroInputFlag.has_value()) {
    auto poisonZeroInputValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), static_cast<int64_t>(*poisonZeroInputFlag));
    op = createCallLLVMIntrinsicOp(rewriter, loc, llvmIntrinName,
                                   operand.getType(),
                                   {operand, poisonZeroInputValue});
  } else {
    op = createCallLLVMIntrinsicOp(rewriter, loc, llvmIntrinName,
                                   operand.getType(), operand);
  }

  return getLLVMIntCast(rewriter, op->getResult(0),
                        mlir::cast<mlir::IntegerType>(resultTy),
                        /*isUnsigned=*/true, resultIntTy.getWidth());
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

class CIRObjSizeOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ObjSizeOp> {
public:
  using OpConversionPattern<mlir::cir::ObjSizeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ObjSizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto llvmResTy = getTypeConverter()->convertType(op.getType());
    auto loc = op->getLoc();

    mlir::cir::SizeInfoType kindInfo = op.getKind();
    auto falseValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), false);
    auto trueValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), true);

    replaceOpWithCallLLVMIntrinsicOp(
        rewriter, op, "llvm.objectsize", llvmResTy,
        mlir::ValueRange{adaptor.getPtr(),
                         kindInfo == mlir::cir::SizeInfoType::max ? falseValue
                                                                  : trueValue,
                         trueValue, op.getDynamic() ? trueValue : falseValue});

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

static mlir::LLVM::AtomicOrdering getLLVMAtomicOrder(mlir::cir::MemOrder memo) {
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

class CIRAtomicCmpXchgLowering
    : public mlir::OpConversionPattern<mlir::cir::AtomicCmpXchg> {
public:
  using OpConversionPattern<mlir::cir::AtomicCmpXchg>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AtomicCmpXchg op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto expected = adaptor.getExpected();
    auto desired = adaptor.getDesired();

    // FIXME: add syncscope.
    auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
        op.getLoc(), adaptor.getPtr(), expected, desired,
        getLLVMAtomicOrder(adaptor.getSuccOrder()),
        getLLVMAtomicOrder(adaptor.getFailOrder()));
    cmpxchg.setWeak(adaptor.getWeak());
    cmpxchg.setVolatile_(adaptor.getIsVolatile());

    // Check result and apply stores accordingly.
    auto old = rewriter.create<mlir::LLVM::ExtractValueOp>(
        op.getLoc(), cmpxchg.getResult(), 0);
    auto cmp = rewriter.create<mlir::LLVM::ExtractValueOp>(
        op.getLoc(), cmpxchg.getResult(), 1);

    auto extCmp = rewriter.create<mlir::LLVM::ZExtOp>(
        op.getLoc(), rewriter.getI8Type(), cmp);
    rewriter.replaceOp(op, {old, extCmp});
    return mlir::success();
  }
};

class CIRAtomicXchgLowering
    : public mlir::OpConversionPattern<mlir::cir::AtomicXchg> {
public:
  using OpConversionPattern<mlir::cir::AtomicXchg>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AtomicXchg op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME: add syncscope.
    auto llvmOrder = getLLVMAtomicOrder(adaptor.getMemOrder());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
        op, mlir::LLVM::AtomicBinOp::xchg, adaptor.getPtr(), adaptor.getVal(),
        llvmOrder);
    return mlir::success();
  }
};

class CIRAtomicFetchLowering
    : public mlir::OpConversionPattern<mlir::cir::AtomicFetch> {
public:
  using OpConversionPattern<mlir::cir::AtomicFetch>::OpConversionPattern;

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

  mlir::Value buildMinMaxPostOp(mlir::cir::AtomicFetch op, OpAdaptor adaptor,
                                mlir::ConversionPatternRewriter &rewriter,
                                mlir::Value rmwVal, bool isSigned) const {
    auto loc = op.getLoc();
    mlir::LLVM::ICmpPredicate pred;
    if (op.getBinop() == mlir::cir::AtomicFetchKind::Max) {
      pred = isSigned ? mlir::LLVM::ICmpPredicate::sgt
                      : mlir::LLVM::ICmpPredicate::ugt;
    } else { // Min
      pred = isSigned ? mlir::LLVM::ICmpPredicate::slt
                      : mlir::LLVM::ICmpPredicate::ult;
    }

    auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(), pred),
        rmwVal, adaptor.getVal());
    return rewriter.create<mlir::LLVM::SelectOp>(loc, cmp, rmwVal,
                                                 adaptor.getVal());
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
    case mlir::cir::AtomicFetchKind::Max:
    case mlir::cir::AtomicFetchKind::Min:
      llvm_unreachable("handled in buildMinMaxPostOp");
    }
    llvm_unreachable("Unknown atomic fetch opcode");
  }

  mlir::LLVM::AtomicBinOp getLLVMAtomicBinOp(mlir::cir::AtomicFetchKind k,
                                             bool isInt,
                                             bool isSignedInt) const {
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
    case mlir::cir::AtomicFetchKind::Max: {
      if (!isInt)
        return mlir::LLVM::AtomicBinOp::fmax;
      return isSignedInt ? mlir::LLVM::AtomicBinOp::max
                         : mlir::LLVM::AtomicBinOp::umax;
    }
    case mlir::cir::AtomicFetchKind::Min: {
      if (!isInt)
        return mlir::LLVM::AtomicBinOp::fmin;
      return isSignedInt ? mlir::LLVM::AtomicBinOp::min
                         : mlir::LLVM::AtomicBinOp::umin;
    }
    }
    llvm_unreachable("Unknown atomic fetch opcode");
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AtomicFetch op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    bool isInt, isSignedInt = false; // otherwise it's float.
    if (auto intTy =
            mlir::dyn_cast<mlir::cir::IntType>(op.getVal().getType())) {
      isInt = true;
      isSignedInt = intTy.isSigned();
    } else if (mlir::isa<mlir::cir::SingleType, mlir::cir::DoubleType>(
                   op.getVal().getType()))
      isInt = false;
    else {
      return op.emitError()
             << "Unsupported type: " << adaptor.getVal().getType();
    }

    // FIXME: add syncscope.
    auto llvmOrder = getLLVMAtomicOrder(adaptor.getMemOrder());
    auto llvmBinOpc = getLLVMAtomicBinOp(op.getBinop(), isInt, isSignedInt);
    auto rmwVal = rewriter.create<mlir::LLVM::AtomicRMWOp>(
        op.getLoc(), llvmBinOpc, adaptor.getPtr(), adaptor.getVal(), llvmOrder);

    mlir::Value result = rmwVal.getRes();
    if (!op.getFetchFirst()) {
      if (op.getBinop() == mlir::cir::AtomicFetchKind::Max ||
          op.getBinop() == mlir::cir::AtomicFetchKind::Min)
        result = buildMinMaxPostOp(op, adaptor, rewriter, rmwVal.getRes(),
                                   isSignedInt);
      else
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

    auto resTy = mlir::cast<mlir::IntegerType>(
        getTypeConverter()->convertType(op.getType()));

    std::string llvmIntrinName = "llvm.bswap.i";
    llvmIntrinName.append(std::to_string(resTy.getWidth()));

    rewriter.replaceOpWithNewOp<mlir::LLVM::ByteSwapOp>(op, adaptor.getInput());

    return mlir::LogicalResult::success();
  }
};

class CIRRotateOpLowering
    : public mlir::OpConversionPattern<mlir::cir::RotateOp> {
public:
  using OpConversionPattern<mlir::cir::RotateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::RotateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Note that LLVM intrinsic calls to @llvm.fsh{r,l}.i* have the same type as
    // the operand.
    auto src = adaptor.getSrc();
    if (op.getLeft())
      rewriter.replaceOpWithNewOp<mlir::LLVM::FshlOp>(op, src, src,
                                                      adaptor.getAmt());
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FshrOp>(op, src, src,
                                                      adaptor.getAmt());
    return mlir::LogicalResult::success();
  }
};

class CIRSelectOpLowering
    : public mlir::OpConversionPattern<mlir::cir::SelectOp> {
public:
  using OpConversionPattern<mlir::cir::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto getConstantBool = [](mlir::Value value) -> std::optional<bool> {
      auto definingOp = mlir::dyn_cast_if_present<mlir::cir::ConstantOp>(
          value.getDefiningOp());
      if (!definingOp)
        return std::nullopt;

      auto constValue =
          mlir::dyn_cast<mlir::cir::BoolAttr>(definingOp.getValue());
      if (!constValue)
        return std::nullopt;

      return constValue.getValue();
    };

    // Two special cases in the LLVMIR codegen of select op:
    // - select %0, %1, false => and %0, %1
    // - select %0, true, %1 => or %0, %1
    auto trueValue = op.getTrueValue();
    auto falseValue = op.getFalseValue();
    if (mlir::isa<mlir::cir::BoolType>(trueValue.getType())) {
      if (std::optional<bool> falseValueBool = getConstantBool(falseValue);
          falseValueBool.has_value() && !*falseValueBool) {
        // select %0, %1, false => and %0, %1
        rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(
            op, adaptor.getCondition(), adaptor.getTrueValue());
        return mlir::success();
      }
      if (std::optional<bool> trueValueBool = getConstantBool(trueValue);
          trueValueBool.has_value() && *trueValueBool) {
        // select %0, true, %1 => or %0, %1
        rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(
            op, adaptor.getCondition(), adaptor.getFalseValue());
        return mlir::success();
      }
    }

    auto llvmCondition = rewriter.create<mlir::LLVM::TruncOp>(
        op.getLoc(), mlir::IntegerType::get(op->getContext(), 1),
        adaptor.getCondition());
    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
        op, llvmCondition, adaptor.getTrueValue(), adaptor.getFalseValue());

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

class CIRGetMemberOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GetMemberOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::GetMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetMemberOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto llResTy = getTypeConverter()->convertType(op.getType());
    const auto structTy =
        mlir::cast<mlir::cir::StructType>(op.getAddrTy().getPointee());
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

class CIRGetRuntimeMemberOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GetRuntimeMemberOp> {
public:
  using mlir::OpConversionPattern<
      mlir::cir::GetRuntimeMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetRuntimeMemberOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto llvmResTy = getTypeConverter()->convertType(op.getType());
    auto llvmElementTy = mlir::IntegerType::get(op.getContext(), 8);

    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, llvmResTy, llvmElementTy, adaptor.getAddr(), adaptor.getMember());
    return mlir::success();
  }
};

class CIRPtrDiffOpLowering
    : public mlir::OpConversionPattern<mlir::cir::PtrDiffOp> {
public:
  using OpConversionPattern<mlir::cir::PtrDiffOp>::OpConversionPattern;

  uint64_t getTypeSize(mlir::Type type, mlir::Operation &op) const {
    mlir::DataLayout layout(op.getParentOfType<mlir::ModuleOp>());
    // For LLVM purposes we treat void as u8.
    if (isa<mlir::cir::VoidType>(type))
      type = mlir::cir::IntType::get(type.getContext(), 8, /*isSigned=*/false);
    return llvm::divideCeil(layout.getTypeSizeInBits(type), 8);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::PtrDiffOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstTy = mlir::cast<mlir::cir::IntType>(op.getType());
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);

    auto lhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                       adaptor.getLhs());
    auto rhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                       adaptor.getRhs());

    auto diff =
        rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), llvmDstTy, lhs, rhs);

    auto ptrTy = mlir::cast<mlir::cir::PointerType>(op.getLhs().getType());
    auto typeSize = getTypeSize(ptrTy.getPointee(), *op);

    // Avoid silly division by 1.
    auto resultVal = diff.getResult();
    if (typeSize != 1) {
      auto typeSizeVal = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), llvmDstTy, mlir::IntegerAttr::get(llvmDstTy, typeSize));

      if (dstTy.isUnsigned())
        resultVal = rewriter.create<mlir::LLVM::UDivOp>(op.getLoc(), llvmDstTy,
                                                        diff, typeSizeVal);
      else
        resultVal = rewriter.create<mlir::LLVM::SDivOp>(op.getLoc(), llvmDstTy,
                                                        diff, typeSizeVal);
    }
    rewriter.replaceOp(op, resultVal);
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
    llvm::SmallVector<mlir::LLVM::GEPArg> offsets;
    mlir::Type eltType;
    if (!symAddr) {
      // Get the vtable address point from a global variable
      auto module = op->getParentOfType<mlir::ModuleOp>();
      auto *symbol =
          mlir::SymbolTable::lookupSymbolIn(module, op.getNameAttr());
      if (auto llvmSymbol = dyn_cast<mlir::LLVM::GlobalOp>(symbol)) {
        eltType = llvmSymbol.getType();
      } else if (auto cirSymbol = dyn_cast<mlir::cir::GlobalOp>(symbol)) {
        eltType = converter->convertType(cirSymbol.getSymType());
      }
      symAddr = rewriter.create<mlir::LLVM::AddressOfOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()),
          *op.getName());
      offsets = llvm::SmallVector<mlir::LLVM::GEPArg>{
          0, op.getVtableIndex(), op.getAddressPointIndex()};
    } else {
      // Get indirect vtable address point retrieval
      symAddr = adaptor.getSymAddr();
      eltType = converter->convertType(symAddr.getType());
      offsets =
          llvm::SmallVector<mlir::LLVM::GEPArg>{op.getAddressPointIndex()};
    }

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

#define GET_BUILTIN_LOWERING_CLASSES
#include "clang/CIR/Dialect/IR/CIRBuiltinsLowering.inc"

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

    rewriter.create<mlir::LLVM::Trap>(loc);

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

    // this is for the lowering to LLVM from LLVm dialect. Otherwise, if we
    // don't have the result (i.e. void type as a result of operation), the
    // element type attribute will be attached to the whole instruction, but not
    // to the operand
    if (!op.getNumResults())
      opAttrs.push_back(mlir::Attribute());

    llvm::SmallVector<mlir::Value> llvmOperands;
    llvm::SmallVector<mlir::Value> cirOperands;
    for (size_t i = 0; i < op.getOperands().size(); ++i) {
      auto llvmOps = adaptor.getOperands()[i];
      auto cirOps = op.getOperands()[i];
      llvmOperands.insert(llvmOperands.end(), llvmOps.begin(), llvmOps.end());
      cirOperands.insert(cirOperands.end(), cirOps.begin(), cirOps.end());
    }

    // so far we infer the llvm dialect element type attr from
    // CIR operand type.
    for (std::size_t i = 0; i < op.getOperandAttrs().size(); ++i) {
      if (!op.getOperandAttrs()[i]) {
        opAttrs.push_back(mlir::Attribute());
        continue;
      }

      std::vector<mlir::NamedAttribute> attrs;
      auto typ = cast<mlir::cir::PointerType>(cirOperands[i].getType());
      auto typAttr = mlir::TypeAttr::get(
          getTypeConverter()->convertType(typ.getPointee()));

      attrs.push_back(rewriter.getNamedAttr(llvmAttrName, typAttr));
      auto newDict = rewriter.getDictionaryAttr(attrs);
      opAttrs.push_back(newDict);
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::InlineAsmOp>(
        op, llResTy, llvmOperands, op.getAsmStringAttr(),
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

    if (auto arTy = mlir::dyn_cast<mlir::cir::ArrayType>(storageType))
      storageSize = arTy.getSize() * 8;
    else if (auto intTy = mlir::dyn_cast<mlir::cir::IntType>(storageType))
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

    resultVal = createIntCast(rewriter, resultVal,
                              mlir::cast<mlir::IntegerType>(resultTy));

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

    if (auto arTy = mlir::dyn_cast<mlir::cir::ArrayType>(storageType))
      storageSize = arTy.getSize() * 8;
    else if (auto intTy = mlir::dyn_cast<mlir::cir::IntType>(storageType))
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
    auto newOp =
        createIntCast(rewriter, val, mlir::cast<mlir::IntegerType>(resTy),
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

class CIRCmpThreeWayOpLowering
    : public mlir::OpConversionPattern<mlir::cir::CmpThreeWayOp> {
public:
  using mlir::OpConversionPattern<
      mlir::cir::CmpThreeWayOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpThreeWayOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.isIntegralComparison() || !op.isStrongOrdering()) {
      op.emitError() << "unsupported three-way comparison type";
      return mlir::failure();
    }

    auto cmpInfo = op.getInfo();
    assert(cmpInfo.getLt() == -1 && cmpInfo.getEq() == 0 &&
           cmpInfo.getGt() == 1);

    auto operandTy = mlir::cast<mlir::cir::IntType>(op.getLhs().getType());
    auto resultTy = op.getType();
    auto llvmIntrinsicName = getLLVMIntrinsicName(
        operandTy.isSigned(), operandTy.getWidth(), resultTy.getWidth());

    rewriter.setInsertionPoint(op);

    auto llvmLhs = adaptor.getLhs();
    auto llvmRhs = adaptor.getRhs();
    auto llvmResultTy = getTypeConverter()->convertType(resultTy);
    auto callIntrinsicOp =
        createCallLLVMIntrinsicOp(rewriter, op.getLoc(), llvmIntrinsicName,
                                  llvmResultTy, {llvmLhs, llvmRhs});

    rewriter.replaceOp(op, callIntrinsicOp);
    return mlir::success();
  }

private:
  static std::string getLLVMIntrinsicName(bool signedCmp, unsigned operandWidth,
                                          unsigned resultWidth) {
    // The intrinsic's name takes the form:
    // `llvm.<scmp|ucmp>.i<resultWidth>.i<operandWidth>`

    std::string result = "llvm.";

    if (signedCmp)
      result.append("scmp.");
    else
      result.append("ucmp.");

    // Result type part.
    result.push_back('i');
    result.append(std::to_string(resultWidth));
    result.push_back('.');

    // Operand type part.
    result.push_back('i');
    result.append(std::to_string(operandWidth));

    return result;
  }
};

class CIRClearCacheOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ClearCacheOp> {
public:
  using OpConversionPattern<mlir::cir::ClearCacheOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ClearCacheOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto begin = adaptor.getBegin();
    auto end = adaptor.getEnd();
    auto intrinNameAttr =
        mlir::StringAttr::get(op.getContext(), "llvm.clear_cache");
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, mlir::Type{}, intrinNameAttr, mlir::ValueRange{begin, end});

    return mlir::success();
  }
};

class CIRUndefOpLowering
    : public mlir::OpConversionPattern<mlir::cir::UndefOp> {

  using mlir::OpConversionPattern<mlir::cir::UndefOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UndefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto typ = getTypeConverter()->convertType(op.getRes().getType());

    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, typ);
    return mlir::success();
  }
};

class CIREhTypeIdOpLowering
    : public mlir::OpConversionPattern<mlir::cir::EhTypeIdOp> {
public:
  using OpConversionPattern<mlir::cir::EhTypeIdOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::EhTypeIdOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        op.getTypeSymAttr());
    mlir::LLVM::CallIntrinsicOp newOp = createCallLLVMIntrinsicOp(
        rewriter, op.getLoc(), "llvm.eh.typeid.for.p0", rewriter.getI32Type(),
        mlir::ValueRange{addrOp});
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// Make sure the LLVM function we are about to create a call for actually
// exists, if not create one. Returns a function
void getOrCreateLLVMFuncOp(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, mlir::ModuleOp mod,
                           mlir::LLVM::LLVMFuncOp enclosingfnOp,
                           llvm::StringRef fnName, mlir::Type fnTy) {
  auto *sourceSymbol = mlir::SymbolTable::lookupSymbolIn(mod, fnName);
  if (!sourceSymbol) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(enclosingfnOp);
    rewriter.create<mlir::LLVM::LLVMFuncOp>(loc, fnName, fnTy);
  }
}

class CIRCatchParamOpLowering
    : public mlir::OpConversionPattern<mlir::cir::CatchParamOp> {
public:
  using OpConversionPattern<mlir::cir::CatchParamOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CatchParamOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto modOp = op->getParentOfType<mlir::ModuleOp>();
    auto enclosingFnOp = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (op.isBegin()) {
      // Get or create `declare ptr @__cxa_begin_catch(ptr)`
      StringRef fnName = "__cxa_begin_catch";
      auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
      auto fnTy = mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {llvmPtrTy},
                                                    /*isVarArg=*/false);
      getOrCreateLLVMFuncOp(rewriter, op.getLoc(), modOp, enclosingFnOp, fnName,
                            fnTy);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, mlir::TypeRange{llvmPtrTy}, fnName,
          mlir::ValueRange{adaptor.getExceptionPtr()});
      return mlir::success();
    } else if (op.isEnd()) {
      StringRef fnName = "__cxa_end_catch";
      auto fnTy = mlir::LLVM::LLVMFunctionType::get(
          mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {},
          /*isVarArg=*/false);
      getOrCreateLLVMFuncOp(rewriter, op.getLoc(), modOp, enclosingFnOp, fnName,
                            fnTy);
      rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), mlir::TypeRange{},
                                          fnName, mlir::ValueRange{});
      rewriter.eraseOp(op);
      return mlir::success();
    }
    llvm_unreachable("only begin/end supposed to make to lowering stage");
    return mlir::failure();
  }
};

class CIRResumeOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ResumeOp> {
public:
  using OpConversionPattern<mlir::cir::ResumeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ResumeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // %lpad.val = insertvalue { ptr, i32 } poison, ptr %exception_ptr, 0
    // %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %selector, 1
    // resume { ptr, i32 } %lpad.val2
    SmallVector<int64_t> slotIdx = {0};
    SmallVector<int64_t> selectorIdx = {1};
    auto llvmLandingPadStructTy = getLLVMLandingPadStructTy(rewriter);
    mlir::Value poison = rewriter.create<mlir::LLVM::PoisonOp>(
        op.getLoc(), llvmLandingPadStructTy);

    mlir::Value slot = rewriter.create<mlir::LLVM::InsertValueOp>(
        op.getLoc(), poison, adaptor.getExceptionPtr(), slotIdx);
    mlir::Value selector = rewriter.create<mlir::LLVM::InsertValueOp>(
        op.getLoc(), slot, adaptor.getTypeId(), selectorIdx);

    rewriter.replaceOpWithNewOp<mlir::LLVM::ResumeOp>(op, selector);
    return mlir::success();
  }
};

class CIRAllocExceptionOpLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocExceptionOp> {
public:
  using OpConversionPattern<mlir::cir::AllocExceptionOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocExceptionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get or create `declare ptr @__cxa_allocate_exception(i64)`
    StringRef fnName = "__cxa_allocate_exception";
    auto modOp = op->getParentOfType<mlir::ModuleOp>();
    auto enclosingFnOp = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto int64Ty = mlir::IntegerType::get(rewriter.getContext(), 64);
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {int64Ty},
                                                  /*isVarArg=*/false);
    getOrCreateLLVMFuncOp(rewriter, op.getLoc(), modOp, enclosingFnOp, fnName,
                          fnTy);
    auto size = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(),
                                                        adaptor.getSizeAttr());
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, mlir::TypeRange{llvmPtrTy}, fnName, mlir::ValueRange{size});
    return mlir::success();
  }
};

class CIRThrowOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ThrowOp> {
public:
  using OpConversionPattern<mlir::cir::ThrowOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ThrowOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get or create `declare void @__cxa_throw(ptr, ptr, ptr)`
    StringRef fnName = "__cxa_throw";
    auto modOp = op->getParentOfType<mlir::ModuleOp>();
    auto enclosingFnOp = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto voidTy = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(
        voidTy, {llvmPtrTy, llvmPtrTy, llvmPtrTy},
        /*isVarArg=*/false);
    getOrCreateLLVMFuncOp(rewriter, op.getLoc(), modOp, enclosingFnOp, fnName,
                          fnTy);
    mlir::Value typeInfo = rewriter.create<mlir::LLVM::AddressOfOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        adaptor.getTypeInfoAttr());

    mlir::Value dtor;
    if (op.getDtor()) {
      dtor = rewriter.create<mlir::LLVM::AddressOfOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
          adaptor.getDtorAttr());
    } else {
      dtor = rewriter.create<mlir::LLVM::ZeroOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, mlir::TypeRange{}, fnName,
        mlir::ValueRange{adaptor.getExceptionPtr(), typeInfo, dtor});
    return mlir::success();
  }
};

void populateCIRToLLVMConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter,
                                         mlir::DataLayout &dataLayout) {
  patterns.add<CIRReturnLowering>(patterns.getContext());
  patterns.add<CIRAllocaLowering>(converter, dataLayout, patterns.getContext());
  patterns.add<
      CIRCmpOpLowering, CIRSelectOpLowering, CIRBitClrsbOpLowering,
      CIRBitClzOpLowering, CIRBitCtzOpLowering, CIRBitFfsOpLowering,
      CIRBitParityOpLowering, CIRBitPopcountOpLowering,
      CIRAtomicCmpXchgLowering, CIRAtomicXchgLowering, CIRAtomicFetchLowering,
      CIRByteswapOpLowering, CIRRotateOpLowering, CIRBrCondOpLowering,
      CIRPtrStrideOpLowering, CIRCallLowering, CIRTryCallLowering,
      CIREhInflightOpLowering, CIRUnaryOpLowering, CIRBinOpLowering,
      CIRBinOpOverflowOpLowering, CIRShiftOpLowering, CIRLoadLowering,
      CIRConstantLowering, CIRStoreLowering, CIRFuncLowering, CIRCastOpLowering,
      CIRGlobalOpLowering, CIRGetGlobalOpLowering, CIRComplexCreateOpLowering,
      CIRComplexRealOpLowering, CIRComplexImagOpLowering,
      CIRComplexRealPtrOpLowering, CIRComplexImagPtrOpLowering,
      CIRVAStartLowering, CIRVAEndLowering, CIRVACopyLowering, CIRVAArgLowering,
      CIRBrOpLowering, CIRGetMemberOpLowering, CIRGetRuntimeMemberOpLowering,
      CIRSwitchFlatOpLowering, CIRPtrDiffOpLowering, CIRCopyOpLowering,
      CIRMemCpyOpLowering, CIRFAbsOpLowering, CIRExpectOpLowering,
      CIRVTableAddrPointOpLowering, CIRVectorCreateLowering,
      CIRVectorCmpOpLowering, CIRVectorSplatLowering, CIRVectorTernaryLowering,
      CIRVectorShuffleIntsLowering, CIRVectorShuffleVecLowering,
      CIRStackSaveLowering, CIRUnreachableLowering, CIRTrapLowering,
      CIRInlineAsmOpLowering, CIRSetBitfieldLowering, CIRGetBitfieldLowering,
      CIRPrefetchLowering, CIRObjSizeOpLowering, CIRIsConstantOpLowering,
      CIRCmpThreeWayOpLowering, CIRClearCacheOpLowering, CIRUndefOpLowering,
      CIREhTypeIdOpLowering, CIRCatchParamOpLowering, CIRResumeOpLowering,
      CIRAllocExceptionOpLowering, CIRThrowOpLowering
#define GET_BUILTIN_LOWERING_LIST
#include "clang/CIR/Dialect/IR/CIRBuiltinsLowering.inc"
#undef GET_BUILTIN_LOWERING_LIST
      >(converter, patterns.getContext());
}

namespace {

std::unique_ptr<mlir::cir::LowerModule>
prepareLowerModule(mlir::ModuleOp module) {
  mlir::PatternRewriter rewriter{module->getContext()};
  // If the triple is not present, e.g. CIR modules parsed from text, we
  // cannot init LowerModule properly.
  assert(!::cir::MissingFeatures::makeTripleAlwaysPresent());
  if (!module->hasAttr("cir.triple"))
    return {};
  return mlir::cir::createLowerModule(module, rewriter);
}

// FIXME: change the type of lowerModule to `LowerModule &` to have better
// lambda capturing experience. Also blocked by makeTripleAlwaysPresent.
void prepareTypeConverter(mlir::LLVMTypeConverter &converter,
                          mlir::DataLayout &dataLayout,
                          mlir::cir::LowerModule *lowerModule) {
  converter.addConversion([&, lowerModule](
                              mlir::cir::PointerType type) -> mlir::Type {
    // Drop pointee type since LLVM dialect only allows opaque pointers.

    auto addrSpace =
        mlir::cast_if_present<mlir::cir::AddressSpaceAttr>(type.getAddrSpace());
    // Null addrspace attribute indicates the default addrspace.
    if (!addrSpace)
      return mlir::LLVM::LLVMPointerType::get(type.getContext());

    assert(lowerModule && "CIR AS map is not available");
    // Pass through target addrspace and map CIR addrspace to LLVM addrspace by
    // querying the target info.
    unsigned targetAS =
        addrSpace.isTarget()
            ? addrSpace.getTargetValue()
            : lowerModule->getTargetLoweringInfo()
                  .getTargetAddrSpaceFromCIRAddrSpace(addrSpace);

    return mlir::LLVM::LLVMPointerType::get(type.getContext(), targetAS);
  });
  converter.addConversion([&](mlir::cir::DataMemberType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(),
                                  dataLayout.getTypeSizeInBits(type));
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
  converter.addConversion([&](mlir::cir::FP80Type type) -> mlir::Type {
    return mlir::FloatType::getF80(type.getContext());
  });
  converter.addConversion([&](mlir::cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](mlir::cir::FP16Type type) -> mlir::Type {
    return mlir::FloatType::getF16(type.getContext());
  });
  converter.addConversion([&](mlir::cir::BF16Type type) -> mlir::Type {
    return mlir::FloatType::getBF16(type.getContext());
  });
  converter.addConversion([&](mlir::cir::ComplexType type) -> mlir::Type {
    // A complex type is lowered to an LLVM struct that contains the real and
    // imaginary part as data fields.
    mlir::Type elementTy = converter.convertType(type.getElementTy());
    mlir::Type structFields[2] = {elementTy, elementTy};
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                  structFields);
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

static void buildCtorDtorList(
    mlir::ModuleOp module, StringRef globalXtorName, StringRef llvmXtorName,
    llvm::function_ref<std::pair<StringRef, int>(mlir::Attribute)> createXtor) {
  llvm::SmallVector<std::pair<StringRef, int>, 2> globalXtors;
  for (auto namedAttr : module->getAttrs()) {
    if (namedAttr.getName() == globalXtorName) {
      for (auto attr : mlir::cast<mlir::ArrayAttr>(namedAttr.getValue()))
        globalXtors.emplace_back(createXtor(attr));
      break;
    }
  }

  if (globalXtors.empty())
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
      mlir::LLVM::LLVMArrayType::get(CtorStructTy, globalXtors.size());

  auto loc = module.getLoc();
  auto newGlobalOp = builder.create<mlir::LLVM::GlobalOp>(
      loc, CtorStructArrayTy, true, mlir::LLVM::Linkage::Appending,
      llvmXtorName, mlir::Attribute());

  newGlobalOp.getRegion().push_back(new mlir::Block());
  builder.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());

  mlir::Value result =
      builder.create<mlir::LLVM::UndefOp>(loc, CtorStructArrayTy);

  for (uint64_t I = 0; I < globalXtors.size(); I++) {
    auto fn = globalXtors[I];
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

// The unreachable code is not lowered by applyPartialConversion function
// since it traverses blocks in the dominance order. At the same time we
// do need to lower such code - otherwise verification errors occur.
// For instance, the next CIR code:
//
//    cir.func @foo(%arg0: !s32i) -> !s32i {
//      %4 = cir.cast(int_to_bool, %arg0 : !s32i), !cir.bool
//      cir.if %4 {
//        %5 = cir.const #cir.int<1> : !s32i
//        cir.return %5 : !s32i
//      } else {
//        %5 = cir.const #cir.int<0> : !s32i
//       cir.return %5 : !s32i
//      }
//     cir.return %arg0 : !s32i
//    }
//
// contains an unreachable return operation (the last one). After the flattening
// pass it will be placed into the unreachable block. And the possible error
// after the lowering pass is: error: 'cir.return' op expects parent op to be
// one of 'cir.func, cir.scope, cir.if ... The reason that this operation was
// not lowered and the new parent is llvm.func.
//
// In the future we may want to get rid of this function and use DCE pass or
// something similar. But now we need to guarantee the absence of the dialect
// verification errors.
void collect_unreachable(mlir::Operation *parent,
                         llvm::SmallVector<mlir::Operation *> &ops) {

  llvm::SmallVector<mlir::Block *> unreachable_blocks;
  parent->walk([&](mlir::Block *blk) { // check
    if (blk->hasNoPredecessors() && !blk->isEntryBlock())
      unreachable_blocks.push_back(blk);
  });

  std::set<mlir::Block *> visited;
  for (auto *root : unreachable_blocks) {
    // We create a work list for each unreachable block.
    // Thus we traverse operations in some order.
    std::deque<mlir::Block *> workList;
    workList.push_back(root);

    while (!workList.empty()) {
      auto *blk = workList.back();
      workList.pop_back();
      if (visited.count(blk))
        continue;
      visited.emplace(blk);

      for (auto &op : *blk)
        ops.push_back(&op);

      for (auto it = blk->succ_begin(); it != blk->succ_end(); ++it)
        workList.push_back(*it);
    }
  }
}

// Create a string global for annotation related string.
mlir::LLVM::GlobalOp
getAnnotationStringGlobal(mlir::StringAttr strAttr, mlir::ModuleOp &module,
                          llvm::StringMap<mlir::LLVM::GlobalOp> &globalsMap,
                          mlir::OpBuilder &globalVarBuilder,
                          mlir::Location &loc, bool isArg = false) {
  llvm::StringRef str = strAttr.getValue();
  if (!globalsMap.contains(str)) {
    auto llvmStrTy = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(module.getContext(), 8), str.size() + 1);
    auto strGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
        loc, llvmStrTy,
        /*isConstant=*/true, mlir::LLVM::Linkage::Private,
        ".str" +
            (globalsMap.empty() ? ""
                                : "." + std::to_string(globalsMap.size())) +
            ".annotation" + (isArg ? ".arg" : ""),
        mlir::StringAttr::get(module.getContext(), std::string(str) + '\0'),
        /*alignment=*/isArg ? 1 : 0);
    if (!isArg)
      strGlobalOp.setSection(ConvertCIRToLLVMPass::annotationSection);
    strGlobalOp.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
    strGlobalOp.setDsoLocal(true);
    globalsMap[str] = strGlobalOp;
  }
  return globalsMap[str];
}

mlir::Value lowerAnnotationValue(
    mlir::ArrayAttr annotValue, mlir::ModuleOp &module,
    mlir::OpBuilder &varInitBuilder, mlir::OpBuilder &globalVarBuilder,
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap,
    llvm::SmallVector<mlir::Type> &annoStructFields,
    mlir::LLVM::LLVMStructType &annoStructTy,
    mlir::LLVM::LLVMPointerType &annoPtrTy, mlir::Location &loc) {
  mlir::Value valueEntry =
      varInitBuilder.create<mlir::LLVM::UndefOp>(loc, annoStructTy);
  auto globalValueName = mlir::cast<mlir::StringAttr>(annotValue[0]);
  mlir::Operation *globalValue =
      mlir::SymbolTable::lookupSymbolIn(module, globalValueName);
  // The first field is ptr to the global value
  auto globalValueFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
      loc, annoPtrTy, globalValueName);

  valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
      loc, valueEntry, globalValueFld, 0);
  mlir::cir::AnnotationAttr annotation =
      mlir::cast<mlir::cir::AnnotationAttr>(annotValue[1]);

  // The second field is ptr to the annotation name
  mlir::StringAttr annotationName = annotation.getName();
  auto annotationNameFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
      loc, annoPtrTy,
      getAnnotationStringGlobal(annotationName, module, stringGlobalsMap,
                                globalVarBuilder, loc)
          .getSymName());

  valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
      loc, valueEntry, annotationNameFld, 1);

  // The third field is ptr to the translation unit name,
  // and the fourth field is the line number
  auto annotLoc = globalValue->getLoc();
  if (mlir::isa<mlir::FusedLoc>(annotLoc)) {
    auto FusedLoc = mlir::cast<mlir::FusedLoc>(annotLoc);
    annotLoc = FusedLoc.getLocations()[0];
  }
  auto annotFileLoc = mlir::cast<mlir::FileLineColLoc>(annotLoc);
  assert(annotFileLoc && "annotation value has to be FileLineColLoc");
  // To be consistent with clang code gen, we add trailing null char
  auto fileName = mlir::StringAttr::get(
      module.getContext(), std::string(annotFileLoc.getFilename().getValue()));
  auto fileNameFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
      loc, annoPtrTy,
      getAnnotationStringGlobal(fileName, module, stringGlobalsMap,
                                globalVarBuilder, loc)
          .getSymName());
  valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(loc, valueEntry,
                                                                fileNameFld, 2);
  unsigned int lineNo = annotFileLoc.getLine();
  auto lineNoFld = varInitBuilder.create<mlir::LLVM::ConstantOp>(
      loc, annoStructFields[3], lineNo);
  valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(loc, valueEntry,
                                                                lineNoFld, 3);
  // The fifth field is ptr to the annotation args var, it could be null
  if (annotation.isNoArgs()) {
    auto nullPtrFld = varInitBuilder.create<mlir::LLVM::ZeroOp>(loc, annoPtrTy);
    valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
        loc, valueEntry, nullPtrFld, 4);
  } else {
    mlir::ArrayAttr argsAttr = annotation.getArgs();
    // First time we see this argsAttr, create a global for it
    // and build its initializer
    if (!argsVarMap.contains(argsAttr)) {
      llvm::SmallVector<mlir::Type> argStrutFldTypes;
      llvm::SmallVector<mlir::Value> argStrutFields;
      for (mlir::Attribute arg : annotation.getArgs()) {
        if (auto strArgAttr = mlir::dyn_cast<mlir::StringAttr>(arg)) {
          // Call getAnnotationStringGlobal here to make sure
          // have a global for this string before
          // creation of the args var.
          getAnnotationStringGlobal(strArgAttr, module, argStringGlobalsMap,
                                    globalVarBuilder, loc, true);
          // This will become a ptr to the global string
          argStrutFldTypes.push_back(annoPtrTy);
        } else if (auto intArgAttr = mlir::dyn_cast<mlir::IntegerAttr>(arg)) {
          argStrutFldTypes.push_back(intArgAttr.getType());
        } else {
          llvm_unreachable("Unsupported annotation arg type");
        }
      }

      mlir::LLVM::LLVMStructType argsStructTy =
          mlir::LLVM::LLVMStructType::getLiteral(globalVarBuilder.getContext(),
                                                 argStrutFldTypes);
      auto argsGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
          loc, argsStructTy, true, mlir::LLVM::Linkage::Private,
          ".args" +
              (argsVarMap.empty() ? ""
                                  : "." + std::to_string(argsVarMap.size())) +
              ".annotation",
          mlir::Attribute());
      argsGlobalOp.setSection(ConvertCIRToLLVMPass::annotationSection);
      argsGlobalOp.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
      argsGlobalOp.setDsoLocal(true);

      // Create the initializer for this args global
      argsGlobalOp.getRegion().push_back(new mlir::Block());
      mlir::OpBuilder argsInitBuilder(module.getContext());
      argsInitBuilder.setInsertionPointToEnd(
          argsGlobalOp.getInitializerBlock());

      mlir::Value argsStructInit =
          argsInitBuilder.create<mlir::LLVM::UndefOp>(loc, argsStructTy);
      int idx = 0;
      for (mlir::Attribute arg : annotation.getArgs()) {
        if (auto strArgAttr = mlir::dyn_cast<mlir::StringAttr>(arg)) {
          // This would be simply return with existing map entry value
          // from argStringGlobalsMap as string global is already
          // created in the previous loop.
          mlir::LLVM::GlobalOp argStrVar =
              getAnnotationStringGlobal(strArgAttr, module, argStringGlobalsMap,
                                        globalVarBuilder, loc, true);
          auto argStrVarAddr = argsInitBuilder.create<mlir::LLVM::AddressOfOp>(
              loc, annoPtrTy, argStrVar.getSymName());
          argsStructInit = argsInitBuilder.create<mlir::LLVM::InsertValueOp>(
              loc, argsStructInit, argStrVarAddr, idx++);
        } else if (auto intArgAttr = mlir::dyn_cast<mlir::IntegerAttr>(arg)) {
          auto intArgFld = argsInitBuilder.create<mlir::LLVM::ConstantOp>(
              loc, intArgAttr.getType(), intArgAttr.getValue());
          argsStructInit = argsInitBuilder.create<mlir::LLVM::InsertValueOp>(
              loc, argsStructInit, intArgFld, idx++);
        } else {
          llvm_unreachable("Unsupported annotation arg type");
        }
      }
      argsInitBuilder.create<mlir::LLVM::ReturnOp>(loc, argsStructInit);
      argsVarMap[argsAttr] = argsGlobalOp;
    }
    auto argsVarView = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
        loc, annoPtrTy, argsVarMap[argsAttr].getSymName());
    valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
        loc, valueEntry, argsVarView, 4);
  }
  return valueEntry;
}

void ConvertCIRToLLVMPass::buildGlobalAnnotationsVar() {
  mlir::ModuleOp module = getOperation();
  mlir::Attribute attr = module->getAttr("cir.global_annotations");
  if (!attr)
    return;
  if (auto globalAnnotValues =
          mlir::dyn_cast<mlir::cir::GlobalAnnotationValuesAttr>(attr)) {
    auto annotationValuesArray =
        mlir::dyn_cast<mlir::ArrayAttr>(globalAnnotValues.getAnnotations());
    if (!annotationValuesArray || annotationValuesArray.empty())
      return;
    mlir::OpBuilder globalVarBuilder(module.getContext());
    globalVarBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());

    // Create a global array for annotation values with element type of
    // struct { ptr, ptr, ptr, i32, ptr }
    mlir::LLVM::LLVMPointerType annoPtrTy =
        mlir::LLVM::LLVMPointerType::get(globalVarBuilder.getContext());
    llvm::SmallVector<mlir::Type> annoStructFields;
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(globalVarBuilder.getI32Type());
    annoStructFields.push_back(annoPtrTy);

    mlir::LLVM::LLVMStructType annoStructTy =
        mlir::LLVM::LLVMStructType::getLiteral(globalVarBuilder.getContext(),
                                               annoStructFields);
    mlir::LLVM::LLVMArrayType annoStructArrayTy =
        mlir::LLVM::LLVMArrayType::get(annoStructTy,
                                       annotationValuesArray.size());
    mlir::Location loc = module.getLoc();
    auto annotationGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
        loc, annoStructArrayTy, false, mlir::LLVM::Linkage::Appending,
        "llvm.global.annotations", mlir::Attribute());
    annotationGlobalOp.setSection("llvm.metadata");
    annotationGlobalOp.getRegion().push_back(new mlir::Block());
    mlir::OpBuilder varInitBuilder(module.getContext());
    varInitBuilder.setInsertionPointToEnd(
        annotationGlobalOp.getInitializerBlock());
    // Globals created for annotation strings and args to be
    // placed before the var llvm.global.annotations.
    // This is consistent with clang code gen.
    globalVarBuilder.setInsertionPoint(annotationGlobalOp);

    mlir::Value result =
        varInitBuilder.create<mlir::LLVM::UndefOp>(loc, annoStructArrayTy);
    // Track globals created for annotation related strings
    llvm::StringMap<mlir::LLVM::GlobalOp> stringGlobalsMap;
    // Track globals created for annotation arg related strings.
    // They are different from annotation strings, as strings used in args
    // are not in annotationSection, and also has aligment 1.
    llvm::StringMap<mlir::LLVM::GlobalOp> argStringGlobalsMap;
    // Track globals created for annotation args.
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> argsVarMap;

    int idx = 0;
    for (mlir::Attribute entry : annotationValuesArray) {
      auto annotValue = cast<mlir::ArrayAttr>(entry);
      mlir::Value init = lowerAnnotationValue(
          annotValue, module, varInitBuilder, globalVarBuilder,
          stringGlobalsMap, argStringGlobalsMap, argsVarMap, annoStructFields,
          annoStructTy, annoPtrTy, loc);
      result = varInitBuilder.create<mlir::LLVM::InsertValueOp>(loc, result,
                                                                init, idx++);
    }
    varInitBuilder.create<mlir::LLVM::ReturnOp>(loc, result);
  }
}

void ConvertCIRToLLVMPass::runOnOperation() {
  auto module = getOperation();
  mlir::DataLayout dataLayout(module);
  mlir::LLVMTypeConverter converter(&getContext());
  std::unique_ptr<mlir::cir::LowerModule> lowerModule =
      prepareLowerModule(module);
  prepareTypeConverter(converter, dataLayout, lowerModule.get());

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRToLLVMConversionPatterns(patterns, converter, dataLayout);
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
  target.addLegalOp<mlir::LLVM::ZeroOp>();

  getOperation()->removeAttr("cir.sob");
  getOperation()->removeAttr("cir.lang");

  llvm::SmallVector<mlir::Operation *> ops;
  ops.push_back(module);
  collect_unreachable(module, ops);

  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();

  // Emit the llvm.global_ctors array.
  buildCtorDtorList(
      module, "cir.global_ctors", "llvm.global_ctors",
      [](mlir::Attribute attr) {
        assert(mlir::isa<mlir::cir::GlobalCtorAttr>(attr) &&
               "must be a GlobalCtorAttr");
        auto ctorAttr = mlir::cast<mlir::cir::GlobalCtorAttr>(attr);
        return std::make_pair(ctorAttr.getName(), ctorAttr.getPriority());
      });
  // Emit the llvm.global_dtors array.
  buildCtorDtorList(
      module, "cir.global_dtors", "llvm.global_dtors",
      [](mlir::Attribute attr) {
        assert(mlir::isa<mlir::cir::GlobalDtorAttr>(attr) &&
               "must be a GlobalDtorAttr");
        auto dtorAttr = mlir::cast<mlir::cir::GlobalDtorAttr>(attr);
        return std::make_pair(dtorAttr.getName(), dtorAttr.getPriority());
      });
  buildGlobalAnnotationsVar();
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
