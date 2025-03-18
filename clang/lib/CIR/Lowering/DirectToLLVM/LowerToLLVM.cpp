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

#include "LowerToLLVM.h"

#include <deque>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

namespace {
/// If the given type is a vector type, return the vector's element type.
/// Otherwise return the given type unchanged.
// TODO(cir): Return the vector element type once we have support for vectors
// instead of the identity type.
mlir::Type elementTypeIfVector(mlir::Type type) {
  assert(!cir::MissingFeatures::vectorType());
  return type;
}
} // namespace

/// Given a type convertor and a data layout, convert the given type to a type
/// that is suitable for memory operations. For example, this can be used to
/// lower cir.bool accesses to i8.
static mlir::Type convertTypeForMemory(const mlir::TypeConverter &converter,
                                       mlir::DataLayout const &dataLayout,
                                       mlir::Type type) {
  // TODO(cir): Handle other types similarly to clang's codegen
  // convertTypeForMemory
  if (isa<cir::BoolType>(type)) {
    return mlir::IntegerType::get(type.getContext(),
                                  dataLayout.getTypeSizeInBits(type));
  }

  return converter.convertType(type);
}

static mlir::Value createIntCast(mlir::OpBuilder &bld, mlir::Value src,
                                 mlir::IntegerType dstTy,
                                 bool isSigned = false) {
  mlir::Type srcTy = src.getType();
  assert(mlir::isa<mlir::IntegerType>(srcTy));

  unsigned srcWidth = mlir::cast<mlir::IntegerType>(srcTy).getWidth();
  unsigned dstWidth = mlir::cast<mlir::IntegerType>(dstTy).getWidth();
  mlir::Location loc = src.getLoc();

  if (dstWidth > srcWidth && isSigned)
    return bld.create<mlir::LLVM::SExtOp>(loc, dstTy, src);
  else if (dstWidth > srcWidth)
    return bld.create<mlir::LLVM::ZExtOp>(loc, dstTy, src);
  else if (dstWidth < srcWidth)
    return bld.create<mlir::LLVM::TruncOp>(loc, dstTy, src);
  else
    return bld.create<mlir::LLVM::BitcastOp>(loc, dstTy, src);
}

/// Emits the value from memory as expected by its users. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitFromMemory(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::DataLayout const &dataLayout,
                                  cir::LoadOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitFromMemory
  if (auto boolTy = mlir::dyn_cast<cir::BoolType>(op.getResult().getType())) {
    // Create a cast value from specified size in datalayout to i1
    assert(value.getType().isInteger(dataLayout.getTypeSizeInBits(boolTy)));
    return createIntCast(rewriter, value, rewriter.getI1Type());
  }

  return value;
}

/// Emits a value to memory with the expected scalar type. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitToMemory(mlir::ConversionPatternRewriter &rewriter,
                                mlir::DataLayout const &dataLayout,
                                mlir::Type origType, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitToMemory
  if (auto boolTy = mlir::dyn_cast<cir::BoolType>(origType)) {
    // Create zext of value from i1 to i8
    mlir::IntegerType memType =
        rewriter.getIntegerType(dataLayout.getTypeSizeInBits(boolTy));
    return createIntCast(rewriter, value, memType);
  }

  return value;
}

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage) {
  using CIR = cir::GlobalLinkageKind;
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
  llvm_unreachable("Unknown CIR linkage type");
}

static mlir::Value getLLVMIntCast(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Value llvmSrc, mlir::Type llvmDstIntTy,
                                  bool isUnsigned, uint64_t cirSrcWidth,
                                  uint64_t cirDstIntWidth) {
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

class CIRAttrToValue {
public:
  CIRAttrToValue(mlir::Operation *parentOp,
                 mlir::ConversionPatternRewriter &rewriter,
                 const mlir::TypeConverter *converter)
      : parentOp(parentOp), rewriter(rewriter), converter(converter) {}

  mlir::Value visit(mlir::Attribute attr) {
    return llvm::TypeSwitch<mlir::Attribute, mlir::Value>(attr)
        .Case<cir::IntAttr, cir::FPAttr, cir::ConstPtrAttr>(
            [&](auto attrT) { return visitCirAttr(attrT); })
        .Default([&](auto attrT) { return mlir::Value(); });
  }

  mlir::Value visitCirAttr(cir::IntAttr intAttr);
  mlir::Value visitCirAttr(cir::FPAttr fltAttr);
  mlir::Value visitCirAttr(cir::ConstPtrAttr ptrAttr);

private:
  mlir::Operation *parentOp;
  mlir::ConversionPatternRewriter &rewriter;
  const mlir::TypeConverter *converter;
};

/// IntAttr visitor.
mlir::Value CIRAttrToValue::visitCirAttr(cir::IntAttr intAttr) {
  mlir::Location loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(intAttr.getType()), intAttr.getValue());
}

/// ConstPtrAttr visitor.
mlir::Value CIRAttrToValue::visitCirAttr(cir::ConstPtrAttr ptrAttr) {
  mlir::Location loc = parentOp->getLoc();
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
mlir::Value CIRAttrToValue::visitCirAttr(cir::FPAttr fltAttr) {
  mlir::Location loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(fltAttr.getType()), fltAttr.getValue());
}

// This class handles rewriting initializer attributes for types that do not
// require region initialization.
class GlobalInitAttrRewriter {
public:
  GlobalInitAttrRewriter(mlir::Type type,
                         mlir::ConversionPatternRewriter &rewriter)
      : llvmType(type), rewriter(rewriter) {}

  mlir::Attribute visit(mlir::Attribute attr) {
    return llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(attr)
        .Case<cir::IntAttr, cir::FPAttr, cir::BoolAttr>(
            [&](auto attrT) { return visitCirAttr(attrT); })
        .Default([&](auto attrT) { return mlir::Attribute(); });
  }

  mlir::Attribute visitCirAttr(cir::IntAttr attr) {
    return rewriter.getIntegerAttr(llvmType, attr.getValue());
  }
  mlir::Attribute visitCirAttr(cir::FPAttr attr) {
    return rewriter.getFloatAttr(llvmType, attr.getValue());
  }
  mlir::Attribute visitCirAttr(cir::BoolAttr attr) {
    return rewriter.getBoolAttr(attr.getValue());
  }

private:
  mlir::Type llvmType;
  mlir::ConversionPatternRewriter &rewriter;
};

// This pass requires the CIR to be in a "flat" state. All blocks in each
// function must belong to the parent region. Once scopes and control flow
// are implemented in CIR, a pass will be run before this one to flatten
// the CIR and get it into the state that this pass requires.
struct ConvertCIRToLLVMPass
    : public mlir::PassWrapper<ConvertCIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::DLTIDialect,
                    mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();
  }
  void runOnOperation() final;

  void processCIRAttrs(mlir::ModuleOp module);

  StringRef getDescription() const override {
    return "Convert the prepared CIR dialect module to LLVM dialect";
  }

  StringRef getArgument() const override { return "cir-flat-to-llvm"; }
};

mlir::Type CIRToLLVMCastOpLowering::convertTy(mlir::Type ty) const {
  return getTypeConverter()->convertType(ty);
}

mlir::LogicalResult CIRToLLVMCastOpLowering::matchAndRewrite(
    cir::CastOp castOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // For arithmetic conversions, LLVM IR uses the same instruction to convert
  // both individual scalars and entire vectors. This lowering pass handles
  // both situations.

  switch (castOp.getKind()) {
  case cir::CastKind::array_to_ptrdecay: {
    const auto ptrTy = mlir::cast<cir::PointerType>(castOp.getType());
    mlir::Value sourceValue = adaptor.getOperands().front();
    mlir::Type targetType = convertTy(ptrTy);
    mlir::Type elementTy = convertTypeForMemory(*getTypeConverter(), dataLayout,
                                                ptrTy.getPointee());
    llvm::SmallVector<mlir::LLVM::GEPArg> offset{0};
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        castOp, targetType, elementTy, sourceValue, offset);
    break;
  }
  case cir::CastKind::int_to_bool: {
    assert(!cir::MissingFeatures::opCmp());
    mlir::Type dstType = castOp.getResult().getType();
    mlir::Type llvmDstType = getTypeConverter()->convertType(dstType);
    auto zeroBool = rewriter.create<mlir::LLVM::ConstantOp>(
        castOp.getLoc(), llvmDstType, mlir::BoolAttr::get(getContext(), false));
    rewriter.replaceOp(castOp, zeroBool);
    return castOp.emitError() << "NYI int_to_bool cast";
  }
  case cir::CastKind::integral: {
    mlir::Type srcType = castOp.getSrc().getType();
    mlir::Type dstType = castOp.getResult().getType();
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstType = getTypeConverter()->convertType(dstType);
    cir::IntType srcIntType =
        mlir::cast<cir::IntType>(elementTypeIfVector(srcType));
    cir::IntType dstIntType =
        mlir::cast<cir::IntType>(elementTypeIfVector(dstType));
    rewriter.replaceOp(castOp, getLLVMIntCast(rewriter, llvmSrcVal, llvmDstType,
                                              srcIntType.isUnsigned(),
                                              srcIntType.getWidth(),
                                              dstIntType.getWidth()));
    break;
  }
  case cir::CastKind::floating: {
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstTy =
        getTypeConverter()->convertType(castOp.getResult().getType());

    mlir::Type srcTy = elementTypeIfVector(castOp.getSrc().getType());
    mlir::Type dstTy = elementTypeIfVector(castOp.getResult().getType());

    if (!mlir::isa<cir::CIRFPTypeInterface>(dstTy) ||
        !mlir::isa<cir::CIRFPTypeInterface>(srcTy))
      return castOp.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

    auto getFloatWidth = [](mlir::Type ty) -> unsigned {
      return mlir::cast<cir::CIRFPTypeInterface>(ty).getWidth();
    };

    if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPExtOp>(castOp, llvmDstTy,
                                                       llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::int_to_ptr: {
    auto dstTy = mlir::cast<cir::PointerType>(castOp.getType());
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::ptr_to_int: {
    auto dstTy = mlir::cast<cir::IntType>(castOp.getType());
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::float_to_bool: {
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    auto kind = mlir::LLVM::FCmpPredicate::une;

    // Check if float is not equal to zero.
    auto zeroFloat = rewriter.create<mlir::LLVM::ConstantOp>(
        castOp.getLoc(), llvmSrcVal.getType(),
        mlir::FloatAttr::get(llvmSrcVal.getType(), 0.0));

    // Extend comparison result to either bool (C++) or int (C).
    rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(castOp, kind, llvmSrcVal,
                                                    zeroFloat);

    return mlir::success();
  }
  case cir::CastKind::bool_to_int: {
    auto dstTy = mlir::cast<cir::IntType>(castOp.getType());
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
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
  case cir::CastKind::bool_to_float: {
    mlir::Type dstTy = castOp.getType();
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::int_to_float: {
    mlir::Type dstTy = castOp.getType();
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (mlir::cast<cir::IntType>(elementTypeIfVector(castOp.getSrc().getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::float_to_int: {
    mlir::Type dstTy = castOp.getType();
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (mlir::cast<cir::IntType>(
            elementTypeIfVector(castOp.getResult().getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPToUIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::bitcast:
    assert(!MissingFeatures::cxxABI());
    assert(!MissingFeatures::dataMemberType());
    break;
  case cir::CastKind::ptr_to_bool:
    assert(!cir::MissingFeatures::opCmp());
    break;
  case cir::CastKind::address_space: {
    mlir::Type dstTy = castOp.getType();
    mlir::Value llvmSrcVal = adaptor.getOperands().front();
    mlir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddrSpaceCastOp>(castOp, llvmDstTy,
                                                             llvmSrcVal);
    break;
  }
  case cir::CastKind::member_ptr_to_bool:
    assert(!MissingFeatures::cxxABI());
    assert(!MissingFeatures::methodType());
    break;
  default: {
    return castOp.emitError("Unhandled cast kind: ")
           << castOp.getKindAttrName();
  }
  }

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAllocaOpLowering::matchAndRewrite(
    cir::AllocaOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::opAllocaDynAllocSize());
  mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), typeConverter->convertType(rewriter.getIndexType()),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  mlir::Type elementTy =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getAllocaType());
  mlir::Type resultTy = convertTypeForMemory(*getTypeConverter(), dataLayout,
                                             op.getResult().getType());

  assert(!cir::MissingFeatures::addressSpace());
  assert(!cir::MissingFeatures::opAllocaAnnotations());

  rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
      op, resultTy, elementTy, size, op.getAlignmentAttr().getInt());

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMReturnOpLowering::matchAndRewrite(
    cir::ReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, adaptor.getOperands());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMLoadOpLowering::matchAndRewrite(
    cir::LoadOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const mlir::Type llvmTy = convertTypeForMemory(
      *getTypeConverter(), dataLayout, op.getResult().getType());
  assert(!cir::MissingFeatures::opLoadStoreMemOrder());
  assert(!cir::MissingFeatures::opLoadStoreAlignment());
  unsigned alignment = (unsigned)dataLayout.getTypeABIAlignment(llvmTy);

  assert(!cir::MissingFeatures::lowerModeOptLevel());

  // TODO: nontemporal, syncscope.
  assert(!cir::MissingFeatures::opLoadStoreVolatile());
  mlir::LLVM::LoadOp newLoad = rewriter.create<mlir::LLVM::LoadOp>(
      op->getLoc(), llvmTy, adaptor.getAddr(), alignment,
      /*volatile=*/false, /*nontemporal=*/false,
      /*invariant=*/false, /*invariantGroup=*/false,
      mlir::LLVM::AtomicOrdering::not_atomic);

  // Convert adapted result to its original type if needed.
  mlir::Value result =
      emitFromMemory(rewriter, dataLayout, op, newLoad.getResult());
  rewriter.replaceOp(op, result);
  assert(!cir::MissingFeatures::opLoadStoreTbaa());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMStoreOpLowering::matchAndRewrite(
    cir::StoreOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::opLoadStoreMemOrder());
  assert(!cir::MissingFeatures::opLoadStoreAlignment());
  const mlir::Type llvmTy =
      getTypeConverter()->convertType(op.getValue().getType());
  unsigned alignment = (unsigned)dataLayout.getTypeABIAlignment(llvmTy);

  assert(!cir::MissingFeatures::lowerModeOptLevel());

  // Convert adapted value to its memory type if needed.
  mlir::Value value = emitToMemory(rewriter, dataLayout,
                                   op.getValue().getType(), adaptor.getValue());
  // TODO: nontemporal, syncscope.
  assert(!cir::MissingFeatures::opLoadStoreVolatile());
  mlir::LLVM::StoreOp storeOp = rewriter.create<mlir::LLVM::StoreOp>(
      op->getLoc(), value, adaptor.getAddr(), alignment, /*volatile=*/false,
      /*nontemporal=*/false, /*invariantGroup=*/false,
      mlir::LLVM::AtomicOrdering::not_atomic);
  rewriter.replaceOp(op, storeOp);
  assert(!cir::MissingFeatures::opLoadStoreTbaa());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMConstantOpLowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Attribute attr = op.getValue();

  if (mlir::isa<mlir::IntegerType>(op.getType())) {
    // Verified cir.const operations cannot actually be of these types, but the
    // lowering pass may generate temporary cir.const operations with these
    // types. This is OK since MLIR allows unverified operations to be alive
    // during a pass as long as they don't live past the end of the pass.
    attr = op.getValue();
  } else if (mlir::isa<cir::BoolType>(op.getType())) {
    int value = (op.getValue() ==
                 cir::BoolAttr::get(getContext(),
                                    cir::BoolType::get(getContext()), true));
    attr = rewriter.getIntegerAttr(typeConverter->convertType(op.getType()),
                                   value);
  } else if (mlir::isa<cir::IntType>(op.getType())) {
    assert(!cir::MissingFeatures::opGlobalViewAttr());

    attr = rewriter.getIntegerAttr(
        typeConverter->convertType(op.getType()),
        mlir::cast<cir::IntAttr>(op.getValue()).getValue());
  } else if (mlir::isa<cir::CIRFPTypeInterface>(op.getType())) {
    attr = rewriter.getFloatAttr(
        typeConverter->convertType(op.getType()),
        mlir::cast<cir::FPAttr>(op.getValue()).getValue());
  } else if (mlir::isa<cir::PointerType>(op.getType())) {
    // Optimize with dedicated LLVM op for null pointers.
    if (mlir::isa<cir::ConstPtrAttr>(op.getValue())) {
      if (mlir::cast<cir::ConstPtrAttr>(op.getValue()).isNullValue()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
            op, typeConverter->convertType(op.getType()));
        return mlir::success();
      }
    }
    assert(!cir::MissingFeatures::opGlobalViewAttr());
    attr = op.getValue();
  } else {
    return op.emitError() << "unsupported constant type " << op.getType();
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
      op, getTypeConverter()->convertType(op.getType()), attr);

  return mlir::success();
}

/// Convert the `cir.func` attributes to `llvm.func` attributes.
/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out
/// argument attributes.
void CIRToLLVMFuncOpLowering::lowerFuncAttributes(
    cir::FuncOp func, bool filterArgAndResAttrs,
    SmallVectorImpl<mlir::NamedAttribute> &result) const {
  assert(!cir::MissingFeatures::opFuncCallingConv());
  for (mlir::NamedAttribute attr : func->getAttrs()) {
    if (attr.getName() == mlir::SymbolTable::getSymbolAttrName() ||
        attr.getName() == func.getFunctionTypeAttrName() ||
        attr.getName() == getLinkageAttrNameString() ||
        (filterArgAndResAttrs &&
         (attr.getName() == func.getArgAttrsAttrName() ||
          attr.getName() == func.getResAttrsAttrName())))
      continue;

    assert(!cir::MissingFeatures::opFuncExtraAttrs());
    result.push_back(attr);
  }
}

mlir::LogicalResult CIRToLLVMFuncOpLowering::matchAndRewrite(
    cir::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  cir::FuncType fnType = op.getFunctionType();
  assert(!cir::MissingFeatures::opFuncDsolocal());
  bool isDsoLocal = false;
  mlir::TypeConverter::SignatureConversion signatureConversion(
      fnType.getNumInputs());

  for (const auto &argType : llvm::enumerate(fnType.getInputs())) {
    mlir::Type convertedType = typeConverter->convertType(argType.value());
    if (!convertedType)
      return mlir::failure();
    signatureConversion.addInputs(argType.index(), convertedType);
  }

  mlir::Type resultType =
      getTypeConverter()->convertType(fnType.getReturnType());

  // Create the LLVM function operation.
  mlir::Type llvmFnTy = mlir::LLVM::LLVMFunctionType::get(
      resultType ? resultType : mlir::LLVM::LLVMVoidType::get(getContext()),
      signatureConversion.getConvertedTypes(),
      /*isVarArg=*/fnType.isVarArg());
  // LLVMFuncOp expects a single FileLine Location instead of a fused
  // location.
  mlir::Location loc = op.getLoc();
  if (mlir::FusedLoc fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc))
    loc = fusedLoc.getLocations()[0];
  assert((mlir::isa<mlir::FileLineColLoc>(loc) ||
          mlir::isa<mlir::UnknownLoc>(loc)) &&
         "expected single location or unknown location here");

  assert(!cir::MissingFeatures::opFuncLinkage());
  mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::External;
  assert(!cir::MissingFeatures::opFuncCallingConv());
  mlir::LLVM::CConv cconv = mlir::LLVM::CConv::C;
  SmallVector<mlir::NamedAttribute, 4> attributes;
  lowerFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

  mlir::LLVM::LLVMFuncOp fn = rewriter.create<mlir::LLVM::LLVMFuncOp>(
      loc, op.getName(), llvmFnTy, linkage, isDsoLocal, cconv,
      mlir::SymbolRefAttr(), attributes);

  assert(!cir::MissingFeatures::opFuncVisibility());

  rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
  if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                         &signatureConversion)))
    return mlir::failure();

  rewriter.eraseOp(op);

  return mlir::LogicalResult::success();
}

/// Replace CIR global with a region initialized LLVM global and update
/// insertion point to the end of the initializer block.
void CIRToLLVMGlobalOpLowering::setupRegionInitializedLLVMGlobalOp(
    cir::GlobalOp op, mlir::ConversionPatternRewriter &rewriter) const {
  const mlir::Type llvmType =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getSymType());

  // FIXME: These default values are placeholders until the the equivalent
  //        attributes are available on cir.global ops. This duplicates code
  //        in CIRToLLVMGlobalOpLowering::matchAndRewrite() but that will go
  //        away when the placeholders are no longer needed.
  assert(!cir::MissingFeatures::opGlobalConstant());
  const bool isConst = false;
  assert(!cir::MissingFeatures::addressSpace());
  const unsigned addrSpace = 0;
  assert(!cir::MissingFeatures::opGlobalDSOLocal());
  const bool isDsoLocal = true;
  assert(!cir::MissingFeatures::opGlobalThreadLocal());
  const bool isThreadLocal = false;
  assert(!cir::MissingFeatures::opGlobalAlignment());
  const uint64_t alignment = 0;
  const mlir::LLVM::Linkage linkage = convertLinkage(op.getLinkage());
  const StringRef symbol = op.getSymName();

  SmallVector<mlir::NamedAttribute> attributes;
  mlir::LLVM::GlobalOp newGlobalOp =
      rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, llvmType, isConst, linkage, symbol, nullptr, alignment, addrSpace,
          isDsoLocal, isThreadLocal,
          /*comdat=*/mlir::SymbolRefAttr(), attributes);
  newGlobalOp.getRegion().emplaceBlock();
  rewriter.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());
}

mlir::LogicalResult
CIRToLLVMGlobalOpLowering::matchAndRewriteRegionInitializedGlobal(
    cir::GlobalOp op, mlir::Attribute init,
    mlir::ConversionPatternRewriter &rewriter) const {
  // TODO: Generalize this handling when more types are needed here.
  assert(isa<cir::ConstPtrAttr>(init));

  // TODO(cir): once LLVM's dialect has proper equivalent attributes this
  // should be updated. For now, we use a custom op to initialize globals
  // to the appropriate value.
  const mlir::Location loc = op.getLoc();
  setupRegionInitializedLLVMGlobalOp(op, rewriter);
  CIRAttrToValue valueConverter(op, rewriter, typeConverter);
  mlir::Value value = valueConverter.visit(init);
  rewriter.create<mlir::LLVM::ReturnOp>(loc, value);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGlobalOpLowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  std::optional<mlir::Attribute> init = op.getInitialValue();

  // Fetch required values to create LLVM op.
  const mlir::Type cirSymType = op.getSymType();

  // This is the LLVM dialect type.
  const mlir::Type llvmType =
      convertTypeForMemory(*getTypeConverter(), dataLayout, cirSymType);
  // FIXME: These default values are placeholders until the the equivalent
  //        attributes are available on cir.global ops.
  assert(!cir::MissingFeatures::opGlobalConstant());
  const bool isConst = false;
  assert(!cir::MissingFeatures::addressSpace());
  const unsigned addrSpace = 0;
  assert(!cir::MissingFeatures::opGlobalDSOLocal());
  const bool isDsoLocal = true;
  assert(!cir::MissingFeatures::opGlobalThreadLocal());
  const bool isThreadLocal = false;
  assert(!cir::MissingFeatures::opGlobalAlignment());
  const uint64_t alignment = 0;
  const mlir::LLVM::Linkage linkage = convertLinkage(op.getLinkage());
  const StringRef symbol = op.getSymName();
  SmallVector<mlir::NamedAttribute> attributes;

  if (init.has_value()) {
    if (mlir::isa<cir::FPAttr, cir::IntAttr, cir::BoolAttr>(init.value())) {
      GlobalInitAttrRewriter initRewriter(llvmType, rewriter);
      init = initRewriter.visit(init.value());
      // If initRewriter returned a null attribute, init will have a value but
      // the value will be null. If that happens, initRewriter didn't handle the
      // attribute type. It probably needs to be added to
      // GlobalInitAttrRewriter.
      if (!init.value()) {
        op.emitError() << "unsupported initializer '" << init.value() << "'";
        return mlir::failure();
      }
    } else if (mlir::isa<cir::ConstPtrAttr>(init.value())) {
      // TODO(cir): once LLVM's dialect has proper equivalent attributes this
      // should be updated. For now, we use a custom op to initialize globals
      // to the appropriate value.
      return matchAndRewriteRegionInitializedGlobal(op, init.value(), rewriter);
    } else {
      // We will only get here if new initializer types are added and this
      // code is not updated to handle them.
      op.emitError() << "unsupported initializer '" << init.value() << "'";
      return mlir::failure();
    }
  }

  // Rewrite op.
  rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
      op, llvmType, isConst, linkage, symbol, init.value_or(mlir::Attribute()),
      alignment, addrSpace, isDsoLocal, isThreadLocal,
      /*comdat=*/mlir::SymbolRefAttr(), attributes);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMUnaryOpLowering::matchAndRewrite(
    cir::UnaryOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(op.getType() == op.getInput().getType() &&
         "Unary operation's operand type and result type are different");
  mlir::Type type = op.getType();
  mlir::Type elementType = type;
  bool isVector = false;
  assert(!cir::MissingFeatures::vectorType());
  mlir::Type llvmType = getTypeConverter()->convertType(type);
  mlir::Location loc = op.getLoc();

  // Integer unary operations: + - ~ ++ --
  if (mlir::isa<cir::IntType>(elementType)) {
    mlir::LLVM::IntegerOverflowFlags maybeNSW =
        mlir::LLVM::IntegerOverflowFlags::none;
    if (mlir::dyn_cast<cir::IntType>(elementType).isSigned()) {
      assert(!cir::MissingFeatures::opUnarySignedOverflow());
      // TODO: For now, assume signed overflow is undefined. We'll need to add
      // an attribute to the unary op to control this.
      maybeNSW = mlir::LLVM::IntegerOverflowFlags::nsw;
    }

    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      assert(!isVector && "++ not allowed on vector types");
      mlir::LLVM::ConstantOp one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, mlir::IntegerAttr::get(llvmType, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(
          op, llvmType, adaptor.getInput(), one, maybeNSW);
      return mlir::success();
    }
    case cir::UnaryOpKind::Dec: {
      assert(!isVector && "-- not allowed on vector types");
      mlir::LLVM::ConstantOp one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, mlir::IntegerAttr::get(llvmType, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(
          op, llvmType, adaptor.getInput(), one, maybeNSW);
      return mlir::success();
    }
    case cir::UnaryOpKind::Plus:
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    case cir::UnaryOpKind::Minus: {
      assert(!isVector &&
             "Add vector handling when vector types are supported");
      mlir::LLVM::ConstantOp zero = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, mlir::IntegerAttr::get(llvmType, 0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(
          op, llvmType, zero, adaptor.getInput(), maybeNSW);
      return mlir::success();
    }
    case cir::UnaryOpKind::Not: {
      // bit-wise compliment operator, implemented as an XOR with -1.
      assert(!isVector &&
             "Add vector handling when vector types are supported");
      mlir::LLVM::ConstantOp minusOne = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, mlir::IntegerAttr::get(llvmType, -1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(
          op, llvmType, adaptor.getInput(), minusOne);
      return mlir::success();
    }
    }
    llvm_unreachable("Unexpected unary op for int");
  }

  // Floating point unary operations: + - ++ --
  if (mlir::isa<cir::CIRFPTypeInterface>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      assert(!isVector && "++ not allowed on vector types");
      mlir::LLVM::ConstantOp one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, rewriter.getFloatAttr(llvmType, 1.0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, one,
                                                      adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Dec: {
      assert(!isVector && "-- not allowed on vector types");
      mlir::LLVM::ConstantOp minusOne = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, rewriter.getFloatAttr(llvmType, -1.0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, minusOne,
                                                      adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Plus:
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    case cir::UnaryOpKind::Minus:
      rewriter.replaceOpWithNewOp<mlir::LLVM::FNegOp>(op, llvmType,
                                                      adaptor.getInput());
      return mlir::success();
    case cir::UnaryOpKind::Not:
      return op.emitError() << "Unary not is invalid for floating-point types";
    }
    llvm_unreachable("Unexpected unary op for float");
  }

  // Boolean unary operations: ! only. (For all others, the operand has
  // already been promoted to int.)
  if (mlir::isa<cir::BoolType>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc:
    case cir::UnaryOpKind::Dec:
    case cir::UnaryOpKind::Plus:
    case cir::UnaryOpKind::Minus:
      // Some of these are allowed in source code, but we shouldn't get here
      // with a boolean type.
      return op.emitError() << "Unsupported unary operation on boolean type";
    case cir::UnaryOpKind::Not: {
      assert(!isVector && "NYI: op! on vector mask");
      mlir::LLVM::ConstantOp one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, rewriter.getIntegerAttr(llvmType, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, llvmType,
                                                     adaptor.getInput(), one);
      return mlir::success();
    }
    }
    llvm_unreachable("Unexpected unary op for bool");
  }

  // Pointer unary operations: + only.  (++ and -- of pointers are implemented
  // with cir.ptr_stride, not cir.unary.)
  if (mlir::isa<cir::PointerType>(elementType)) {
    return op.emitError()
           << "Unary operation on pointer types is not yet implemented";
  }

  return op.emitError() << "Unary operation has unsupported type: "
                        << elementType;
}

static void prepareTypeConverter(mlir::LLVMTypeConverter &converter,
                                 mlir::DataLayout &dataLayout) {
  converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    // Drop pointee type since LLVM dialect only allows opaque pointers.
    assert(!cir::MissingFeatures::addressSpace());
    unsigned targetAS = 0;

    return mlir::LLVM::LLVMPointerType::get(type.getContext(), targetAS);
  });
  converter.addConversion([&](cir::ArrayType type) -> mlir::Type {
    mlir::Type ty =
        convertTypeForMemory(converter, dataLayout, type.getEltType());
    return mlir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 1,
                                  mlir::IntegerType::Signless);
  });
  converter.addConversion([&](cir::IntType type) -> mlir::Type {
    // LLVM doesn't work with signed types, so we drop the CIR signs here.
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::Float128Type::get(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::Float16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::BFloat16Type::get(type.getContext());
  });
}

// The applyPartialConversion function traverses blocks in the dominance order,
// so it does not lower and operations that are not reachachable from the
// operations passed in as arguments. Since we do need to lower such code in
// order to avoid verification errors occur, we cannot just pass the module op
// to applyPartialConversion. We must build a set of unreachable ops and
// explicitly add them, along with the module, to the vector we pass to
// applyPartialConversion.
//
// For instance, this CIR code:
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
//      cir.return %arg0 : !s32i
//    }
//
// contains an unreachable return operation (the last one). After the flattening
// pass it will be placed into the unreachable block. The possible error
// after the lowering pass is: error: 'cir.return' op expects parent op to be
// one of 'cir.func, cir.scope, cir.if ... The reason that this operation was
// not lowered and the new parent is llvm.func.
//
// In the future we may want to get rid of this function and use a DCE pass or
// something similar. But for now we need to guarantee the absence of the
// dialect verification errors.
static void collectUnreachable(mlir::Operation *parent,
                               llvm::SmallVector<mlir::Operation *> &ops) {

  llvm::SmallVector<mlir::Block *> unreachableBlocks;
  parent->walk([&](mlir::Block *blk) { // check
    if (blk->hasNoPredecessors() && !blk->isEntryBlock())
      unreachableBlocks.push_back(blk);
  });

  std::set<mlir::Block *> visited;
  for (mlir::Block *root : unreachableBlocks) {
    // We create a work list for each unreachable block.
    // Thus we traverse operations in some order.
    std::deque<mlir::Block *> workList;
    workList.push_back(root);

    while (!workList.empty()) {
      mlir::Block *blk = workList.back();
      workList.pop_back();
      if (visited.count(blk))
        continue;
      visited.emplace(blk);

      for (mlir::Operation &op : *blk)
        ops.push_back(&op);

      for (mlir::Block *succ : blk->getSuccessors())
        workList.push_back(succ);
    }
  }
}

void ConvertCIRToLLVMPass::processCIRAttrs(mlir::ModuleOp module) {
  // Lower the module attributes to LLVM equivalents.
  if (mlir::Attribute tripleAttr =
          module->getAttr(cir::CIRDialect::getTripleAttrName()))
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    tripleAttr);
}

void ConvertCIRToLLVMPass::runOnOperation() {
  llvm::TimeTraceScope scope("Convert CIR to LLVM Pass");

  mlir::ModuleOp module = getOperation();
  mlir::DataLayout dl(module);
  mlir::LLVMTypeConverter converter(&getContext());
  prepareTypeConverter(converter, dl);

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<CIRToLLVMReturnOpLowering>(patterns.getContext());
  // This could currently be merged with the group below, but it will get more
  // arguments later, so we'll keep it separate for now.
  patterns.add<CIRToLLVMAllocaOpLowering>(converter, patterns.getContext(), dl);
  patterns.add<CIRToLLVMLoadOpLowering>(converter, patterns.getContext(), dl);
  patterns.add<CIRToLLVMStoreOpLowering>(converter, patterns.getContext(), dl);
  patterns.add<CIRToLLVMGlobalOpLowering>(converter, patterns.getContext(), dl);
  patterns.add<CIRToLLVMCastOpLowering>(converter, patterns.getContext(), dl);
  patterns.add<CIRToLLVMConstantOpLowering>(converter, patterns.getContext(),
                                            dl);
  patterns.add<
      // clang-format off
               CIRToLLVMBrOpLowering,
               CIRToLLVMFuncOpLowering,
               CIRToLLVMTrapOpLowering,
               CIRToLLVMUnaryOpLowering
      // clang-format on
      >(converter, patterns.getContext());

  processCIRAttrs(module);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::BuiltinDialect, cir::CIRDialect,
                           mlir::func::FuncDialect>();

  llvm::SmallVector<mlir::Operation *> ops;
  ops.push_back(module);
  collectUnreachable(module, ops);

  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}

mlir::LogicalResult CIRToLLVMBrOpLowering::matchAndRewrite(
    cir::BrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, adaptor.getOperands(),
                                                op.getDest());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMTrapOpLowering::matchAndRewrite(
    cir::TrapOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Location loc = op->getLoc();
  rewriter.eraseOp(op);

  rewriter.create<mlir::LLVM::Trap>(loc);

  // Note that the call to llvm.trap is not a terminator in LLVM dialect.
  // So we must emit an additional llvm.unreachable to terminate the current
  // block.
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);

  return mlir::success();
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

void populateCIRToLLVMPasses(mlir::OpPassManager &pm) {
  mlir::populateCIRPreLoweringPasses(pm);
  pm.addPass(createConvertCIRToLLVMPass());
}

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp mlirModule, LLVMContext &llvmCtx) {
  llvm::TimeTraceScope scope("lower from CIR to LLVM directly");

  mlir::MLIRContext *mlirCtx = mlirModule.getContext();

  mlir::PassManager pm(mlirCtx);
  populateCIRToLLVMPasses(pm);

  if (mlir::failed(pm.run(mlirModule))) {
    // FIXME: Handle any errors where they occurs and return a nullptr here.
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");
  }

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  mlir::registerCIRDialectTranslation(*mlirCtx);

  llvm::TimeTraceScope translateScope("translateModuleToLLVMIR");

  StringRef moduleName = mlirModule.getName().value_or("CIRToLLVMModule");
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(mlirModule, llvmCtx, moduleName);

  if (!llvmModule) {
    // FIXME: Handle any errors where they occurs and return a nullptr here.
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");
  }

  return llvmModule;
}
} // namespace direct
} // namespace cir
