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

#include <array>
#include <deque>
#include <optional>

#include "aiir/Conversion/LLVMCommon/TypeConverter.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Types.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Export.h"
#include "aiir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

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
aiir::Type elementTypeIfVector(aiir::Type type) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(type)
      .Case<cir::VectorType, aiir::VectorType>(
          [](auto p) { return p.getElementType(); })
      .Default([](aiir::Type p) { return p; });
}
} // namespace

/// Given a type convertor and a data layout, convert the given type to a type
/// that is suitable for memory operations. For example, this can be used to
/// lower cir.bool accesses to i8.
static aiir::Type convertTypeForMemory(const aiir::TypeConverter &converter,
                                       aiir::DataLayout const &dataLayout,
                                       aiir::Type type) {
  // TODO(cir): Handle other types similarly to clang's codegen
  // convertTypeForMemory
  if (isa<cir::BoolType>(type)) {
    return aiir::IntegerType::get(type.getContext(),
                                  dataLayout.getTypeSizeInBits(type));
  }

  return converter.convertType(type);
}

static aiir::Value createIntCast(aiir::OpBuilder &bld, aiir::Value src,
                                 aiir::IntegerType dstTy,
                                 bool isSigned = false) {
  aiir::Type srcTy = src.getType();
  assert(aiir::isa<aiir::IntegerType>(srcTy));

  unsigned srcWidth = aiir::cast<aiir::IntegerType>(srcTy).getWidth();
  unsigned dstWidth = aiir::cast<aiir::IntegerType>(dstTy).getWidth();
  aiir::Location loc = src.getLoc();

  if (dstWidth > srcWidth && isSigned)
    return aiir::LLVM::SExtOp::create(bld, loc, dstTy, src);
  if (dstWidth > srcWidth)
    return aiir::LLVM::ZExtOp::create(bld, loc, dstTy, src);
  if (dstWidth < srcWidth)
    return aiir::LLVM::TruncOp::create(bld, loc, dstTy, src);
  return aiir::LLVM::BitcastOp::create(bld, loc, dstTy, src);
}

static aiir::LLVM::Visibility
lowerCIRVisibilityToLLVMVisibility(cir::VisibilityKind visibilityKind) {
  switch (visibilityKind) {
  case cir::VisibilityKind::Default:
    return ::aiir::LLVM::Visibility::Default;
  case cir::VisibilityKind::Hidden:
    return ::aiir::LLVM::Visibility::Hidden;
  case cir::VisibilityKind::Protected:
    return ::aiir::LLVM::Visibility::Protected;
  }
}

/// Emits the value from memory as expected by its users. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static aiir::Value emitFromMemory(aiir::ConversionPatternRewriter &rewriter,
                                  aiir::DataLayout const &dataLayout,
                                  cir::LoadOp op, aiir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitFromMemory
  if (auto boolTy = aiir::dyn_cast<cir::BoolType>(op.getType())) {
    // Create a cast value from specified size in datalayout to i1
    assert(value.getType().isInteger(dataLayout.getTypeSizeInBits(boolTy)));
    return createIntCast(rewriter, value, rewriter.getI1Type());
  }

  return value;
}

/// Emits a value to memory with the expected scalar type. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static aiir::Value emitToMemory(aiir::ConversionPatternRewriter &rewriter,
                                aiir::DataLayout const &dataLayout,
                                aiir::Type origType, aiir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitToMemory
  if (auto boolTy = aiir::dyn_cast<cir::BoolType>(origType)) {
    // Create zext of value from i1 to i8
    aiir::IntegerType memType =
        rewriter.getIntegerType(dataLayout.getTypeSizeInBits(boolTy));
    return createIntCast(rewriter, value, memType);
  }

  return value;
}

aiir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage) {
  using CIR = cir::GlobalLinkageKind;
  using LLVM = aiir::LLVM::Linkage;

  switch (linkage) {
  case CIR::AppendingLinkage:
    return LLVM::Appending;
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

aiir::LogicalResult CIRToLLVMCopyOpLowering::matchAndRewrite(
    cir::CopyOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::DataLayout layout(op->getParentOfType<aiir::ModuleOp>());
  const aiir::Value length = aiir::LLVM::ConstantOp::create(
      rewriter, op.getLoc(), rewriter.getI64Type(), op.getLength(layout));
  assert(!cir::MissingFeatures::aggValueSlotVolatile());
  rewriter.replaceOpWithNewOp<aiir::LLVM::MemcpyOp>(
      op, adaptor.getDst(), adaptor.getSrc(), length, op.getIsVolatile());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMMemCpyOpLowering::matchAndRewrite(
    cir::MemCpyOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::MemcpyOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
      /*isVolatile=*/false);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMMemMoveOpLowering::matchAndRewrite(
    cir::MemMoveOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::MemmoveOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
      /*isVolatile=*/false);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMMemSetOpLowering::matchAndRewrite(
    cir::MemSetOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {

  auto memset = rewriter.replaceOpWithNewOp<aiir::LLVM::MemsetOp>(
      op, adaptor.getDst(), adaptor.getVal(), adaptor.getLen(),
      /*isVolatile=*/false);

  if (op.getAlignmentAttr()) {
    // Construct a list full of empty attributes.
    llvm::SmallVector<aiir::Attribute> attrs{memset.getNumOperands(),
                                             rewriter.getDictionaryAttr({})};
    llvm::SmallVector<aiir::NamedAttribute> destAttrs;
    destAttrs.push_back(
        {aiir::LLVM::LLVMDialect::getAlignAttrName(), op.getAlignmentAttr()});
    attrs[memset.odsIndex_dst] = rewriter.getDictionaryAttr(destAttrs);

    auto arrayAttr = rewriter.getArrayAttr(attrs);
    memset.setArgAttrsAttr(arrayAttr);
  }

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMSqrtOpLowering::matchAndRewrite(
    cir::SqrtOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::SqrtOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMCosOpLowering::matchAndRewrite(
    cir::CosOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::CosOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMExpOpLowering::matchAndRewrite(
    cir::ExpOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::ExpOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMExp2OpLowering::matchAndRewrite(
    cir::Exp2Op op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::Exp2Op>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLogOpLowering::matchAndRewrite(
    cir::LogOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::LogOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLog10OpLowering::matchAndRewrite(
    cir::Log10Op op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::Log10Op>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLog2OpLowering::matchAndRewrite(
    cir::Log2Op op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::Log2Op>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMNearbyintOpLowering::matchAndRewrite(
    cir::NearbyintOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::NearbyintOp>(op, resTy,
                                                       adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMRintOpLowering::matchAndRewrite(
    cir::RintOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::RintOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMRoundOpLowering::matchAndRewrite(
    cir::RoundOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::RoundOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMRoundEvenOpLowering::matchAndRewrite(
    cir::RoundEvenOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::RoundEvenOp>(op, resTy,
                                                       adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMSinOpLowering::matchAndRewrite(
    cir::SinOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::SinOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMTanOpLowering::matchAndRewrite(
    cir::TanOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::TanOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMTruncOpLowering::matchAndRewrite(
    cir::TruncOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::FTruncOp>(op, resTy,
                                                    adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFloorOpLowering::matchAndRewrite(
    cir::FloorOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::FFloorOp>(op, resTy,
                                                    adaptor.getSrc());
  return aiir::success();
}

static aiir::Value getLLVMIntCast(aiir::ConversionPatternRewriter &rewriter,
                                  aiir::Value llvmSrc, aiir::Type llvmDstIntTy,
                                  bool isUnsigned, uint64_t cirSrcWidth,
                                  uint64_t cirDstIntWidth) {
  if (cirSrcWidth == cirDstIntWidth)
    return llvmSrc;

  auto loc = llvmSrc.getLoc();
  if (cirSrcWidth < cirDstIntWidth) {
    if (isUnsigned)
      return aiir::LLVM::ZExtOp::create(rewriter, loc, llvmDstIntTy, llvmSrc);
    return aiir::LLVM::SExtOp::create(rewriter, loc, llvmDstIntTy, llvmSrc);
  }

  // Otherwise truncate
  return aiir::LLVM::TruncOp::create(rewriter, loc, llvmDstIntTy, llvmSrc);
}

class CIRAttrToValue {
public:
  CIRAttrToValue(aiir::Operation *parentOp,
                 aiir::ConversionPatternRewriter &rewriter,
                 const aiir::TypeConverter *converter)
      : parentOp(parentOp), rewriter(rewriter), converter(converter) {}

  aiir::Value visit(aiir::Attribute attr) {
    return llvm::TypeSwitch<aiir::Attribute, aiir::Value>(attr)
        .Case<cir::BoolAttr, cir::IntAttr, cir::FPAttr, cir::ConstComplexAttr,
              cir::ConstArrayAttr, cir::ConstRecordAttr, cir::ConstVectorAttr,
              cir::ConstPtrAttr, cir::GlobalViewAttr, cir::TypeInfoAttr,
              cir::UndefAttr, cir::VTableAttr, cir::ZeroAttr>(
            [&](auto attrT) { return visitCirAttr(attrT); })
        .Default([&](auto attrT) { return aiir::Value(); });
  }

  aiir::Value visitCirAttr(cir::BoolAttr boolAttr);
  aiir::Value visitCirAttr(cir::IntAttr intAttr);
  aiir::Value visitCirAttr(cir::FPAttr fltAttr);
  aiir::Value visitCirAttr(cir::ConstComplexAttr complexAttr);
  aiir::Value visitCirAttr(cir::ConstPtrAttr ptrAttr);
  aiir::Value visitCirAttr(cir::ConstArrayAttr attr);
  aiir::Value visitCirAttr(cir::ConstRecordAttr attr);
  aiir::Value visitCirAttr(cir::ConstVectorAttr attr);
  aiir::Value visitCirAttr(cir::GlobalViewAttr attr);
  aiir::Value visitCirAttr(cir::TypeInfoAttr attr);
  aiir::Value visitCirAttr(cir::UndefAttr attr);
  aiir::Value visitCirAttr(cir::VTableAttr attr);
  aiir::Value visitCirAttr(cir::ZeroAttr attr);

private:
  aiir::Operation *parentOp;
  aiir::ConversionPatternRewriter &rewriter;
  const aiir::TypeConverter *converter;
};

/// Switches on the type of attribute and calls the appropriate conversion.
aiir::Value lowerCirAttrAsValue(aiir::Operation *parentOp,
                                const aiir::Attribute attr,
                                aiir::ConversionPatternRewriter &rewriter,
                                const aiir::TypeConverter *converter) {
  CIRAttrToValue valueConverter(parentOp, rewriter, converter);
  aiir::Value value = valueConverter.visit(attr);
  if (!value)
    llvm_unreachable("unhandled attribute type");
  return value;
}

void convertSideEffectForCall(aiir::Operation *callOp, bool isNothrow,
                              cir::SideEffect sideEffect,
                              aiir::LLVM::MemoryEffectsAttr &memoryEffect,
                              bool &noUnwind, bool &willReturn,
                              bool &noReturn) {
  using aiir::LLVM::ModRefInfo;

  switch (sideEffect) {
  case cir::SideEffect::All:
    memoryEffect = {};
    noUnwind = isNothrow;
    willReturn = false;
    break;

  case cir::SideEffect::Pure:
    memoryEffect = aiir::LLVM::MemoryEffectsAttr::get(
        callOp->getContext(), /*other=*/ModRefInfo::Ref,
        /*argMem=*/ModRefInfo::Ref,
        /*inaccessibleMem=*/ModRefInfo::Ref,
        /*errnoMem=*/ModRefInfo::Ref,
        /*targetMem0=*/ModRefInfo::Ref,
        /*targetMem1=*/ModRefInfo::Ref);
    noUnwind = true;
    willReturn = true;
    break;

  case cir::SideEffect::Const:
    memoryEffect = aiir::LLVM::MemoryEffectsAttr::get(
        callOp->getContext(), /*other=*/ModRefInfo::NoModRef,
        /*argMem=*/ModRefInfo::NoModRef,
        /*inaccessibleMem=*/ModRefInfo::NoModRef,
        /*errnoMem=*/ModRefInfo::NoModRef,
        /*targetMem0=*/ModRefInfo::NoModRef,
        /*targetMem1=*/ModRefInfo::NoModRef);
    noUnwind = true;
    willReturn = true;
    break;
  }

  noReturn = callOp->hasAttr(CIRDialect::getNoReturnAttrName());
}

static aiir::LLVM::CallIntrinsicOp
createCallLLVMIntrinsicOp(aiir::ConversionPatternRewriter &rewriter,
                          aiir::Location loc, const llvm::Twine &intrinsicName,
                          aiir::Type resultTy, aiir::ValueRange operands) {
  auto intrinsicNameAttr =
      aiir::StringAttr::get(rewriter.getContext(), intrinsicName);
  return aiir::LLVM::CallIntrinsicOp::create(rewriter, loc, resultTy,
                                             intrinsicNameAttr, operands);
}

static aiir::LLVM::CallIntrinsicOp replaceOpWithCallLLVMIntrinsicOp(
    aiir::ConversionPatternRewriter &rewriter, aiir::Operation *op,
    const llvm::Twine &intrinsicName, aiir::Type resultTy,
    aiir::ValueRange operands) {
  aiir::LLVM::CallIntrinsicOp callIntrinOp = createCallLLVMIntrinsicOp(
      rewriter, op->getLoc(), intrinsicName, resultTy, operands);
  rewriter.replaceOp(op, callIntrinOp.getOperation());
  return callIntrinOp;
}

aiir::LogicalResult CIRToLLVMLLVMIntrinsicCallOpLowering::matchAndRewrite(
    cir::LLVMIntrinsicCallOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type llvmResTy =
      getTypeConverter()->convertType(op->getResultTypes()[0]);
  if (!llvmResTy)
    return op.emitError("expected LLVM result type");
  StringRef name = op.getIntrinsicName();

  // Some LLVM intrinsics require ElementType attribute to be attached to
  // the argument of pointer type. That prevents us from generating LLVM IR
  // because from LLVM dialect, we have LLVM IR like the below which fails
  // LLVM IR verification.
  // %3 = call i64 @llvm.aarch64.ldxr.p0(ptr %2)
  // The expected LLVM IR should be like
  // %3 = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i32) %2)
  // TODO(cir): AIIR LLVM dialect should handle this part as CIR has no way
  // to set LLVM IR attribute.
  assert(!cir::MissingFeatures::intrinsicElementTypeSupport());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm." + name, llvmResTy,
                                   adaptor.getOperands());
  return aiir::success();
}

/// BoolAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::BoolAttr boolAttr) {
  aiir::Location loc = parentOp->getLoc();
  aiir::DataLayout layout(parentOp->getParentOfType<aiir::ModuleOp>());
  aiir::Value boolVal = aiir::LLVM::ConstantOp::create(
      rewriter, loc, converter->convertType(boolAttr.getType()),
      boolAttr.getValue());
  return emitToMemory(rewriter, layout, boolAttr.getType(), boolVal);
}

/// IntAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::IntAttr intAttr) {
  aiir::Location loc = parentOp->getLoc();
  return aiir::LLVM::ConstantOp::create(
      rewriter, loc, converter->convertType(intAttr.getType()),
      intAttr.getValue());
}

/// FPAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::FPAttr fltAttr) {
  aiir::Location loc = parentOp->getLoc();
  return aiir::LLVM::ConstantOp::create(
      rewriter, loc, converter->convertType(fltAttr.getType()),
      fltAttr.getValue());
}

/// ConstComplexAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::ConstComplexAttr complexAttr) {
  auto complexType = aiir::cast<cir::ComplexType>(complexAttr.getType());
  aiir::Type complexElemTy = complexType.getElementType();
  aiir::Type complexElemLLVMTy = converter->convertType(complexElemTy);

  aiir::Attribute components[2];
  if (const auto intType = aiir::dyn_cast<cir::IntType>(complexElemTy)) {
    components[0] = rewriter.getIntegerAttr(
        complexElemLLVMTy,
        aiir::cast<cir::IntAttr>(complexAttr.getReal()).getValue());
    components[1] = rewriter.getIntegerAttr(
        complexElemLLVMTy,
        aiir::cast<cir::IntAttr>(complexAttr.getImag()).getValue());
  } else {
    components[0] = rewriter.getFloatAttr(
        complexElemLLVMTy,
        aiir::cast<cir::FPAttr>(complexAttr.getReal()).getValue());
    components[1] = rewriter.getFloatAttr(
        complexElemLLVMTy,
        aiir::cast<cir::FPAttr>(complexAttr.getImag()).getValue());
  }

  aiir::Location loc = parentOp->getLoc();
  return aiir::LLVM::ConstantOp::create(
      rewriter, loc, converter->convertType(complexAttr.getType()),
      rewriter.getArrayAttr(components));
}

/// ConstPtrAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::ConstPtrAttr ptrAttr) {
  aiir::Location loc = parentOp->getLoc();
  if (ptrAttr.isNullValue()) {
    return aiir::LLVM::ZeroOp::create(
        rewriter, loc, converter->convertType(ptrAttr.getType()));
  }
  aiir::DataLayout layout(parentOp->getParentOfType<aiir::ModuleOp>());
  aiir::Value ptrVal = aiir::LLVM::ConstantOp::create(
      rewriter, loc,
      rewriter.getIntegerType(layout.getTypeSizeInBits(ptrAttr.getType())),
      ptrAttr.getValue().getInt());
  return aiir::LLVM::IntToPtrOp::create(
      rewriter, loc, converter->convertType(ptrAttr.getType()), ptrVal);
}

// ConstArrayAttr visitor
aiir::Value CIRAttrToValue::visitCirAttr(cir::ConstArrayAttr attr) {
  aiir::Type llvmTy = converter->convertType(attr.getType());
  aiir::Location loc = parentOp->getLoc();
  aiir::Value result;

  if (attr.hasTrailingZeros()) {
    aiir::Type arrayTy = attr.getType();
    result = aiir::LLVM::ZeroOp::create(rewriter, loc,
                                        converter->convertType(arrayTy));
  } else {
    result = aiir::LLVM::UndefOp::create(rewriter, loc, llvmTy);
  }

  // Iteratively lower each constant element of the array.
  if (auto arrayAttr = aiir::dyn_cast<aiir::ArrayAttr>(attr.getElts())) {
    for (auto [idx, elt] : llvm::enumerate(arrayAttr)) {
      aiir::DataLayout dataLayout(parentOp->getParentOfType<aiir::ModuleOp>());
      aiir::Value init = visit(elt);
      result =
          aiir::LLVM::InsertValueOp::create(rewriter, loc, result, init, idx);
    }
  } else if (auto strAttr = aiir::dyn_cast<aiir::StringAttr>(attr.getElts())) {
    // TODO(cir): this diverges from traditional lowering. Normally the string
    // would be a global constant that is memcopied.
    auto arrayTy = aiir::dyn_cast<cir::ArrayType>(strAttr.getType());
    assert(arrayTy && "String attribute must have an array type");
    aiir::Type eltTy = arrayTy.getElementType();
    for (auto [idx, elt] : llvm::enumerate(strAttr)) {
      auto init = aiir::LLVM::ConstantOp::create(
          rewriter, loc, converter->convertType(eltTy), elt);
      result =
          aiir::LLVM::InsertValueOp::create(rewriter, loc, result, init, idx);
    }
  } else {
    llvm_unreachable("unexpected ConstArrayAttr elements");
  }

  return result;
}

/// ConstRecord visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::ConstRecordAttr constRecord) {
  const aiir::Type llvmTy = converter->convertType(constRecord.getType());
  const aiir::Location loc = parentOp->getLoc();
  aiir::Value result = aiir::LLVM::UndefOp::create(rewriter, loc, llvmTy);

  // Iteratively lower each constant element of the record.
  for (auto [idx, elt] : llvm::enumerate(constRecord.getMembers())) {
    aiir::Value init = visit(elt);
    result =
        aiir::LLVM::InsertValueOp::create(rewriter, loc, result, init, idx);
  }

  return result;
}

/// ConstVectorAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::ConstVectorAttr attr) {
  const aiir::Type llvmTy = converter->convertType(attr.getType());
  const aiir::Location loc = parentOp->getLoc();

  SmallVector<aiir::Attribute> aiirValues;
  for (const aiir::Attribute elementAttr : attr.getElts()) {
    aiir::Attribute aiirAttr;
    if (auto intAttr = aiir::dyn_cast<cir::IntAttr>(elementAttr)) {
      aiirAttr = rewriter.getIntegerAttr(
          converter->convertType(intAttr.getType()), intAttr.getValue());
    } else if (auto floatAttr = aiir::dyn_cast<cir::FPAttr>(elementAttr)) {
      aiirAttr = rewriter.getFloatAttr(
          converter->convertType(floatAttr.getType()), floatAttr.getValue());
    } else {
      llvm_unreachable(
          "vector constant with an element that is neither an int nor a float");
    }
    aiirValues.push_back(aiirAttr);
  }

  return aiir::LLVM::ConstantOp::create(
      rewriter, loc, llvmTy,
      aiir::DenseElementsAttr::get(aiir::cast<aiir::ShapedType>(llvmTy),
                                   aiirValues));
}

// GlobalViewAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::GlobalViewAttr globalAttr) {
  auto moduleOp = parentOp->getParentOfType<aiir::ModuleOp>();
  aiir::DataLayout dataLayout(moduleOp);
  aiir::Type sourceType;
  assert(!cir::MissingFeatures::addressSpace());
  llvm::StringRef symName;
  aiir::Operation *sourceSymbol =
      aiir::SymbolTable::lookupSymbolIn(moduleOp, globalAttr.getSymbol());
  if (auto llvmSymbol = dyn_cast<aiir::LLVM::GlobalOp>(sourceSymbol)) {
    sourceType = llvmSymbol.getType();
    symName = llvmSymbol.getSymName();
  } else if (auto cirSymbol = dyn_cast<cir::GlobalOp>(sourceSymbol)) {
    sourceType =
        convertTypeForMemory(*converter, dataLayout, cirSymbol.getSymType());
    symName = cirSymbol.getSymName();
  } else if (auto llvmFun = dyn_cast<aiir::LLVM::LLVMFuncOp>(sourceSymbol)) {
    sourceType = llvmFun.getFunctionType();
    symName = llvmFun.getSymName();
  } else if (auto fun = dyn_cast<cir::FuncOp>(sourceSymbol)) {
    sourceType = converter->convertType(fun.getFunctionType());
    symName = fun.getSymName();
  } else if (auto alias = dyn_cast<aiir::LLVM::AliasOp>(sourceSymbol)) {
    sourceType = alias.getType();
    symName = alias.getSymName();
  } else {
    llvm_unreachable("Unexpected GlobalOp type");
  }

  aiir::Location loc = parentOp->getLoc();
  aiir::Value addrOp = aiir::LLVM::AddressOfOp::create(
      rewriter, loc, aiir::LLVM::LLVMPointerType::get(rewriter.getContext()),
      symName);

  if (globalAttr.getIndices()) {
    llvm::SmallVector<aiir::LLVM::GEPArg> indices;

    if (aiir::isa<aiir::LLVM::LLVMArrayType, aiir::LLVM::LLVMStructType>(
            sourceType))
      indices.push_back(0);

    for (aiir::Attribute idx : globalAttr.getIndices()) {
      auto intAttr = aiir::cast<aiir::IntegerAttr>(idx);
      indices.push_back(intAttr.getValue().getSExtValue());
    }
    aiir::Type resTy = addrOp.getType();
    aiir::Type eltTy = converter->convertType(sourceType);
    addrOp =
        aiir::LLVM::GEPOp::create(rewriter, loc, resTy, eltTy, addrOp, indices,
                                  aiir::LLVM::GEPNoWrapFlags::none);
  }

  // We can have a global view with an integer type in the case of method
  // pointers. With the Itanium ABI, the #cir.method attribute is lowered to a
  // #cir.global_view with a pointer-sized integer representing the address of
  // the method.
  if (auto intTy = aiir::dyn_cast<cir::IntType>(globalAttr.getType())) {
    aiir::Type llvmDstTy = converter->convertType(globalAttr.getType());
    return aiir::LLVM::PtrToIntOp::create(rewriter, parentOp->getLoc(),
                                          llvmDstTy, addrOp);
  }

  if (auto ptrTy = aiir::dyn_cast<cir::PointerType>(globalAttr.getType())) {
    aiir::Type llvmEltTy =
        convertTypeForMemory(*converter, dataLayout, ptrTy.getPointee());

    if (llvmEltTy == sourceType)
      return addrOp;

    aiir::Type llvmDstTy = converter->convertType(globalAttr.getType());
    return aiir::LLVM::BitcastOp::create(rewriter, parentOp->getLoc(),
                                         llvmDstTy, addrOp);
  }

  llvm_unreachable("Expecting pointer or integer type for GlobalViewAttr");
}

// TypeInfoAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::TypeInfoAttr typeInfoAttr) {
  aiir::Type llvmTy = converter->convertType(typeInfoAttr.getType());
  aiir::Location loc = parentOp->getLoc();
  aiir::Value result = aiir::LLVM::UndefOp::create(rewriter, loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(typeInfoAttr.getData())) {
    aiir::Value init = visit(elt);
    result =
        aiir::LLVM::InsertValueOp::create(rewriter, loc, result, init, idx);
  }

  return result;
}

/// UndefAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::UndefAttr undefAttr) {
  aiir::Location loc = parentOp->getLoc();
  return aiir::LLVM::UndefOp::create(
      rewriter, loc, converter->convertType(undefAttr.getType()));
}

// VTableAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::VTableAttr vtableArr) {
  aiir::Type llvmTy = converter->convertType(vtableArr.getType());
  aiir::Location loc = parentOp->getLoc();
  aiir::Value result = aiir::LLVM::UndefOp::create(rewriter, loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(vtableArr.getData())) {
    aiir::Value init = visit(elt);
    result =
        aiir::LLVM::InsertValueOp::create(rewriter, loc, result, init, idx);
  }

  return result;
}

/// ZeroAttr visitor.
aiir::Value CIRAttrToValue::visitCirAttr(cir::ZeroAttr attr) {
  aiir::Location loc = parentOp->getLoc();
  return aiir::LLVM::ZeroOp::create(rewriter, loc,
                                    converter->convertType(attr.getType()));
}

// This class handles rewriting initializer attributes for types that do not
// require region initialization.
class GlobalInitAttrRewriter {
public:
  GlobalInitAttrRewriter(aiir::Type type,
                         aiir::ConversionPatternRewriter &rewriter)
      : llvmType(type), rewriter(rewriter) {}

  aiir::Attribute visit(aiir::Attribute attr) {
    return llvm::TypeSwitch<aiir::Attribute, aiir::Attribute>(attr)
        .Case<cir::IntAttr, cir::FPAttr, cir::BoolAttr>(
            [&](auto attrT) { return visitCirAttr(attrT); })
        .Default([&](auto attrT) { return aiir::Attribute(); });
  }

  aiir::Attribute visitCirAttr(cir::IntAttr attr) {
    return rewriter.getIntegerAttr(llvmType, attr.getValue());
  }

  aiir::Attribute visitCirAttr(cir::FPAttr attr) {
    return rewriter.getFloatAttr(llvmType, attr.getValue());
  }

  aiir::Attribute visitCirAttr(cir::BoolAttr attr) {
    return rewriter.getBoolAttr(attr.getValue());
  }

private:
  aiir::Type llvmType;
  aiir::ConversionPatternRewriter &rewriter;
};

// This pass requires the CIR to be in a "flat" state. All blocks in each
// function must belong to the parent region. Once scopes and control flow
// are implemented in CIR, a pass will be run before this one to flatten
// the CIR and get it into the state that this pass requires.
struct ConvertCIRToLLVMPass
    : public aiir::PassWrapper<ConvertCIRToLLVMPass,
                               aiir::OperationPass<aiir::ModuleOp>> {
  void getDependentDialects(aiir::DialectRegistry &registry) const override {
    registry.insert<aiir::BuiltinDialect, aiir::DLTIDialect,
                    aiir::LLVM::LLVMDialect, aiir::func::FuncDialect>();
  }
  void runOnOperation() final;

  void processCIRAttrs(aiir::ModuleOp module);

  void resolveBlockAddressOp(LLVMBlockAddressInfo &blockInfoAddr);

  StringRef getDescription() const override {
    return "Convert the prepared CIR dialect module to LLVM dialect";
  }

  StringRef getArgument() const override { return "cir-flat-to-llvm"; }
};

aiir::LogicalResult CIRToLLVMACosOpLowering::matchAndRewrite(
    cir::ACosOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::ACosOp>(op, resTy,
                                                  adaptor.getOperands()[0]);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMASinOpLowering::matchAndRewrite(
    cir::ASinOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::ASinOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMIsFPClassOpLowering::matchAndRewrite(
    cir::IsFPClassOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value src = adaptor.getSrc();
  cir::FPClassTest flags = adaptor.getFlags();
  aiir::IntegerType retTy = rewriter.getI1Type();

  rewriter.replaceOpWithNewOp<aiir::LLVM::IsFPClass>(
      op, retTy, src, static_cast<uint32_t>(flags));
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMSignBitOpLowering::matchAndRewrite(
    cir::SignBitOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::isPPC_FP128Ty());

  aiir::DataLayout layout(op->getParentOfType<aiir::ModuleOp>());
  int width = layout.getTypeSizeInBits(op.getInput().getType());
  if (auto longDoubleType =
          aiir::dyn_cast<cir::LongDoubleType>(op.getInput().getType())) {
    if (aiir::isa<cir::FP80Type>(longDoubleType.getUnderlying())) {
      // If the underlying type of LongDouble is FP80Type,
      // DataLayout::getTypeSizeInBits returns 128.
      // See https://github.com/llvm/clangir/issues/1057.
      // Set the width to 80 manually.
      width = 80;
    }
  }
  aiir::Type intTy = aiir::IntegerType::get(rewriter.getContext(), width);
  auto bitcast = aiir::LLVM::BitcastOp::create(rewriter, op->getLoc(), intTy,
                                               adaptor.getInput());

  auto zero = aiir::LLVM::ConstantOp::create(rewriter, op->getLoc(), intTy, 0);
  auto cmpResult = aiir::LLVM::ICmpOp::create(rewriter, op.getLoc(),
                                              aiir::LLVM::ICmpPredicate::slt,
                                              bitcast.getResult(), zero);
  rewriter.replaceOp(op, cmpResult);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAssumeOpLowering::matchAndRewrite(
    cir::AssumeOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto cond = adaptor.getPredicate();
  rewriter.replaceOpWithNewOp<aiir::LLVM::AssumeOp>(op, cond);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAssumeAlignedOpLowering::matchAndRewrite(
    cir::AssumeAlignedOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  SmallVector<aiir::Value, 3> opBundleArgs{adaptor.getPointer()};

  auto alignment = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                                  adaptor.getAlignmentAttr());
  opBundleArgs.push_back(alignment);

  if (aiir::Value offset = adaptor.getOffset())
    opBundleArgs.push_back(offset);

  auto cond = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                             rewriter.getI1Type(), 1);
  aiir::LLVM::AssumeOp::create(rewriter, op.getLoc(), cond, "align",
                               opBundleArgs);

  // The llvm.assume operation does not have a result, so we need to replace
  // all uses of this cir.assume_aligned operation with the input ptr itself.
  rewriter.replaceOp(op, adaptor.getPointer());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAssumeSepStorageOpLowering::matchAndRewrite(
    cir::AssumeSepStorageOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto cond = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                             rewriter.getI1Type(), 1);
  rewriter.replaceOpWithNewOp<aiir::LLVM::AssumeOp>(
      op, cond, aiir::LLVM::AssumeSeparateStorageTag{}, adaptor.getPtr1(),
      adaptor.getPtr2());
  return aiir::success();
}

static aiir::LLVM::AtomicOrdering
getLLVMMemOrder(std::optional<cir::MemOrder> memorder) {
  if (!memorder)
    return aiir::LLVM::AtomicOrdering::not_atomic;
  switch (*memorder) {
  case cir::MemOrder::Relaxed:
    return aiir::LLVM::AtomicOrdering::monotonic;
  case cir::MemOrder::Consume:
  case cir::MemOrder::Acquire:
    return aiir::LLVM::AtomicOrdering::acquire;
  case cir::MemOrder::Release:
    return aiir::LLVM::AtomicOrdering::release;
  case cir::MemOrder::AcquireRelease:
    return aiir::LLVM::AtomicOrdering::acq_rel;
  case cir::MemOrder::SequentiallyConsistent:
    return aiir::LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown memory order");
}

static llvm::StringRef getLLVMSyncScope(cir::SyncScopeKind syncScope) {
  return syncScope == cir::SyncScopeKind::SingleThread ? "singlethread" : "";
}

static std::optional<llvm::StringRef>
getLLVMSyncScope(std::optional<cir::SyncScopeKind> syncScope) {
  if (syncScope.has_value())
    return getLLVMSyncScope(*syncScope);
  return std::nullopt;
}

aiir::LogicalResult CIRToLLVMAtomicCmpXchgOpLowering::matchAndRewrite(
    cir::AtomicCmpXchgOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value expected = adaptor.getExpected();
  aiir::Value desired = adaptor.getDesired();

  auto cmpxchg = aiir::LLVM::AtomicCmpXchgOp::create(
      rewriter, op.getLoc(), adaptor.getPtr(), expected, desired,
      getLLVMMemOrder(adaptor.getSuccOrder()),
      getLLVMMemOrder(adaptor.getFailOrder()),
      getLLVMSyncScope(op.getSyncScope()));

  cmpxchg.setAlignment(adaptor.getAlignment());
  cmpxchg.setWeak(adaptor.getWeak());
  cmpxchg.setVolatile_(adaptor.getIsVolatile());

  // Check result and apply stores accordingly.
  auto old = aiir::LLVM::ExtractValueOp::create(rewriter, op.getLoc(),
                                                cmpxchg.getResult(), 0);
  auto cmp = aiir::LLVM::ExtractValueOp::create(rewriter, op.getLoc(),
                                                cmpxchg.getResult(), 1);

  rewriter.replaceOp(op, {old, cmp});
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAtomicXchgOpLowering::matchAndRewrite(
    cir::AtomicXchgOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::atomicSyncScopeID());
  aiir::LLVM::AtomicOrdering llvmOrder = getLLVMMemOrder(adaptor.getMemOrder());
  llvm::StringRef llvmSyncScope = getLLVMSyncScope(adaptor.getSyncScope());
  rewriter.replaceOpWithNewOp<aiir::LLVM::AtomicRMWOp>(
      op, aiir::LLVM::AtomicBinOp::xchg, adaptor.getPtr(), adaptor.getVal(),
      llvmOrder, llvmSyncScope);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAtomicTestAndSetOpLowering::matchAndRewrite(
    cir::AtomicTestAndSetOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::atomicSyncScopeID());

  aiir::LLVM::AtomicOrdering llvmOrder = getLLVMMemOrder(op.getMemOrder());

  auto one = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                            rewriter.getI8Type(), 1);
  auto rmw = aiir::LLVM::AtomicRMWOp::create(
      rewriter, op.getLoc(), aiir::LLVM::AtomicBinOp::xchg, adaptor.getPtr(),
      one, llvmOrder, /*syncscope=*/llvm::StringRef(),
      adaptor.getAlignment().value_or(0), op.getIsVolatile());

  auto zero = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                             rewriter.getI8Type(), 0);
  auto cmp = aiir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(), aiir::LLVM::ICmpPredicate::ne, rmw, zero);

  rewriter.replaceOp(op, cmp);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAtomicClearOpLowering::matchAndRewrite(
    cir::AtomicClearOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::atomicSyncScopeID());

  aiir::LLVM::AtomicOrdering llvmOrder = getLLVMMemOrder(op.getMemOrder());
  auto zero = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                             rewriter.getI8Type(), 0);
  auto store = aiir::LLVM::StoreOp::create(
      rewriter, op.getLoc(), zero, adaptor.getPtr(),
      adaptor.getAlignment().value_or(0), op.getIsVolatile(),
      /*isNonTemporal=*/false, /*isInvariantGroup=*/false, llvmOrder);

  rewriter.replaceOp(op, store);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAtomicFenceOpLowering::matchAndRewrite(
    cir::AtomicFenceOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::LLVM::AtomicOrdering llvmOrder = getLLVMMemOrder(adaptor.getOrdering());

  auto fence = aiir::LLVM::FenceOp::create(rewriter, op.getLoc(), llvmOrder);
  fence.setSyncscope(getLLVMSyncScope(adaptor.getSyncscope()));

  rewriter.replaceOp(op, fence);

  return aiir::success();
}

static aiir::LLVM::AtomicBinOp
getLLVMAtomicBinOp(cir::AtomicFetchKind k, bool isInt, bool isSignedInt) {
  switch (k) {
  case cir::AtomicFetchKind::Add:
    return isInt ? aiir::LLVM::AtomicBinOp::add : aiir::LLVM::AtomicBinOp::fadd;
  case cir::AtomicFetchKind::Sub:
    return isInt ? aiir::LLVM::AtomicBinOp::sub : aiir::LLVM::AtomicBinOp::fsub;
  case cir::AtomicFetchKind::And:
    return aiir::LLVM::AtomicBinOp::_and;
  case cir::AtomicFetchKind::Xor:
    return aiir::LLVM::AtomicBinOp::_xor;
  case cir::AtomicFetchKind::Or:
    return aiir::LLVM::AtomicBinOp::_or;
  case cir::AtomicFetchKind::Nand:
    return aiir::LLVM::AtomicBinOp::nand;
  case cir::AtomicFetchKind::Max: {
    if (!isInt)
      return aiir::LLVM::AtomicBinOp::fmax;
    return isSignedInt ? aiir::LLVM::AtomicBinOp::max
                       : aiir::LLVM::AtomicBinOp::umax;
  }
  case cir::AtomicFetchKind::Min: {
    if (!isInt)
      return aiir::LLVM::AtomicBinOp::fmin;
    return isSignedInt ? aiir::LLVM::AtomicBinOp::min
                       : aiir::LLVM::AtomicBinOp::umin;
  }
  case cir::AtomicFetchKind::UIncWrap:
    return aiir::LLVM::AtomicBinOp::uinc_wrap;
  case cir::AtomicFetchKind::UDecWrap:
    return aiir::LLVM::AtomicBinOp::udec_wrap;
  }
  llvm_unreachable("Unknown atomic fetch opcode");
}

static llvm::StringLiteral getLLVMBinopForPostAtomic(cir::AtomicFetchKind k,
                                                     bool isInt) {
  switch (k) {
  case cir::AtomicFetchKind::Add:
    return isInt ? aiir::LLVM::AddOp::getOperationName()
                 : aiir::LLVM::FAddOp::getOperationName();
  case cir::AtomicFetchKind::Sub:
    return isInt ? aiir::LLVM::SubOp::getOperationName()
                 : aiir::LLVM::FSubOp::getOperationName();
  case cir::AtomicFetchKind::And:
    return aiir::LLVM::AndOp::getOperationName();
  case cir::AtomicFetchKind::Xor:
    return aiir::LLVM::XOrOp::getOperationName();
  case cir::AtomicFetchKind::Or:
    return aiir::LLVM::OrOp::getOperationName();
  case cir::AtomicFetchKind::Nand:
    // There's no nand binop in LLVM, this is later fixed with a not.
    return aiir::LLVM::AndOp::getOperationName();
  case cir::AtomicFetchKind::Max:
  case cir::AtomicFetchKind::Min:
    llvm_unreachable("handled in buildMinMaxPostOp");
  case cir::AtomicFetchKind::UIncWrap:
  case cir::AtomicFetchKind::UDecWrap:
    llvm_unreachable("uinc_wrap and udec_wrap are always fetch_first");
  }
  llvm_unreachable("Unknown atomic fetch opcode");
}

aiir::Value CIRToLLVMAtomicFetchOpLowering::buildPostOp(
    cir::AtomicFetchOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter, aiir::Value rmwVal,
    bool isInt) const {
  SmallVector<aiir::Value> atomicOperands = {rmwVal, adaptor.getVal()};
  SmallVector<aiir::Type> atomicResTys = {rmwVal.getType()};
  return rewriter
      .create(op.getLoc(),
              rewriter.getStringAttr(
                  getLLVMBinopForPostAtomic(op.getBinop(), isInt)),
              atomicOperands, atomicResTys, {})
      ->getResult(0);
}

aiir::Value CIRToLLVMAtomicFetchOpLowering::buildMinMaxPostOp(
    cir::AtomicFetchOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter, aiir::Value rmwVal, bool isInt,
    bool isSigned) const {
  aiir::Location loc = op.getLoc();

  if (!isInt) {
    if (op.getBinop() == cir::AtomicFetchKind::Max)
      return aiir::LLVM::MaxNumOp::create(rewriter, loc, rmwVal,
                                          adaptor.getVal());
    return aiir::LLVM::MinNumOp::create(rewriter, loc, rmwVal,
                                        adaptor.getVal());
  }

  aiir::LLVM::ICmpPredicate pred;
  if (op.getBinop() == cir::AtomicFetchKind::Max) {
    pred = isSigned ? aiir::LLVM::ICmpPredicate::sgt
                    : aiir::LLVM::ICmpPredicate::ugt;
  } else { // Min
    pred = isSigned ? aiir::LLVM::ICmpPredicate::slt
                    : aiir::LLVM::ICmpPredicate::ult;
  }
  aiir::Value cmp = aiir::LLVM::ICmpOp::create(
      rewriter, loc,
      aiir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(), pred), rmwVal,
      adaptor.getVal());
  return aiir::LLVM::SelectOp::create(rewriter, loc, cmp, rmwVal,
                                      adaptor.getVal());
}

aiir::LogicalResult CIRToLLVMAtomicFetchOpLowering::matchAndRewrite(
    cir::AtomicFetchOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  bool isInt = false;
  bool isSignedInt = false;
  if (auto intTy = aiir::dyn_cast<cir::IntType>(op.getVal().getType())) {
    isInt = true;
    isSignedInt = intTy.isSigned();
  } else if (aiir::isa<cir::SingleType, cir::DoubleType>(
                 op.getVal().getType())) {
    isInt = false;
  } else {
    return op.emitError() << "Unsupported type: " << op.getVal().getType();
  }

  aiir::LLVM::AtomicOrdering llvmOrder = getLLVMMemOrder(op.getMemOrder());
  llvm::StringRef llvmSyncScope = getLLVMSyncScope(op.getSyncScope());
  aiir::LLVM::AtomicBinOp llvmBinOp =
      getLLVMAtomicBinOp(op.getBinop(), isInt, isSignedInt);
  auto rmwVal = aiir::LLVM::AtomicRMWOp::create(
      rewriter, op.getLoc(), llvmBinOp, adaptor.getPtr(), adaptor.getVal(),
      llvmOrder, llvmSyncScope);

  aiir::Value result = rmwVal.getResult();
  if (!op.getFetchFirst()) {
    if (op.getBinop() == cir::AtomicFetchKind::Max ||
        op.getBinop() == cir::AtomicFetchKind::Min)
      result = buildMinMaxPostOp(op, adaptor, rewriter, rmwVal.getRes(), isInt,
                                 isSignedInt);
    else
      result = buildPostOp(op, adaptor, rewriter, rmwVal.getRes(), isInt);

    // Compensate lack of nand binop in LLVM IR.
    if (op.getBinop() == cir::AtomicFetchKind::Nand) {
      auto negOne = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                                   result.getType(), -1);
      result = aiir::LLVM::XOrOp::create(rewriter, op.getLoc(), result, negOne);
    }
  }

  rewriter.replaceOp(op, result);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMBitClrsbOpLowering::matchAndRewrite(
    cir::BitClrsbOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto zero = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                             adaptor.getInput().getType(), 0);
  auto isNeg = aiir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(),
      aiir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                         aiir::LLVM::ICmpPredicate::slt),
      adaptor.getInput(), zero);

  auto negOne = aiir::LLVM::ConstantOp::create(
      rewriter, op.getLoc(), adaptor.getInput().getType(), -1);
  auto flipped = aiir::LLVM::XOrOp::create(rewriter, op.getLoc(),
                                           adaptor.getInput(), negOne);

  auto select = aiir::LLVM::SelectOp::create(rewriter, op.getLoc(), isNeg,
                                             flipped, adaptor.getInput());

  auto resTy = getTypeConverter()->convertType(op.getType());
  auto clz = aiir::LLVM::CountLeadingZerosOp::create(
      rewriter, op.getLoc(), resTy, select, /*is_zero_poison=*/false);

  auto one = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(), resTy, 1);
  auto res = aiir::LLVM::SubOp::create(rewriter, op.getLoc(), clz, one,
                                       aiir::LLVM::IntegerOverflowFlags::nuw);
  rewriter.replaceOp(op, res);

  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMBitClzOpLowering::matchAndRewrite(
    cir::BitClzOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp = aiir::LLVM::CountLeadingZerosOp::create(
      rewriter, op.getLoc(), resTy, adaptor.getInput(), op.getPoisonZero());
  rewriter.replaceOp(op, llvmOp);
  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMBitCtzOpLowering::matchAndRewrite(
    cir::BitCtzOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp = aiir::LLVM::CountTrailingZerosOp::create(
      rewriter, op.getLoc(), resTy, adaptor.getInput(), op.getPoisonZero());
  rewriter.replaceOp(op, llvmOp);
  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMBitFfsOpLowering::matchAndRewrite(
    cir::BitFfsOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto ctz = aiir::LLVM::CountTrailingZerosOp::create(rewriter, op.getLoc(),
                                                      resTy, adaptor.getInput(),
                                                      /*is_zero_poison=*/true);

  auto one = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(), resTy, 1);
  auto ctzAddOne = aiir::LLVM::AddOp::create(rewriter, op.getLoc(), ctz, one);

  auto zeroInputTy = aiir::LLVM::ConstantOp::create(
      rewriter, op.getLoc(), adaptor.getInput().getType(), 0);
  auto isZero = aiir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(),
      aiir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                         aiir::LLVM::ICmpPredicate::eq),
      adaptor.getInput(), zeroInputTy);

  auto zero = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(), resTy, 0);
  auto res = aiir::LLVM::SelectOp::create(rewriter, op.getLoc(), isZero, zero,
                                          ctzAddOne);
  rewriter.replaceOp(op, res);

  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMBitParityOpLowering::matchAndRewrite(
    cir::BitParityOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto popcnt = aiir::LLVM::CtPopOp::create(rewriter, op.getLoc(), resTy,
                                            adaptor.getInput());

  auto one = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(), resTy, 1);
  auto popcntMod2 =
      aiir::LLVM::AndOp::create(rewriter, op.getLoc(), popcnt, one);
  rewriter.replaceOp(op, popcntMod2);

  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMBitPopcountOpLowering::matchAndRewrite(
    cir::BitPopcountOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp = aiir::LLVM::CtPopOp::create(rewriter, op.getLoc(), resTy,
                                            adaptor.getInput());
  rewriter.replaceOp(op, llvmOp);
  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMBitReverseOpLowering::matchAndRewrite(
    cir::BitReverseOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::BitReverseOp>(op, adaptor.getInput());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMBrCondOpLowering::matchAndRewrite(
    cir::BrCondOp brOp, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // When ZExtOp is implemented, we'll need to check if the condition is a
  // ZExtOp and if so, delete it if it has a single use.
  assert(!cir::MissingFeatures::zextOp());

  aiir::Value i1Condition = adaptor.getCond();

  rewriter.replaceOpWithNewOp<aiir::LLVM::CondBrOp>(
      brOp, i1Condition, brOp.getDestTrue(), adaptor.getDestOperandsTrue(),
      brOp.getDestFalse(), adaptor.getDestOperandsFalse());

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMByteSwapOpLowering::matchAndRewrite(
    cir::ByteSwapOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::ByteSwapOp>(op, adaptor.getInput());
  return aiir::LogicalResult::success();
}

aiir::Type CIRToLLVMCastOpLowering::convertTy(aiir::Type ty) const {
  return getTypeConverter()->convertType(ty);
}

aiir::LogicalResult CIRToLLVMCastOpLowering::matchAndRewrite(
    cir::CastOp castOp, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // For arithmetic conversions, LLVM IR uses the same instruction to convert
  // both individual scalars and entire vectors. This lowering pass handles
  // both situations.

  switch (castOp.getKind()) {
  case cir::CastKind::array_to_ptrdecay: {
    const auto ptrTy = aiir::cast<cir::PointerType>(castOp.getType());
    aiir::Value sourceValue = adaptor.getSrc();
    aiir::Type targetType = convertTy(ptrTy);
    aiir::Type elementTy = convertTypeForMemory(*getTypeConverter(), dataLayout,
                                                ptrTy.getPointee());
    llvm::SmallVector<aiir::LLVM::GEPArg> offset{0};
    rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
        castOp, targetType, elementTy, sourceValue, offset);
    break;
  }
  case cir::CastKind::int_to_bool: {
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Value zeroInt = aiir::LLVM::ConstantOp::create(
        rewriter, castOp.getLoc(), llvmSrcVal.getType(), 0);
    rewriter.replaceOpWithNewOp<aiir::LLVM::ICmpOp>(
        castOp, aiir::LLVM::ICmpPredicate::ne, llvmSrcVal, zeroInt);
    break;
  }
  case cir::CastKind::integral: {
    aiir::Type srcType = castOp.getSrc().getType();
    aiir::Type dstType = castOp.getType();
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstType = getTypeConverter()->convertType(dstType);
    cir::IntType srcIntType =
        aiir::cast<cir::IntType>(elementTypeIfVector(srcType));
    cir::IntType dstIntType =
        aiir::cast<cir::IntType>(elementTypeIfVector(dstType));
    rewriter.replaceOp(castOp, getLLVMIntCast(rewriter, llvmSrcVal, llvmDstType,
                                              srcIntType.isUnsigned(),
                                              srcIntType.getWidth(),
                                              dstIntType.getWidth()));
    break;
  }
  case cir::CastKind::floating: {
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(castOp.getType());

    aiir::Type srcTy = elementTypeIfVector(castOp.getSrc().getType());
    aiir::Type dstTy = elementTypeIfVector(castOp.getType());

    if (!aiir::isa<cir::FPTypeInterface>(dstTy) ||
        !aiir::isa<cir::FPTypeInterface>(srcTy))
      return castOp.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

    auto getFloatWidth = [](aiir::Type ty) -> unsigned {
      return aiir::cast<cir::FPTypeInterface>(ty).getWidth();
    };

    if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
      rewriter.replaceOpWithNewOp<aiir::LLVM::FPTruncOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<aiir::LLVM::FPExtOp>(castOp, llvmDstTy,
                                                       llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::int_to_ptr: {
    auto dstTy = aiir::cast<cir::PointerType>(castOp.getType());
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<aiir::LLVM::IntToPtrOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::ptr_to_int: {
    auto dstTy = aiir::cast<cir::IntType>(castOp.getType());
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<aiir::LLVM::PtrToIntOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::float_to_bool: {
    aiir::Value llvmSrcVal = adaptor.getSrc();
    auto kind = aiir::LLVM::FCmpPredicate::une;

    // Check if float is not equal to zero.
    auto zeroFloat = aiir::LLVM::ConstantOp::create(
        rewriter, castOp.getLoc(), llvmSrcVal.getType(),
        aiir::FloatAttr::get(llvmSrcVal.getType(), 0.0));

    // Extend comparison result to either bool (C++) or int (C).
    rewriter.replaceOpWithNewOp<aiir::LLVM::FCmpOp>(castOp, kind, llvmSrcVal,
                                                    zeroFloat);

    return aiir::success();
  }
  case cir::CastKind::bool_to_int: {
    auto dstTy = aiir::cast<cir::IntType>(castOp.getType());
    aiir::Value llvmSrcVal = adaptor.getSrc();
    auto llvmSrcTy = aiir::cast<aiir::IntegerType>(llvmSrcVal.getType());
    auto llvmDstTy =
        aiir::cast<aiir::IntegerType>(getTypeConverter()->convertType(dstTy));

    if (llvmSrcTy.getWidth() == llvmDstTy.getWidth())
      rewriter.replaceOpWithNewOp<aiir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<aiir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::bool_to_float: {
    aiir::Type dstTy = castOp.getType();
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<aiir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::int_to_float: {
    aiir::Type dstTy = castOp.getType();
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (aiir::cast<cir::IntType>(elementTypeIfVector(castOp.getSrc().getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<aiir::LLVM::SIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<aiir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::float_to_int: {
    aiir::Type dstTy = castOp.getType();
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (aiir::cast<cir::IntType>(elementTypeIfVector(castOp.getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<aiir::LLVM::FPToSIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<aiir::LLVM::FPToUIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::bitcast: {
    aiir::Type dstTy = castOp.getType();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);

    assert(!MissingFeatures::cxxABI());
    assert(!MissingFeatures::dataMemberType());

    aiir::Value llvmSrcVal = adaptor.getSrc();
    rewriter.replaceOpWithNewOp<aiir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                       llvmSrcVal);
    return aiir::success();
  }
  case cir::CastKind::ptr_to_bool: {
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Value zeroPtr = aiir::LLVM::ZeroOp::create(rewriter, castOp.getLoc(),
                                                     llvmSrcVal.getType());
    rewriter.replaceOpWithNewOp<aiir::LLVM::ICmpOp>(
        castOp, aiir::LLVM::ICmpPredicate::ne, llvmSrcVal, zeroPtr);
    break;
  }
  case cir::CastKind::address_space: {
    aiir::Type dstTy = castOp.getType();
    aiir::Value llvmSrcVal = adaptor.getSrc();
    aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<aiir::LLVM::AddrSpaceCastOp>(castOp, llvmDstTy,
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

  return aiir::success();
}

static aiir::Value convertToIndexTy(aiir::ConversionPatternRewriter &rewriter,
                                    aiir::ModuleOp mod, aiir::Value index,
                                    aiir::Type baseTy, cir::IntType strideTy) {
  aiir::Operation *indexOp = index.getDefiningOp();
  if (!indexOp)
    return index;

  auto indexType = aiir::cast<aiir::IntegerType>(index.getType());
  aiir::DataLayout llvmLayout(mod);
  std::optional<uint64_t> layoutWidth = llvmLayout.getTypeIndexBitwidth(baseTy);

  // If there is no change in width, don't do anything.
  if (!layoutWidth || *layoutWidth == indexType.getWidth())
    return index;

  // If the index comes from a subtraction, make sure the extension happens
  // before it. To achieve that, look at unary minus, which already got
  // lowered to "sub 0, x".
  auto sub = dyn_cast<aiir::LLVM::SubOp>(indexOp);
  bool rewriteSub = false;
  if (sub) {
    if (auto lhsConst =
            dyn_cast<aiir::LLVM::ConstantOp>(sub.getLhs().getDefiningOp())) {
      auto lhsConstInt = aiir::dyn_cast<aiir::IntegerAttr>(lhsConst.getValue());
      if (lhsConstInt && lhsConstInt.getValue() == 0) {
        index = sub.getRhs();
        rewriteSub = true;
      }
    }
  }

  auto llvmDstType = rewriter.getIntegerType(*layoutWidth);
  bool isUnsigned = strideTy && strideTy.isUnsigned();
  index = getLLVMIntCast(rewriter, index, llvmDstType, isUnsigned,
                         indexType.getWidth(), *layoutWidth);

  if (rewriteSub) {
    index = aiir::LLVM::SubOp::create(
        rewriter, index.getLoc(),
        aiir::LLVM::ConstantOp::create(rewriter, index.getLoc(),
                                       index.getType(), 0),
        index);
    // TODO: ensure sub is trivially dead now.
    rewriter.eraseOp(sub);
  }

  return index;
}

aiir::LogicalResult CIRToLLVMPtrStrideOpLowering::matchAndRewrite(
    cir::PtrStrideOp ptrStrideOp, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {

  const aiir::TypeConverter *tc = getTypeConverter();
  const aiir::Type resultTy = tc->convertType(ptrStrideOp.getType());

  aiir::Type elementTy =
      convertTypeForMemory(*tc, dataLayout, ptrStrideOp.getElementType());

  // void and function types doesn't really have a layout to use in GEPs,
  // make it i8 instead.
  if (aiir::isa<aiir::LLVM::LLVMVoidType>(elementTy) ||
      aiir::isa<aiir::LLVM::LLVMFunctionType>(elementTy))
    elementTy = aiir::IntegerType::get(elementTy.getContext(), 8,
                                       aiir::IntegerType::Signless);
  // Zero-extend, sign-extend or trunc the pointer value.
  aiir::Value index = adaptor.getStride();
  index = convertToIndexTy(
      rewriter, ptrStrideOp->getParentOfType<aiir::ModuleOp>(), index,
      adaptor.getBase().getType(),
      dyn_cast<cir::IntType>(ptrStrideOp.getOperand(1).getType()));

  rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
      ptrStrideOp, resultTy, elementTy, adaptor.getBase(), index);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMGetElementOpLowering::matchAndRewrite(
    cir::GetElementOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  if (auto arrayTy =
          aiir::dyn_cast<cir::ArrayType>(op.getBaseType().getPointee())) {
    const aiir::TypeConverter *converter = getTypeConverter();
    const aiir::Type llArrayTy = converter->convertType(arrayTy);
    const aiir::Type llResultTy = converter->convertType(op.getType());
    aiir::Type elementTy =
        convertTypeForMemory(*converter, dataLayout, op.getElementType());

    // void and function types don't really have a layout to use in GEPs,
    // make it i8 instead.
    if (aiir::isa<aiir::LLVM::LLVMVoidType>(elementTy) ||
        aiir::isa<aiir::LLVM::LLVMFunctionType>(elementTy))
      elementTy = rewriter.getIntegerType(8);

    aiir::Value index = adaptor.getIndex();
    index =
        convertToIndexTy(rewriter, op->getParentOfType<aiir::ModuleOp>(), index,
                         adaptor.getBase().getType(),
                         dyn_cast<cir::IntType>(op.getOperand(1).getType()));

    // Since the base address is a pointer to an aggregate, the first
    // offset is always zero. The second offset tell us which member it
    // will access.
    std::array<aiir::LLVM::GEPArg, 2> offset{0, index};
    rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(op, llResultTy, llArrayTy,
                                                   adaptor.getBase(), offset);
    return aiir::success();
  }

  op.emitError() << "NYI: GetElementOp lowering to LLVM for non-array";
  return aiir::failure();
}

aiir::LogicalResult CIRToLLVMBaseClassAddrOpLowering::matchAndRewrite(
    cir::BaseClassAddrOp baseClassOp, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type resultType =
      getTypeConverter()->convertType(baseClassOp.getType());
  aiir::Value derivedAddr = adaptor.getDerivedAddr();
  llvm::SmallVector<aiir::LLVM::GEPArg, 1> offset = {
      adaptor.getOffset().getZExtValue()};
  aiir::Type byteType = aiir::IntegerType::get(resultType.getContext(), 8,
                                               aiir::IntegerType::Signless);
  if (adaptor.getOffset().getZExtValue() == 0) {
    rewriter.replaceOpWithNewOp<aiir::LLVM::BitcastOp>(
        baseClassOp, resultType, adaptor.getDerivedAddr());
    return aiir::success();
  }

  if (baseClassOp.getAssumeNotNull()) {
    rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
        baseClassOp, resultType, byteType, derivedAddr, offset);
  } else {
    auto loc = baseClassOp.getLoc();
    aiir::Value isNull = aiir::LLVM::ICmpOp::create(
        rewriter, loc, aiir::LLVM::ICmpPredicate::eq, derivedAddr,
        aiir::LLVM::ZeroOp::create(rewriter, loc, derivedAddr.getType()));
    aiir::Value adjusted = aiir::LLVM::GEPOp::create(
        rewriter, loc, resultType, byteType, derivedAddr, offset);
    rewriter.replaceOpWithNewOp<aiir::LLVM::SelectOp>(baseClassOp, isNull,
                                                      derivedAddr, adjusted);
  }
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMDerivedClassAddrOpLowering::matchAndRewrite(
    cir::DerivedClassAddrOp derivedClassOp, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type resultType =
      getTypeConverter()->convertType(derivedClassOp.getType());
  aiir::Value baseAddr = adaptor.getBaseAddr();
  // The offset is set in the operation as an unsigned value, but it must be
  // applied as a negative offset.
  int64_t offsetVal = -(adaptor.getOffset().getZExtValue());
  if (offsetVal == 0) {
    // If the offset is zero, we can just return the base address,
    rewriter.replaceOp(derivedClassOp, baseAddr);
    return aiir::success();
  }
  llvm::SmallVector<aiir::LLVM::GEPArg, 1> offset = {offsetVal};
  aiir::Type byteType = aiir::IntegerType::get(resultType.getContext(), 8,
                                               aiir::IntegerType::Signless);
  if (derivedClassOp.getAssumeNotNull()) {
    rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
        derivedClassOp, resultType, byteType, baseAddr, offset,
        aiir::LLVM::GEPNoWrapFlags::inbounds);
  } else {
    aiir::Location loc = derivedClassOp.getLoc();
    aiir::Value isNull = aiir::LLVM::ICmpOp::create(
        rewriter, loc, aiir::LLVM::ICmpPredicate::eq, baseAddr,
        aiir::LLVM::ZeroOp::create(rewriter, loc, baseAddr.getType()));
    aiir::Value adjusted =
        aiir::LLVM::GEPOp::create(rewriter, loc, resultType, byteType, baseAddr,
                                  offset, aiir::LLVM::GEPNoWrapFlags::inbounds);
    rewriter.replaceOpWithNewOp<aiir::LLVM::SelectOp>(derivedClassOp, isNull,
                                                      baseAddr, adjusted);
  }
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMATanOpLowering::matchAndRewrite(
    cir::ATanOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::ATanOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMCeilOpLowering::matchAndRewrite(
    cir::CeilOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::FCeilOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMCopysignOpLowering::matchAndRewrite(
    cir::CopysignOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::CopySignOp>(
      op, resTy, adaptor.getLhs(), adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFMaxNumOpLowering::matchAndRewrite(
    cir::FMaxNumOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::MaxNumOp>(
      op, resTy, adaptor.getLhs(), adaptor.getRhs(),
      aiir::LLVM::FastmathFlags::nsz);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFMinNumOpLowering::matchAndRewrite(
    cir::FMinNumOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::MinNumOp>(
      op, resTy, adaptor.getLhs(), adaptor.getRhs(),
      aiir::LLVM::FastmathFlags::nsz);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFMaximumOpLowering::matchAndRewrite(
    cir::FMaximumOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::MaximumOp>(
      op, resTy, adaptor.getLhs(), adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFMinimumOpLowering::matchAndRewrite(
    cir::FMinimumOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::MinimumOp>(
      op, resTy, adaptor.getLhs(), adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFModOpLowering::matchAndRewrite(
    cir::FModOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::FRemOp>(op, resTy, adaptor.getLhs(),
                                                  adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMPowOpLowering::matchAndRewrite(
    cir::PowOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::PowOp>(op, resTy, adaptor.getLhs(),
                                                 adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMATan2OpLowering::matchAndRewrite(
    cir::ATan2Op op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::ATan2Op>(op, resTy, adaptor.getLhs(),
                                                   adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLroundOpLowering::matchAndRewrite(
    cir::LroundOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::LroundOp>(op, resTy,
                                                    adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLlroundOpLowering::matchAndRewrite(
    cir::LlroundOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::LlroundOp>(op, resTy,
                                                     adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLrintOpLowering::matchAndRewrite(
    cir::LrintOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::LrintOp>(op, resTy, adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLlrintOpLowering::matchAndRewrite(
    cir::LlrintOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::LlrintOp>(op, resTy,
                                                    adaptor.getSrc());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAllocaOpLowering::matchAndRewrite(
    cir::AllocaOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value size =
      op.isDynamic()
          ? adaptor.getDynAllocSize()
          : aiir::LLVM::ConstantOp::create(
                rewriter, op.getLoc(),
                typeConverter->convertType(rewriter.getIndexType()), 1);
  aiir::Type elementTy =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getAllocaType());
  aiir::Type resultTy =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getType());

  assert(!cir::MissingFeatures::addressSpace());
  assert(!cir::MissingFeatures::opAllocaAnnotations());

  rewriter.replaceOpWithNewOp<aiir::LLVM::AllocaOp>(op, resultTy, elementTy,
                                                    size, op.getAlignment());

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMReturnOpLowering::matchAndRewrite(
    cir::ReturnOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::ReturnOp>(op, adaptor.getOperands());
  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMRotateOpLowering::matchAndRewrite(
    cir::RotateOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // Note that LLVM intrinsic calls to @llvm.fsh{r,l}.i* have the same type as
  // the operand.
  aiir::Value input = adaptor.getInput();
  if (op.isRotateLeft())
    rewriter.replaceOpWithNewOp<aiir::LLVM::FshlOp>(op, input, input,
                                                    adaptor.getAmount());
  else
    rewriter.replaceOpWithNewOp<aiir::LLVM::FshrOp>(op, input, input,
                                                    adaptor.getAmount());
  return aiir::LogicalResult::success();
}

static void lowerCallAttributes(cir::CIRCallOpInterface op,
                                SmallVectorImpl<aiir::NamedAttribute> &result) {
  for (aiir::NamedAttribute attr : op->getAttrs()) {
    assert(!cir::MissingFeatures::opFuncCallingConv());
    if (attr.getName() == CIRDialect::getCalleeAttrName() ||
        attr.getName() == CIRDialect::getSideEffectAttrName() ||
        attr.getName() == CIRDialect::getNoThrowAttrName() ||
        attr.getName() == CIRDialect::getNoUnwindAttrName() ||
        attr.getName() == CIRDialect::getNoReturnAttrName())
      continue;

    assert(!cir::MissingFeatures::opFuncExtraAttrs());
    result.push_back(attr);
  }
}

static aiir::LogicalResult
rewriteCallOrInvoke(aiir::Operation *op, aiir::ValueRange callOperands,
                    aiir::ConversionPatternRewriter &rewriter,
                    const aiir::TypeConverter *converter,
                    aiir::FlatSymbolRefAttr calleeAttr,
                    aiir::Block *continueBlock = nullptr,
                    aiir::Block *landingPadBlock = nullptr) {
  llvm::SmallVector<aiir::Type, 8> llvmResults;
  aiir::ValueTypeRange<aiir::ResultRange> cirResults = op->getResultTypes();
  auto call = cast<cir::CIRCallOpInterface>(op);

  if (converter->convertTypes(cirResults, llvmResults).failed())
    return aiir::failure();

  assert(!cir::MissingFeatures::opCallCallConv());

  aiir::LLVM::MemoryEffectsAttr memoryEffects;
  bool noUnwind = false;
  bool willReturn = false;
  bool noReturn = false;
  convertSideEffectForCall(op, call.getNothrow(), call.getSideEffect(),
                           memoryEffects, noUnwind, willReturn, noReturn);

  SmallVector<aiir::NamedAttribute, 4> attributes;
  lowerCallAttributes(call, attributes);

  aiir::LLVM::LLVMFunctionType llvmFnTy;

  // Temporary to handle the case where we need to prepend an operand if the
  // callee is an alias.
  SmallVector<aiir::Value> adjustedCallOperands;

  if (calleeAttr) { // direct call
    aiir::Operation *callee =
        aiir::SymbolTable::lookupNearestSymbolFrom(op, calleeAttr);
    if (auto fn = aiir::dyn_cast<aiir::FunctionOpInterface>(callee)) {
      llvmFnTy = converter->convertType<aiir::LLVM::LLVMFunctionType>(
          fn.getFunctionType());
      assert(llvmFnTy && "Failed to convert function type");
    } else if (auto alias = aiir::cast<aiir::LLVM::AliasOp>(callee)) {
      // If the callee was an alias. In that case,
      // we need to prepend the address of the alias to the operands. The
      // way aliases work in the LLVM dialect is a little counter-intuitive.
      // The AliasOp itself is a pseudo-function that returns the address of
      // the global value being aliased, but when we generate the call we
      // need to insert an operation that gets the address of the AliasOp.
      // This all gets sorted out when the LLVM dialect is lowered to LLVM IR.
      auto symAttr = aiir::cast<aiir::FlatSymbolRefAttr>(calleeAttr);
      auto addrOfAlias =
          aiir::LLVM::AddressOfOp::create(
              rewriter, op->getLoc(),
              aiir::LLVM::LLVMPointerType::get(rewriter.getContext()), symAttr)
              .getResult();
      adjustedCallOperands.push_back(addrOfAlias);

      // Now add the regular operands and assign this to the range value.
      llvm::append_range(adjustedCallOperands, callOperands);
      callOperands = adjustedCallOperands;

      // Clear the callee attribute because we're calling an alias.
      calleeAttr = {};
      llvmFnTy = aiir::cast<aiir::LLVM::LLVMFunctionType>(alias.getType());
    } else {
      // Was this an ifunc?
      return op->emitError("Unexpected callee type!");
    }
  } else { // indirect call
    assert(!op->getOperands().empty() &&
           "operands list must no be empty for the indirect call");
    auto calleeTy = op->getOperands().front().getType();
    auto calleePtrTy = cast<cir::PointerType>(calleeTy);
    auto calleeFuncTy = cast<cir::FuncType>(calleePtrTy.getPointee());
    llvm::append_range(adjustedCallOperands, callOperands);
    llvmFnTy = cast<aiir::LLVM::LLVMFunctionType>(
        converter->convertType(calleeFuncTy));
  }

  assert(!cir::MissingFeatures::opCallCallConv());

  if (landingPadBlock) {
    auto newOp = rewriter.replaceOpWithNewOp<aiir::LLVM::InvokeOp>(
        op, llvmFnTy, calleeAttr, callOperands, continueBlock,
        aiir::ValueRange{}, landingPadBlock, aiir::ValueRange{});
    newOp->setAttrs(attributes);
  } else {
    auto newOp = rewriter.replaceOpWithNewOp<aiir::LLVM::CallOp>(
        op, llvmFnTy, calleeAttr, callOperands);
    newOp->setAttrs(attributes);
    if (memoryEffects)
      newOp.setMemoryEffectsAttr(memoryEffects);
    newOp.setNoUnwind(noUnwind);
    newOp.setWillReturn(willReturn);
    newOp.setNoreturn(noReturn);
  }

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMCallOpLowering::matchAndRewrite(
    cir::CallOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return rewriteCallOrInvoke(op.getOperation(), adaptor.getOperands(), rewriter,
                             getTypeConverter(), op.getCalleeAttr());
}

aiir::LogicalResult CIRToLLVMTryCallOpLowering::matchAndRewrite(
    cir::TryCallOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::opCallCallConv());
  return rewriteCallOrInvoke(op.getOperation(), adaptor.getOperands(), rewriter,
                             getTypeConverter(), op.getCalleeAttr(),
                             op.getNormalDest(), op.getUnwindDest());
}

aiir::LogicalResult CIRToLLVMReturnAddrOpLowering::matchAndRewrite(
    cir::ReturnAddrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type llvmPtrTy = getTypeConverter()->convertType(op.getType());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.returnaddress",
                                   llvmPtrTy, adaptor.getOperands());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFrameAddrOpLowering::matchAndRewrite(
    cir::FrameAddrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type llvmPtrTy = getTypeConverter()->convertType(op.getType());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.frameaddress", llvmPtrTy,
                                   adaptor.getOperands());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMClearCacheOpLowering::matchAndRewrite(
    cir::ClearCacheOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value begin = adaptor.getBegin();
  aiir::Value end = adaptor.getEnd();
  auto intrinNameAttr =
      aiir::StringAttr::get(op.getContext(), "llvm.clear_cache");
  rewriter.replaceOpWithNewOp<aiir::LLVM::CallIntrinsicOp>(
      op, aiir::Type{}, intrinNameAttr, aiir::ValueRange{begin, end});

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAddrOfReturnAddrOpLowering::matchAndRewrite(
    cir::AddrOfReturnAddrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type llvmPtrTy = getTypeConverter()->convertType(op.getType());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.addressofreturnaddress",
                                   llvmPtrTy, adaptor.getOperands());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLoadOpLowering::matchAndRewrite(
    cir::LoadOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type llvmTy =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getType());
  aiir::LLVM::AtomicOrdering ordering = getLLVMMemOrder(op.getMemOrder());
  std::optional<size_t> opAlign = op.getAlignment();
  unsigned alignment =
      (unsigned)opAlign.value_or(dataLayout.getTypeABIAlignment(llvmTy));

  assert(!cir::MissingFeatures::lowerModeOptLevel());

  // TODO: nontemporal.
  assert(!cir::MissingFeatures::opLoadStoreNontemporal());

  std::optional<llvm::StringRef> llvmSyncScope =
      getLLVMSyncScope(op.getSyncScope());

  aiir::LLVM::LoadOp newLoad = aiir::LLVM::LoadOp::create(
      rewriter, op->getLoc(), llvmTy, adaptor.getAddr(), alignment,
      op.getIsVolatile(), /*isNonTemporal=*/false,
      /*isInvariant=*/false, /*isInvariantGroup=*/false, ordering,
      llvmSyncScope.value_or(std::string()));

  // Convert adapted result to its original type if needed.
  aiir::Value result =
      emitFromMemory(rewriter, dataLayout, op, newLoad.getResult());
  rewriter.replaceOp(op, result);
  assert(!cir::MissingFeatures::opLoadStoreTbaa());
  return aiir::LogicalResult::success();
}

aiir::LogicalResult
cir::direct::CIRToLLVMVecMaskedLoadOpLowering::matchAndRewrite(
    cir::VecMaskedLoadOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type llvmResTy =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getType());

  std::optional<size_t> opAlign = op.getAlignment();
  unsigned alignment =
      (unsigned)opAlign.value_or(dataLayout.getTypeABIAlignment(llvmResTy));

  aiir::IntegerAttr alignAttr = rewriter.getI32IntegerAttr(alignment);

  auto newLoad = aiir::LLVM::MaskedLoadOp::create(
      rewriter, op.getLoc(), llvmResTy, adaptor.getAddr(), adaptor.getMask(),
      adaptor.getPassThru(), alignAttr);

  rewriter.replaceOp(op, newLoad.getResult());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMStoreOpLowering::matchAndRewrite(
    cir::StoreOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::LLVM::AtomicOrdering memorder = getLLVMMemOrder(op.getMemOrder());
  const aiir::Type llvmTy =
      getTypeConverter()->convertType(op.getValue().getType());
  std::optional<size_t> opAlign = op.getAlignment();
  unsigned alignment =
      (unsigned)opAlign.value_or(dataLayout.getTypeABIAlignment(llvmTy));

  assert(!cir::MissingFeatures::lowerModeOptLevel());

  // Convert adapted value to its memory type if needed.
  aiir::Value value = emitToMemory(rewriter, dataLayout,
                                   op.getValue().getType(), adaptor.getValue());
  // TODO: nontemporal.
  assert(!cir::MissingFeatures::opLoadStoreNontemporal());
  assert(!cir::MissingFeatures::opLoadStoreTbaa());

  std::optional<llvm::StringRef> llvmSyncScope =
      getLLVMSyncScope(op.getSyncScope());

  aiir::LLVM::StoreOp storeOp = aiir::LLVM::StoreOp::create(
      rewriter, op->getLoc(), value, adaptor.getAddr(), alignment,
      op.getIsVolatile(),
      /*isNonTemporal=*/false, /*isInvariantGroup=*/false, memorder,
      llvmSyncScope.value_or(std::string()));
  rewriter.replaceOp(op, storeOp);
  assert(!cir::MissingFeatures::opLoadStoreTbaa());
  return aiir::LogicalResult::success();
}

bool hasTrailingZeros(cir::ConstArrayAttr attr) {
  auto array = aiir::dyn_cast<aiir::ArrayAttr>(attr.getElts());
  return attr.hasTrailingZeros() ||
         (array && std::count_if(array.begin(), array.end(), [](auto elt) {
            auto ar = dyn_cast<cir::ConstArrayAttr>(elt);
            return ar && hasTrailingZeros(ar);
          }));
}

aiir::LogicalResult CIRToLLVMConstantOpLowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Attribute attr = op.getValue();

  if (aiir::isa<cir::PoisonAttr>(attr)) {
    rewriter.replaceOpWithNewOp<aiir::LLVM::PoisonOp>(
        op, getTypeConverter()->convertType(op.getType()));
    return aiir::success();
  }

  if (aiir::isa<aiir::IntegerType>(op.getType())) {
    // Verified cir.const operations cannot actually be of these types, but the
    // lowering pass may generate temporary cir.const operations with these
    // types. This is OK since AIIR allows unverified operations to be alive
    // during a pass as long as they don't live past the end of the pass.
    attr = op.getValue();
  } else if (aiir::isa<cir::BoolType>(op.getType())) {
    int value = aiir::cast<cir::BoolAttr>(op.getValue()).getValue();
    attr = rewriter.getIntegerAttr(typeConverter->convertType(op.getType()),
                                   value);
  } else if (aiir::isa<cir::IntType>(op.getType())) {
    // Lower GlobalViewAttr to llvm.aiir.addressof + llvm.aiir.ptrtoint
    if (auto ga = aiir::dyn_cast<cir::GlobalViewAttr>(op.getValue())) {
      // We can have a global view with an integer type in the case of method
      // pointers, but the lowering of those doesn't go through this path.
      // They are handled in the visitCirAttr. This is left as an error until
      // we have a test case that reaches it.
      assert(!cir::MissingFeatures::globalViewIntLowering());
      op.emitError() << "global view with integer type";
      return aiir::failure();
    }

    attr = rewriter.getIntegerAttr(
        typeConverter->convertType(op.getType()),
        aiir::cast<cir::IntAttr>(op.getValue()).getValue());
  } else if (aiir::isa<cir::FPTypeInterface>(op.getType())) {
    attr = rewriter.getFloatAttr(
        typeConverter->convertType(op.getType()),
        aiir::cast<cir::FPAttr>(op.getValue()).getValue());
  } else if (aiir::isa<cir::PointerType>(op.getType())) {
    // Optimize with dedicated LLVM op for null pointers.
    if (aiir::isa<cir::ConstPtrAttr>(op.getValue())) {
      if (aiir::cast<cir::ConstPtrAttr>(op.getValue()).isNullValue()) {
        rewriter.replaceOpWithNewOp<aiir::LLVM::ZeroOp>(
            op, typeConverter->convertType(op.getType()));
        return aiir::success();
      }
    }
    // Lower GlobalViewAttr to llvm.aiir.addressof
    if (auto gv = aiir::dyn_cast<cir::GlobalViewAttr>(op.getValue())) {
      auto newOp = lowerCirAttrAsValue(op, gv, rewriter, getTypeConverter());
      rewriter.replaceOp(op, newOp);
      return aiir::success();
    }
    attr = op.getValue();
  } else if (const auto arrTy = aiir::dyn_cast<cir::ArrayType>(op.getType())) {
    const auto constArr = aiir::dyn_cast<cir::ConstArrayAttr>(op.getValue());
    if (!constArr && !isa<cir::ZeroAttr, cir::UndefAttr>(op.getValue()))
      return op.emitError() << "array does not have a constant initializer";

    std::optional<aiir::Attribute> denseAttr;
    if (constArr && hasTrailingZeros(constArr)) {
      const aiir::Value newOp =
          lowerCirAttrAsValue(op, constArr, rewriter, getTypeConverter());
      rewriter.replaceOp(op, newOp);
      return aiir::success();
    } else if (constArr &&
               (denseAttr = lowerConstArrayAttr(constArr, typeConverter))) {
      attr = denseAttr.value();
    } else {
      const aiir::Value initVal =
          lowerCirAttrAsValue(op, op.getValue(), rewriter, typeConverter);
      rewriter.replaceOp(op, initVal);
      return aiir::success();
    }
  } else if (const auto recordAttr =
                 aiir::dyn_cast<cir::ConstRecordAttr>(op.getValue())) {
    auto initVal = lowerCirAttrAsValue(op, recordAttr, rewriter, typeConverter);
    rewriter.replaceOp(op, initVal);
    return aiir::success();
  } else if (const auto vecTy = aiir::dyn_cast<cir::VectorType>(op.getType())) {
    rewriter.replaceOp(op, lowerCirAttrAsValue(op, op.getValue(), rewriter,
                                               getTypeConverter()));
    return aiir::success();
  } else if (auto recTy = aiir::dyn_cast<cir::RecordType>(op.getType())) {
    if (aiir::isa<cir::ZeroAttr, cir::UndefAttr>(attr)) {
      aiir::Value initVal =
          lowerCirAttrAsValue(op, attr, rewriter, typeConverter);
      rewriter.replaceOp(op, initVal);
      return aiir::success();
    }
    return op.emitError() << "unsupported lowering for record constant type "
                          << op.getType();
  } else if (auto complexTy = aiir::dyn_cast<cir::ComplexType>(op.getType())) {
    aiir::Type complexElemTy = complexTy.getElementType();
    aiir::Type complexElemLLVMTy = typeConverter->convertType(complexElemTy);

    if (auto zeroInitAttr = aiir::dyn_cast<cir::ZeroAttr>(op.getValue())) {
      aiir::TypedAttr zeroAttr = rewriter.getZeroAttr(complexElemLLVMTy);
      aiir::ArrayAttr array = rewriter.getArrayAttr({zeroAttr, zeroAttr});
      rewriter.replaceOpWithNewOp<aiir::LLVM::ConstantOp>(
          op, getTypeConverter()->convertType(op.getType()), array);
      return aiir::success();
    }

    auto complexAttr = aiir::cast<cir::ConstComplexAttr>(op.getValue());

    aiir::Attribute components[2];
    if (aiir::isa<cir::IntType>(complexElemTy)) {
      components[0] = rewriter.getIntegerAttr(
          complexElemLLVMTy,
          aiir::cast<cir::IntAttr>(complexAttr.getReal()).getValue());
      components[1] = rewriter.getIntegerAttr(
          complexElemLLVMTy,
          aiir::cast<cir::IntAttr>(complexAttr.getImag()).getValue());
    } else {
      components[0] = rewriter.getFloatAttr(
          complexElemLLVMTy,
          aiir::cast<cir::FPAttr>(complexAttr.getReal()).getValue());
      components[1] = rewriter.getFloatAttr(
          complexElemLLVMTy,
          aiir::cast<cir::FPAttr>(complexAttr.getImag()).getValue());
    }

    attr = rewriter.getArrayAttr(components);
  } else {
    return op.emitError() << "unsupported constant type " << op.getType();
  }

  rewriter.replaceOpWithNewOp<aiir::LLVM::ConstantOp>(
      op, getTypeConverter()->convertType(op.getType()), attr);

  return aiir::success();
}

static uint64_t getTypeSize(aiir::Type type, aiir::Operation &op) {
  aiir::DataLayout layout(op.getParentOfType<aiir::ModuleOp>());
  // For LLVM purposes we treat void as u8.
  if (isa<cir::VoidType>(type))
    type = cir::IntType::get(type.getContext(), 8, /*isSigned=*/false);
  return llvm::divideCeil(layout.getTypeSizeInBits(type), 8);
}

aiir::LogicalResult CIRToLLVMPrefetchOpLowering::matchAndRewrite(
    cir::PrefetchOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::Prefetch>(
      op, adaptor.getAddr(), adaptor.getIsWrite(), adaptor.getLocality(),
      /*DataCache=*/1);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMPtrDiffOpLowering::matchAndRewrite(
    cir::PtrDiffOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto dstTy = aiir::cast<cir::IntType>(op.getType());
  aiir::Type llvmDstTy = getTypeConverter()->convertType(dstTy);

  auto lhs = aiir::LLVM::PtrToIntOp::create(rewriter, op.getLoc(), llvmDstTy,
                                            adaptor.getLhs());
  auto rhs = aiir::LLVM::PtrToIntOp::create(rewriter, op.getLoc(), llvmDstTy,
                                            adaptor.getRhs());

  auto diff =
      aiir::LLVM::SubOp::create(rewriter, op.getLoc(), llvmDstTy, lhs, rhs);

  cir::PointerType ptrTy = op.getLhs().getType();
  assert(!cir::MissingFeatures::llvmLoweringPtrDiffConsidersPointee());
  uint64_t typeSize = getTypeSize(ptrTy.getPointee(), *op);

  // Avoid silly division by 1.
  aiir::Value resultVal = diff.getResult();
  if (typeSize != 1) {
    auto typeSizeVal = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                                      llvmDstTy, typeSize);

    if (dstTy.isUnsigned()) {
      auto uDiv =
          aiir::LLVM::UDivOp::create(rewriter, op.getLoc(), diff, typeSizeVal);
      uDiv.setIsExact(true);
      resultVal = uDiv.getResult();
    } else {
      auto sDiv =
          aiir::LLVM::SDivOp::create(rewriter, op.getLoc(), diff, typeSizeVal);
      sDiv.setIsExact(true);
      resultVal = sDiv.getResult();
    }
  }
  rewriter.replaceOp(op, resultVal);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMExpectOpLowering::matchAndRewrite(
    cir::ExpectOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // TODO(cir): do not generate LLVM intrinsics under -O0
  assert(!cir::MissingFeatures::optInfoAttr());

  std::optional<llvm::APFloat> prob = op.getProb();
  if (prob)
    rewriter.replaceOpWithNewOp<aiir::LLVM::ExpectWithProbabilityOp>(
        op, adaptor.getVal(), adaptor.getExpected(), prob.value());
  else
    rewriter.replaceOpWithNewOp<aiir::LLVM::ExpectOp>(op, adaptor.getVal(),
                                                      adaptor.getExpected());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFAbsOpLowering::matchAndRewrite(
    cir::FAbsOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::FAbsOp>(op, resTy,
                                                  adaptor.getOperands()[0]);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAbsOpLowering::matchAndRewrite(
    cir::AbsOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = typeConverter->convertType(op.getType());
  auto absOp = aiir::LLVM::AbsOp::create(rewriter, op.getLoc(), resTy,
                                         adaptor.getOperands()[0],
                                         adaptor.getMinIsPoison());
  rewriter.replaceOp(op, absOp);
  return aiir::success();
}

/// Convert the `cir.func` attributes to `llvm.func` attributes.
/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out
/// argument attributes.
void CIRToLLVMFuncOpLowering::lowerFuncAttributes(
    cir::FuncOp func, bool filterArgAndResAttrs,
    SmallVectorImpl<aiir::NamedAttribute> &result) const {
  assert(!cir::MissingFeatures::opFuncCallingConv());
  for (aiir::NamedAttribute attr : func->getAttrs()) {
    assert(!cir::MissingFeatures::opFuncCallingConv());
    if (attr.getName() == aiir::SymbolTable::getSymbolAttrName() ||
        attr.getName() == func.getFunctionTypeAttrName() ||
        attr.getName() == getLinkageAttrNameString() ||
        attr.getName() == func.getGlobalVisibilityAttrName() ||
        attr.getName() == func.getDsoLocalAttrName() ||
        attr.getName() == func.getInlineKindAttrName() ||
        attr.getName() == func.getSideEffectAttrName() ||
        attr.getName() == CIRDialect::getNoReturnAttrName() ||
        (filterArgAndResAttrs &&
         (attr.getName() == func.getArgAttrsAttrName() ||
          attr.getName() == func.getResAttrsAttrName())))
      continue;

    assert(!cir::MissingFeatures::opFuncExtraAttrs());
    result.push_back(attr);
  }
}

aiir::LogicalResult CIRToLLVMFuncOpLowering::matchAndRewriteAlias(
    cir::FuncOp op, llvm::StringRef aliasee, aiir::Type ty, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  SmallVector<aiir::NamedAttribute, 4> attributes;
  lowerFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

  aiir::Location loc = op.getLoc();
  auto aliasOp = rewriter.replaceOpWithNewOp<aiir::LLVM::AliasOp>(
      op, ty, convertLinkage(op.getLinkage()), op.getName(), op.getDsoLocal(),
      /*threadLocal=*/false, attributes);

  // Create the alias body
  aiir::OpBuilder builder(op.getContext());
  aiir::Block *block = builder.createBlock(&aliasOp.getInitializerRegion());
  builder.setInsertionPointToStart(block);
  // The type of AddressOfOp is always a pointer.
  assert(!cir::MissingFeatures::addressSpace());
  aiir::Type ptrTy = aiir::LLVM::LLVMPointerType::get(ty.getContext());
  auto addrOp = aiir::LLVM::AddressOfOp::create(builder, loc, ptrTy, aliasee);
  aiir::LLVM::ReturnOp::create(builder, loc, addrOp);

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMFuncOpLowering::matchAndRewrite(
    cir::FuncOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {

  cir::FuncType fnType = op.getFunctionType();
  bool isDsoLocal = op.getDsoLocal();
  aiir::TypeConverter::SignatureConversion signatureConversion(
      fnType.getNumInputs());

  for (const auto &argType : llvm::enumerate(fnType.getInputs())) {
    aiir::Type convertedType = typeConverter->convertType(argType.value());
    if (!convertedType)
      return aiir::failure();
    signatureConversion.addInputs(argType.index(), convertedType);
  }

  aiir::Type resultType =
      getTypeConverter()->convertType(fnType.getReturnType());

  // Create the LLVM function operation.
  aiir::Type llvmFnTy = aiir::LLVM::LLVMFunctionType::get(
      resultType ? resultType : aiir::LLVM::LLVMVoidType::get(getContext()),
      signatureConversion.getConvertedTypes(),
      /*isVarArg=*/fnType.isVarArg());

  // If this is an alias, it needs to be lowered to llvm::AliasOp.
  if (std::optional<llvm::StringRef> aliasee = op.getAliasee())
    return matchAndRewriteAlias(op, *aliasee, llvmFnTy, adaptor, rewriter);

  // LLVMFuncOp expects a single FileLine Location instead of a fused
  // location.
  aiir::Location loc = op.getLoc();
  if (aiir::FusedLoc fusedLoc = aiir::dyn_cast<aiir::FusedLoc>(loc))
    loc = fusedLoc.getLocations()[0];
  assert((aiir::isa<aiir::FileLineColLoc>(loc) ||
          aiir::isa<aiir::UnknownLoc>(loc)) &&
         "expected single location or unknown location here");

  aiir::LLVM::Linkage linkage = convertLinkage(op.getLinkage());
  assert(!cir::MissingFeatures::opFuncCallingConv());
  aiir::LLVM::CConv cconv = aiir::LLVM::CConv::C;
  SmallVector<aiir::NamedAttribute, 4> attributes;
  lowerFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

  aiir::LLVM::LLVMFuncOp fn = aiir::LLVM::LLVMFuncOp::create(
      rewriter, loc, op.getName(), llvmFnTy, linkage, isDsoLocal, cconv,
      aiir::SymbolRefAttr(), attributes);

  assert(!cir::MissingFeatures::opFuncMultipleReturnVals());

  if (std::optional<cir::SideEffect> sideEffectKind = op.getSideEffect()) {
    switch (*sideEffectKind) {
    case cir::SideEffect::All:
      break;
    case cir::SideEffect::Pure:
      fn.setMemoryEffectsAttr(aiir::LLVM::MemoryEffectsAttr::get(
          fn.getContext(),
          /*other=*/aiir::LLVM::ModRefInfo::Ref,
          /*argMem=*/aiir::LLVM::ModRefInfo::Ref,
          /*inaccessibleMem=*/aiir::LLVM::ModRefInfo::Ref,
          /*errnoMem=*/aiir::LLVM::ModRefInfo::Ref,
          /*targetMem0=*/aiir::LLVM::ModRefInfo::Ref,
          /*targetMem1=*/aiir::LLVM::ModRefInfo::Ref));
      fn.setNoUnwind(true);
      fn.setWillReturn(true);
      break;
    case cir::SideEffect::Const:
      fn.setMemoryEffectsAttr(aiir::LLVM::MemoryEffectsAttr::get(
          fn.getContext(),
          /*other=*/aiir::LLVM::ModRefInfo::NoModRef,
          /*argMem=*/aiir::LLVM::ModRefInfo::NoModRef,
          /*inaccessibleMem=*/aiir::LLVM::ModRefInfo::NoModRef,
          /*errnoMem=*/aiir::LLVM::ModRefInfo::NoModRef,
          /*targetMem0=*/aiir::LLVM::ModRefInfo::NoModRef,
          /*targetMem1=*/aiir::LLVM::ModRefInfo::NoModRef));
      fn.setNoUnwind(true);
      fn.setWillReturn(true);
      break;
    }
  }

  if (op->hasAttr(CIRDialect::getNoReturnAttrName()))
    fn.setNoreturn(true);

  if (std::optional<cir::InlineKind> inlineKind = op.getInlineKind()) {
    fn.setNoInline(*inlineKind == cir::InlineKind::NoInline);
    fn.setInlineHint(*inlineKind == cir::InlineKind::InlineHint);
    fn.setAlwaysInline(*inlineKind == cir::InlineKind::AlwaysInline);
  }

  if (std::optional<llvm::StringRef> personality = op.getPersonality())
    fn.setPersonality(*personality);

  fn.setVisibility_(
      lowerCIRVisibilityToLLVMVisibility(op.getGlobalVisibility()));

  rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
  if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                         &signatureConversion)))
    return aiir::failure();

  rewriter.eraseOp(op);

  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMGetGlobalOpLowering::matchAndRewrite(
    cir::GetGlobalOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
  // CIRGen should mitigate this and not emit the get_global.
  if (op->getUses().empty()) {
    rewriter.eraseOp(op);
    return aiir::success();
  }

  aiir::Type type = getTypeConverter()->convertType(op.getType());
  aiir::Operation *newop = aiir::LLVM::AddressOfOp::create(
      rewriter, op.getLoc(), type, op.getName());

  if (op.getTls()) {
    // Handle access to TLS via intrinsic.
    newop = aiir::LLVM::ThreadlocalAddressOp::create(rewriter, op.getLoc(),
                                                     type, newop->getResult(0));
  }

  rewriter.replaceOp(op, newop);
  return aiir::success();
}

/// Replace CIR global with a region initialized LLVM global and update
/// insertion point to the end of the initializer block.
void CIRToLLVMGlobalOpLowering::setupRegionInitializedLLVMGlobalOp(
    cir::GlobalOp op, aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type llvmType =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getSymType());

  // FIXME: These default values are placeholders until the the equivalent
  //        attributes are available on cir.global ops. This duplicates code
  //        in CIRToLLVMGlobalOpLowering::matchAndRewrite() but that will go
  //        away when the placeholders are no longer needed.
  const bool isConst = op.getConstant();
  unsigned addrSpace = 0;
  if (auto targetAS = aiir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
          op.getAddrSpaceAttr()))
    addrSpace = targetAS.getValue();
  const bool isDsoLocal = op.getDsoLocal();
  const bool isThreadLocal = (bool)op.getTlsModelAttr();
  const uint64_t alignment = op.getAlignment().value_or(0);
  const aiir::LLVM::Linkage linkage = convertLinkage(op.getLinkage());
  const StringRef symbol = op.getSymName();
  aiir::SymbolRefAttr comdatAttr = getComdatAttr(op, rewriter);

  SmallVector<aiir::NamedAttribute> attributes;
  aiir::LLVM::GlobalOp newGlobalOp =
      rewriter.replaceOpWithNewOp<aiir::LLVM::GlobalOp>(
          op, llvmType, isConst, linkage, symbol, nullptr, alignment, addrSpace,
          isDsoLocal, isThreadLocal, comdatAttr, attributes);
  newGlobalOp.getRegion().emplaceBlock();
  rewriter.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());
}

aiir::LogicalResult
CIRToLLVMGlobalOpLowering::matchAndRewriteRegionInitializedGlobal(
    cir::GlobalOp op, aiir::Attribute init,
    aiir::ConversionPatternRewriter &rewriter) const {
  // TODO: Generalize this handling when more types are needed here.
  assert(
      (isa<cir::ConstArrayAttr, cir::ConstRecordAttr, cir::ConstVectorAttr,
           cir::ConstPtrAttr, cir::ConstComplexAttr, cir::GlobalViewAttr,
           cir::TypeInfoAttr, cir::UndefAttr, cir::VTableAttr, cir::ZeroAttr>(
          init)));

  // TODO(cir): once LLVM's dialect has proper equivalent attributes this
  // should be updated. For now, we use a custom op to initialize globals
  // to the appropriate value.
  const aiir::Location loc = op.getLoc();
  setupRegionInitializedLLVMGlobalOp(op, rewriter);
  CIRAttrToValue valueConverter(op, rewriter, typeConverter);
  aiir::Value value = valueConverter.visit(init);
  aiir::LLVM::ReturnOp::create(rewriter, loc, value);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMGlobalOpLowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // If this global requires non-trivial initialization or destruction,
  // that needs to be moved to runtime handlers during LoweringPrepare.
  if (!op.getCtorRegion().empty() || !op.getDtorRegion().empty())
    return op.emitError() << "GlobalOp ctor and dtor regions should be removed "
                             "in LoweringPrepare";

  std::optional<aiir::Attribute> init = op.getInitialValue();

  // Fetch required values to create LLVM op.
  const aiir::Type cirSymType = op.getSymType();

  // This is the LLVM dialect type.
  const aiir::Type llvmType =
      convertTypeForMemory(*getTypeConverter(), dataLayout, cirSymType);

  // FIXME: These default values are placeholders until the the equivalent
  //        attributes are available on cir.global ops.
  const bool isConst = op.getConstant();
  unsigned addrSpace = 0;
  if (auto targetAS = aiir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
          op.getAddrSpaceAttr()))
    addrSpace = targetAS.getValue();
  const bool isDsoLocal = op.getDsoLocal();
  const bool isThreadLocal = (bool)op.getTlsModelAttr();
  const uint64_t alignment = op.getAlignment().value_or(0);
  const aiir::LLVM::Linkage linkage = convertLinkage(op.getLinkage());
  const StringRef symbol = op.getSymName();
  SmallVector<aiir::NamedAttribute> attributes;

  // Mark externally_initialized for __device__ and __constant__
  if (auto extInit =
          op->getAttr(CUDAExternallyInitializedAttr::getMnemonic())) {
    attributes.push_back(rewriter.getNamedAttr("externally_initialized",
                                               rewriter.getUnitAttr()));
  }

  if (init.has_value()) {
    if (aiir::isa<cir::FPAttr, cir::IntAttr, cir::BoolAttr>(init.value())) {
      GlobalInitAttrRewriter initRewriter(llvmType, rewriter);
      init = initRewriter.visit(init.value());
      // If initRewriter returned a null attribute, init will have a value but
      // the value will be null. If that happens, initRewriter didn't handle the
      // attribute type. It probably needs to be added to
      // GlobalInitAttrRewriter.
      if (!init.value()) {
        op.emitError() << "unsupported initializer '" << init.value() << "'";
        return aiir::failure();
      }
    } else if (aiir::isa<cir::ConstArrayAttr, cir::ConstVectorAttr,
                         cir::ConstRecordAttr, cir::ConstPtrAttr,
                         cir::ConstComplexAttr, cir::GlobalViewAttr,
                         cir::TypeInfoAttr, cir::UndefAttr, cir::VTableAttr,
                         cir::ZeroAttr>(init.value())) {
      // TODO(cir): once LLVM's dialect has proper equivalent attributes this
      // should be updated. For now, we use a custom op to initialize globals
      // to the appropriate value.
      return matchAndRewriteRegionInitializedGlobal(op, init.value(), rewriter);
    } else {
      // We will only get here if new initializer types are added and this
      // code is not updated to handle them.
      op.emitError() << "unsupported initializer '" << init.value() << "'";
      return aiir::failure();
    }
  }

  aiir::LLVM::Visibility visibility =
      lowerCIRVisibilityToLLVMVisibility(op.getGlobalVisibility());
  aiir::SymbolRefAttr comdatAttr = getComdatAttr(op, rewriter);
  auto newOp = rewriter.replaceOpWithNewOp<aiir::LLVM::GlobalOp>(
      op, llvmType, isConst, linkage, symbol, init.value_or(aiir::Attribute()),
      alignment, addrSpace, isDsoLocal, isThreadLocal, comdatAttr, attributes);
  newOp.setVisibility_(visibility);

  return aiir::success();
}

aiir::SymbolRefAttr
CIRToLLVMGlobalOpLowering::getComdatAttr(cir::GlobalOp &op,
                                         aiir::OpBuilder &builder) const {
  if (!op.getComdat())
    return aiir::SymbolRefAttr{};

  aiir::ModuleOp module = op->getParentOfType<aiir::ModuleOp>();
  aiir::OpBuilder::InsertionGuard guard(builder);
  StringRef comdatName("__llvm_comdat_globals");
  if (!comdatOp) {
    builder.setInsertionPointToStart(module.getBody());
    comdatOp =
        aiir::LLVM::ComdatOp::create(builder, module.getLoc(), comdatName);
  }

  if (auto comdatSelector = comdatOp.lookupSymbol<aiir::LLVM::ComdatSelectorOp>(
          op.getSymName())) {
    return aiir::SymbolRefAttr::get(
        builder.getContext(), comdatName,
        aiir::FlatSymbolRefAttr::get(comdatSelector.getSymNameAttr()));
  }

  builder.setInsertionPointToStart(&comdatOp.getBody().back());
  auto selectorOp = aiir::LLVM::ComdatSelectorOp::create(
      builder, comdatOp.getLoc(), op.getSymName(),
      aiir::LLVM::comdat::Comdat::Any);
  return aiir::SymbolRefAttr::get(
      builder.getContext(), comdatName,
      aiir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr()));
}

aiir::LogicalResult CIRToLLVMSwitchFlatOpLowering::matchAndRewrite(
    cir::SwitchFlatOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {

  llvm::SmallVector<aiir::APInt, 8> caseValues;
  for (aiir::Attribute val : op.getCaseValues()) {
    auto intAttr = cast<cir::IntAttr>(val);
    caseValues.push_back(intAttr.getValue());
  }

  llvm::SmallVector<aiir::Block *, 8> caseDestinations;
  llvm::SmallVector<aiir::ValueRange, 8> caseOperands;

  for (aiir::Block *x : op.getCaseDestinations())
    caseDestinations.push_back(x);

  for (aiir::OperandRange x : op.getCaseOperands())
    caseOperands.push_back(x);

  // Set switch op to branch to the newly created blocks.
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<aiir::LLVM::SwitchOp>(
      op, adaptor.getCondition(), op.getDefaultDestination(),
      op.getDefaultOperands(), caseValues, caseDestinations, caseOperands);
  return aiir::success();
}

static aiir::LLVM::IntegerOverflowFlags nswFlag(bool nsw) {
  return nsw ? aiir::LLVM::IntegerOverflowFlags::nsw
             : aiir::LLVM::IntegerOverflowFlags::none;
}

template <typename CIROp, typename LLVMIntOp>
static aiir::LogicalResult
lowerIncDecOp(CIROp op, typename CIROp::Adaptor adaptor,
              aiir::ConversionPatternRewriter &rewriter, double fpConstant) {
  aiir::Type elementType = elementTypeIfVector(op.getType());
  aiir::Type llvmType = adaptor.getInput().getType();
  aiir::Location loc = op.getLoc();

  if (aiir::isa<cir::IntType>(elementType)) {
    auto maybeNSW = nswFlag(op.getNoSignedWrap());
    auto one = aiir::LLVM::ConstantOp::create(rewriter, loc, llvmType, 1);
    rewriter.replaceOpWithNewOp<LLVMIntOp>(op, adaptor.getInput(), one,
                                           maybeNSW);
    return aiir::success();
  }
  if (aiir::isa<cir::FPTypeInterface>(elementType)) {
    auto fpConst = aiir::LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(llvmType, fpConstant));
    rewriter.replaceOpWithNewOp<aiir::LLVM::FAddOp>(op, fpConst,
                                                    adaptor.getInput());
    return aiir::success();
  }
  return op.emitError() << "Unsupported type for IncOp/DecOp";
}

aiir::LogicalResult CIRToLLVMIncOpLowering::matchAndRewrite(
    cir::IncOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerIncDecOp<cir::IncOp, aiir::LLVM::AddOp>(op, adaptor, rewriter,
                                                      1.0);
}

aiir::LogicalResult CIRToLLVMDecOpLowering::matchAndRewrite(
    cir::DecOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerIncDecOp<cir::DecOp, aiir::LLVM::SubOp>(op, adaptor, rewriter,
                                                      -1.0);
}

aiir::LogicalResult CIRToLLVMMinusOpLowering::matchAndRewrite(
    cir::MinusOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type elementType = elementTypeIfVector(op.getType());
  bool isVector = aiir::isa<cir::VectorType>(op.getType());
  aiir::Type llvmType = adaptor.getInput().getType();
  aiir::Location loc = op.getLoc();

  if (aiir::isa<cir::IntType>(elementType)) {
    auto maybeNSW = nswFlag(op.getNoSignedWrap());
    aiir::Value zero;
    if (isVector)
      zero = aiir::LLVM::ZeroOp::create(rewriter, loc, llvmType);
    else
      zero = aiir::LLVM::ConstantOp::create(rewriter, loc, llvmType, 0);
    rewriter.replaceOpWithNewOp<aiir::LLVM::SubOp>(op, zero, adaptor.getInput(),
                                                   maybeNSW);
    return aiir::success();
  }
  if (aiir::isa<cir::FPTypeInterface>(elementType)) {
    rewriter.replaceOpWithNewOp<aiir::LLVM::FNegOp>(op, adaptor.getInput());
    return aiir::success();
  }
  return op.emitError() << "Unsupported type for unary minus";
}

aiir::LogicalResult CIRToLLVMNotOpLowering::matchAndRewrite(
    cir::NotOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type elementType = elementTypeIfVector(op.getType());
  bool isVector = aiir::isa<cir::VectorType>(op.getType());
  aiir::Type llvmType = adaptor.getInput().getType();
  aiir::Location loc = op.getLoc();

  if (aiir::isa<cir::IntType>(elementType)) {
    aiir::Value minusOne;
    if (isVector) {
      const uint64_t numElements =
          aiir::dyn_cast<cir::VectorType>(op.getType()).getSize();
      SmallVector<int32_t> values(numElements, -1);
      aiir::DenseIntElementsAttr denseVec = rewriter.getI32VectorAttr(values);
      minusOne =
          aiir::LLVM::ConstantOp::create(rewriter, loc, llvmType, denseVec);
    } else {
      minusOne = aiir::LLVM::ConstantOp::create(rewriter, loc, llvmType, -1);
    }
    rewriter.replaceOpWithNewOp<aiir::LLVM::XOrOp>(op, adaptor.getInput(),
                                                   minusOne);
    return aiir::success();
  }
  if (aiir::isa<cir::BoolType>(elementType)) {
    auto one = aiir::LLVM::ConstantOp::create(rewriter, loc, llvmType, 1);
    rewriter.replaceOpWithNewOp<aiir::LLVM::XOrOp>(op, adaptor.getInput(), one);
    return aiir::success();
  }
  return op.emitError() << "Unsupported type for bitwise NOT";
}

static bool isIntTypeUnsigned(aiir::Type type) {
  // TODO: Ideally, we should only need to check cir::IntType here.
  return aiir::isa<cir::IntType>(type)
             ? aiir::cast<cir::IntType>(type).isUnsigned()
             : aiir::cast<aiir::IntegerType>(type).isUnsigned();
}

//===----------------------------------------------------------------------===//
// Binary Op Lowering
//===----------------------------------------------------------------------===//

template <typename BinOp>
static aiir::LLVM::IntegerOverflowFlags intOverflowFlag(BinOp op) {
  if (op.getNoUnsignedWrap())
    return aiir::LLVM::IntegerOverflowFlags::nuw;
  if (op.getNoSignedWrap())
    return aiir::LLVM::IntegerOverflowFlags::nsw;
  return aiir::LLVM::IntegerOverflowFlags::none;
}

/// Lower an arithmetic op that supports saturation, overflow flags, and an FP
/// variant. Used for Add and Sub which share identical dispatch logic.
template <typename UIntSatOp, typename SIntSatOp, typename IntOp, typename FPOp,
          typename CIROp>
static aiir::LogicalResult
lowerSaturatableArithOp(CIROp op, aiir::Value lhs, aiir::Value rhs,
                        aiir::ConversionPatternRewriter &rewriter) {
  const aiir::Type eltType = elementTypeIfVector(op.getRhs().getType());
  if (cir::isIntOrBoolType(eltType)) {
    if (op.getSaturated()) {
      if (isIntTypeUnsigned(eltType))
        rewriter.replaceOpWithNewOp<UIntSatOp>(op, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<SIntSatOp>(op, lhs, rhs);
      return aiir::success();
    }
    rewriter.replaceOpWithNewOp<IntOp>(op, lhs, rhs, intOverflowFlag(op));
  } else {
    rewriter.replaceOpWithNewOp<FPOp>(op, lhs, rhs);
  }
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAddOpLowering::matchAndRewrite(
    cir::AddOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerSaturatableArithOp<aiir::LLVM::UAddSat, aiir::LLVM::SAddSat,
                                 aiir::LLVM::AddOp, aiir::LLVM::FAddOp>(
      op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
}

aiir::LogicalResult CIRToLLVMSubOpLowering::matchAndRewrite(
    cir::SubOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerSaturatableArithOp<aiir::LLVM::USubSat, aiir::LLVM::SSubSat,
                                 aiir::LLVM::SubOp, aiir::LLVM::FSubOp>(
      op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
}

aiir::LogicalResult CIRToLLVMMulOpLowering::matchAndRewrite(
    cir::MulOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Value lhs = adaptor.getLhs();
  const aiir::Value rhs = adaptor.getRhs();
  if (cir::isIntOrBoolType(elementTypeIfVector(op.getRhs().getType()))) {
    rewriter.replaceOpWithNewOp<aiir::LLVM::MulOp>(op, lhs, rhs,
                                                   intOverflowFlag(op));
  } else {
    rewriter.replaceOpWithNewOp<aiir::LLVM::FMulOp>(op, lhs, rhs);
  }
  return aiir::success();
}

/// Lower a binary op that maps to unsigned/signed/FP LLVM ops depending on
/// operand type. Used for Div and Rem which share identical dispatch logic.
template <typename UIntOp, typename SIntOp, typename FPOp, typename CIROp>
static aiir::LogicalResult
lowerIntFPBinaryOp(CIROp op, aiir::Value lhs, aiir::Value rhs,
                   aiir::ConversionPatternRewriter &rewriter) {
  const aiir::Type eltType = elementTypeIfVector(op.getRhs().getType());
  if (cir::isIntOrBoolType(eltType)) {
    if (isIntTypeUnsigned(eltType))
      rewriter.replaceOpWithNewOp<UIntOp>(op, lhs, rhs);
    else
      rewriter.replaceOpWithNewOp<SIntOp>(op, lhs, rhs);
  } else {
    rewriter.replaceOpWithNewOp<FPOp>(op, lhs, rhs);
  }
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMDivOpLowering::matchAndRewrite(
    cir::DivOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerIntFPBinaryOp<aiir::LLVM::UDivOp, aiir::LLVM::SDivOp,
                            aiir::LLVM::FDivOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), rewriter);
}

aiir::LogicalResult CIRToLLVMRemOpLowering::matchAndRewrite(
    cir::RemOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerIntFPBinaryOp<aiir::LLVM::URemOp, aiir::LLVM::SRemOp,
                            aiir::LLVM::FRemOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), rewriter);
}

aiir::LogicalResult CIRToLLVMAndOpLowering::matchAndRewrite(
    cir::AndOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::AndOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMOrOpLowering::matchAndRewrite(
    cir::OrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::OrOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMXorOpLowering::matchAndRewrite(
    cir::XorOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::XOrOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
  return aiir::success();
}

template <typename CIROp, typename UIntOp, typename SIntOp>
static aiir::LogicalResult
lowerMinMaxOp(CIROp op, typename CIROp::Adaptor adaptor,
              aiir::ConversionPatternRewriter &rewriter) {
  const aiir::Value lhs = adaptor.getLhs();
  const aiir::Value rhs = adaptor.getRhs();
  if (isIntTypeUnsigned(elementTypeIfVector(op.getRhs().getType())))
    rewriter.replaceOpWithNewOp<UIntOp>(op, lhs, rhs);
  else
    rewriter.replaceOpWithNewOp<SIntOp>(op, lhs, rhs);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMMaxOpLowering::matchAndRewrite(
    cir::MaxOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerMinMaxOp<cir::MaxOp, aiir::LLVM::UMaxOp, aiir::LLVM::SMaxOp>(
      op, adaptor, rewriter);
}

aiir::LogicalResult CIRToLLVMMinOpLowering::matchAndRewrite(
    cir::MinOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerMinMaxOp<cir::MinOp, aiir::LLVM::UMinOp, aiir::LLVM::SMinOp>(
      op, adaptor, rewriter);
}

/// Convert from a CIR comparison kind to an LLVM IR integral comparison kind.
static aiir::LLVM::ICmpPredicate
convertCmpKindToICmpPredicate(cir::CmpOpKind kind, bool isSigned) {
  using CIR = cir::CmpOpKind;
  using LLVMICmp = aiir::LLVM::ICmpPredicate;
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
static aiir::LLVM::FCmpPredicate
convertCmpKindToFCmpPredicate(cir::CmpOpKind kind) {
  using CIR = cir::CmpOpKind;
  using LLVMFCmp = aiir::LLVM::FCmpPredicate;
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

aiir::LogicalResult CIRToLLVMCmpOpLowering::matchAndRewrite(
    cir::CmpOp cmpOp, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type type = cmpOp.getLhs().getType();

  if (aiir::isa<cir::IntType, aiir::IntegerType>(type)) {
    bool isSigned = aiir::isa<cir::IntType>(type)
                        ? aiir::cast<cir::IntType>(type).isSigned()
                        : aiir::cast<aiir::IntegerType>(type).isSigned();
    aiir::LLVM::ICmpPredicate kind =
        convertCmpKindToICmpPredicate(cmpOp.getKind(), isSigned);
    rewriter.replaceOpWithNewOp<aiir::LLVM::ICmpOp>(
        cmpOp, kind, adaptor.getLhs(), adaptor.getRhs());
    return aiir::success();
  }

  if (auto ptrTy = aiir::dyn_cast<cir::PointerType>(type)) {
    aiir::LLVM::ICmpPredicate kind =
        convertCmpKindToICmpPredicate(cmpOp.getKind(),
                                      /* isSigned=*/false);
    rewriter.replaceOpWithNewOp<aiir::LLVM::ICmpOp>(
        cmpOp, kind, adaptor.getLhs(), adaptor.getRhs());
    return aiir::success();
  }

  if (auto vptrTy = aiir::dyn_cast<cir::VPtrType>(type)) {
    // !cir.vptr is a special case, but it's just a pointer to LLVM.
    auto kind = convertCmpKindToICmpPredicate(cmpOp.getKind(),
                                              /* isSigned=*/false);
    rewriter.replaceOpWithNewOp<aiir::LLVM::ICmpOp>(
        cmpOp, kind, adaptor.getLhs(), adaptor.getRhs());
    return aiir::success();
  }

  if (aiir::isa<cir::FPTypeInterface>(type)) {
    aiir::LLVM::FCmpPredicate kind =
        convertCmpKindToFCmpPredicate(cmpOp.getKind());
    rewriter.replaceOpWithNewOp<aiir::LLVM::FCmpOp>(
        cmpOp, kind, adaptor.getLhs(), adaptor.getRhs());
    return aiir::success();
  }

  if (aiir::isa<cir::ComplexType>(type)) {
    aiir::Value lhs = adaptor.getLhs();
    aiir::Value rhs = adaptor.getRhs();
    aiir::Location loc = cmpOp.getLoc();

    auto complexType = aiir::cast<cir::ComplexType>(cmpOp.getLhs().getType());
    aiir::Type complexElemTy =
        getTypeConverter()->convertType(complexType.getElementType());

    auto lhsReal = aiir::LLVM::ExtractValueOp::create(
        rewriter, loc, complexElemTy, lhs, ArrayRef(int64_t{0}));
    auto lhsImag = aiir::LLVM::ExtractValueOp::create(
        rewriter, loc, complexElemTy, lhs, ArrayRef(int64_t{1}));
    auto rhsReal = aiir::LLVM::ExtractValueOp::create(
        rewriter, loc, complexElemTy, rhs, ArrayRef(int64_t{0}));
    auto rhsImag = aiir::LLVM::ExtractValueOp::create(
        rewriter, loc, complexElemTy, rhs, ArrayRef(int64_t{1}));

    if (cmpOp.getKind() == cir::CmpOpKind::eq) {
      if (complexElemTy.isInteger()) {
        auto realCmp = aiir::LLVM::ICmpOp::create(
            rewriter, loc, aiir::LLVM::ICmpPredicate::eq, lhsReal, rhsReal);
        auto imagCmp = aiir::LLVM::ICmpOp::create(
            rewriter, loc, aiir::LLVM::ICmpPredicate::eq, lhsImag, rhsImag);
        rewriter.replaceOpWithNewOp<aiir::LLVM::AndOp>(cmpOp, realCmp, imagCmp);
        return aiir::success();
      }

      auto realCmp = aiir::LLVM::FCmpOp::create(
          rewriter, loc, aiir::LLVM::FCmpPredicate::oeq, lhsReal, rhsReal);
      auto imagCmp = aiir::LLVM::FCmpOp::create(
          rewriter, loc, aiir::LLVM::FCmpPredicate::oeq, lhsImag, rhsImag);
      rewriter.replaceOpWithNewOp<aiir::LLVM::AndOp>(cmpOp, realCmp, imagCmp);
      return aiir::success();
    }

    if (cmpOp.getKind() == cir::CmpOpKind::ne) {
      if (complexElemTy.isInteger()) {
        auto realCmp = aiir::LLVM::ICmpOp::create(
            rewriter, loc, aiir::LLVM::ICmpPredicate::ne, lhsReal, rhsReal);
        auto imagCmp = aiir::LLVM::ICmpOp::create(
            rewriter, loc, aiir::LLVM::ICmpPredicate::ne, lhsImag, rhsImag);
        rewriter.replaceOpWithNewOp<aiir::LLVM::OrOp>(cmpOp, realCmp, imagCmp);
        return aiir::success();
      }

      auto realCmp = aiir::LLVM::FCmpOp::create(
          rewriter, loc, aiir::LLVM::FCmpPredicate::une, lhsReal, rhsReal);
      auto imagCmp = aiir::LLVM::FCmpOp::create(
          rewriter, loc, aiir::LLVM::FCmpPredicate::une, lhsImag, rhsImag);
      rewriter.replaceOpWithNewOp<aiir::LLVM::OrOp>(cmpOp, realCmp, imagCmp);
      return aiir::success();
    }
  }

  return cmpOp.emitError() << "unsupported type for CmpOp: " << type;
}

/// Shared lowering logic for checked binary arithmetic overflow operations.
/// The \p opStr parameter specifies the arithmetic operation name used in the
/// LLVM intrinsic (e.g., "add", "sub", "mul").
template <typename OpTy>
static aiir::LogicalResult
lowerBinOpOverflow(OpTy op, typename OpTy::Adaptor adaptor,
                   aiir::ConversionPatternRewriter &rewriter,
                   const aiir::TypeConverter *typeConverter,
                   llvm::StringRef opStr) {
  aiir::Location loc = op.getLoc();
  cir::IntType operandTy = op.getLhs().getType();
  cir::IntType resultTy = op.getResult().getType();

  bool sign = operandTy.getIsSigned() || resultTy.getIsSigned();
  unsigned width =
      std::max(operandTy.getWidth() + (sign && operandTy.isUnsigned()),
               resultTy.getWidth() + (sign && resultTy.isUnsigned()));

  aiir::IntegerType encompassedLLVMTy = rewriter.getIntegerType(width);

  aiir::Value lhs = adaptor.getLhs();
  aiir::Value rhs = adaptor.getRhs();
  if (operandTy.getWidth() < width) {
    if (operandTy.isSigned()) {
      lhs = aiir::LLVM::SExtOp::create(rewriter, loc, encompassedLLVMTy, lhs);
      rhs = aiir::LLVM::SExtOp::create(rewriter, loc, encompassedLLVMTy, rhs);
    } else {
      lhs = aiir::LLVM::ZExtOp::create(rewriter, loc, encompassedLLVMTy, lhs);
      rhs = aiir::LLVM::ZExtOp::create(rewriter, loc, encompassedLLVMTy, rhs);
    }
  }

  // The intrinsic name is `@llvm.{s|u}{op}.with.overflow.i{width}`
  std::string intrinName = ("llvm." + llvm::Twine(sign ? 's' : 'u') + opStr +
                            ".with.overflow.i" + llvm::Twine(width))
                               .str();
  auto intrinNameAttr = aiir::StringAttr::get(op.getContext(), intrinName);

  aiir::IntegerType overflowLLVMTy = rewriter.getI1Type();
  auto intrinRetTy = aiir::LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {encompassedLLVMTy, overflowLLVMTy});

  auto callLLVMIntrinOp = aiir::LLVM::CallIntrinsicOp::create(
      rewriter, loc, intrinRetTy, intrinNameAttr, aiir::ValueRange{lhs, rhs});
  aiir::Value intrinRet = callLLVMIntrinOp.getResult(0);

  aiir::Value result = aiir::LLVM::ExtractValueOp::create(
                           rewriter, loc, intrinRet, ArrayRef<int64_t>{0})
                           .getResult();
  aiir::Value overflow = aiir::LLVM::ExtractValueOp::create(
                             rewriter, loc, intrinRet, ArrayRef<int64_t>{1})
                             .getResult();

  if (resultTy.getWidth() < width) {
    aiir::Type resultLLVMTy = typeConverter->convertType(resultTy);
    auto truncResult =
        aiir::LLVM::TruncOp::create(rewriter, loc, resultLLVMTy, result);

    // Extend the truncated result back to the encompassing type to check for
    // any overflows during the truncation.
    aiir::Value truncResultExt;
    if (resultTy.isSigned())
      truncResultExt = aiir::LLVM::SExtOp::create(
          rewriter, loc, encompassedLLVMTy, truncResult);
    else
      truncResultExt = aiir::LLVM::ZExtOp::create(
          rewriter, loc, encompassedLLVMTy, truncResult);
    auto truncOverflow = aiir::LLVM::ICmpOp::create(
        rewriter, loc, aiir::LLVM::ICmpPredicate::ne, truncResultExt, result);

    result = truncResult;
    overflow = aiir::LLVM::OrOp::create(rewriter, loc, overflow, truncOverflow);
  }

  aiir::Type boolLLVMTy =
      typeConverter->convertType(op.getOverflow().getType());
  if (boolLLVMTy != rewriter.getI1Type())
    overflow = aiir::LLVM::ZExtOp::create(rewriter, loc, boolLLVMTy, overflow);

  rewriter.replaceOp(op, aiir::ValueRange{result, overflow});

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAddOverflowOpLowering::matchAndRewrite(
    cir::AddOverflowOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerBinOpOverflow(op, adaptor, rewriter, getTypeConverter(), "add");
}

aiir::LogicalResult CIRToLLVMSubOverflowOpLowering::matchAndRewrite(
    cir::SubOverflowOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerBinOpOverflow(op, adaptor, rewriter, getTypeConverter(), "sub");
}

aiir::LogicalResult CIRToLLVMMulOverflowOpLowering::matchAndRewrite(
    cir::MulOverflowOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return lowerBinOpOverflow(op, adaptor, rewriter, getTypeConverter(), "mul");
}

aiir::LogicalResult CIRToLLVMShiftOpLowering::matchAndRewrite(
    cir::ShiftOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  assert((op.getValue().getType() == op.getType()) &&
         "inconsistent operands' types NYI");

  const aiir::Type llvmTy = getTypeConverter()->convertType(op.getType());
  aiir::Value amt = adaptor.getAmount();
  aiir::Value val = adaptor.getValue();

  auto cirAmtTy = aiir::dyn_cast<cir::IntType>(op.getAmount().getType());
  bool isUnsigned;
  if (cirAmtTy) {
    auto cirValTy = aiir::cast<cir::IntType>(op.getValue().getType());
    isUnsigned = cirValTy.isUnsigned();

    // Ensure shift amount is the same type as the value. Some undefined
    // behavior might occur in the casts below as per [C99 6.5.7.3].
    // Vector type shift amount needs no cast as type consistency is expected to
    // be already be enforced at CIRGen.
    if (cirAmtTy)
      amt = getLLVMIntCast(rewriter, amt, llvmTy, true, cirAmtTy.getWidth(),
                           cirValTy.getWidth());
  } else {
    auto cirValVTy = aiir::cast<cir::VectorType>(op.getValue().getType());
    isUnsigned =
        aiir::cast<cir::IntType>(cirValVTy.getElementType()).isUnsigned();
  }

  // Lower to the proper LLVM shift operation.
  if (op.getIsShiftleft()) {
    rewriter.replaceOpWithNewOp<aiir::LLVM::ShlOp>(op, llvmTy, val, amt);
    return aiir::success();
  }

  if (isUnsigned)
    rewriter.replaceOpWithNewOp<aiir::LLVM::LShrOp>(op, llvmTy, val, amt);
  else
    rewriter.replaceOpWithNewOp<aiir::LLVM::AShrOp>(op, llvmTy, val, amt);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMSelectOpLowering::matchAndRewrite(
    cir::SelectOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto getConstantBool = [](aiir::Value value) -> cir::BoolAttr {
    auto definingOp = value.getDefiningOp<cir::ConstantOp>();
    if (!definingOp)
      return {};

    auto constValue = definingOp.getValueAttr<cir::BoolAttr>();
    if (!constValue)
      return {};

    return constValue;
  };

  // Two special cases in the LLVMIR codegen of select op:
  // - select %0, %1, false => and %0, %1
  // - select %0, true, %1 => or %0, %1
  if (aiir::isa<cir::BoolType>(op.getTrueValue().getType())) {
    cir::BoolAttr trueValue = getConstantBool(op.getTrueValue());
    cir::BoolAttr falseValue = getConstantBool(op.getFalseValue());
    if (falseValue && !falseValue.getValue()) {
      // select %0, %1, false => and %0, %1
      rewriter.replaceOpWithNewOp<aiir::LLVM::AndOp>(op, adaptor.getCondition(),
                                                     adaptor.getTrueValue());
      return aiir::success();
    }
    if (trueValue && trueValue.getValue()) {
      // select %0, true, %1 => or %0, %1
      rewriter.replaceOpWithNewOp<aiir::LLVM::OrOp>(op, adaptor.getCondition(),
                                                    adaptor.getFalseValue());
      return aiir::success();
    }
  }

  aiir::Value llvmCondition = adaptor.getCondition();
  rewriter.replaceOpWithNewOp<aiir::LLVM::SelectOp>(
      op, llvmCondition, adaptor.getTrueValue(), adaptor.getFalseValue());

  return aiir::success();
}

static void prepareTypeConverter(aiir::LLVMTypeConverter &converter,
                                 aiir::DataLayout &dataLayout) {
  converter.addConversion([&](cir::PointerType type) -> aiir::Type {
    aiir::ptr::MemorySpaceAttrInterface addrSpaceAttr = type.getAddrSpace();
    unsigned numericAS = 0;

    if (auto langAsAttr =
            aiir::dyn_cast_if_present<cir::LangAddressSpaceAttr>(addrSpaceAttr))
      llvm_unreachable("lowering LangAddressSpaceAttr NYI");
    else if (auto targetAsAttr =
                 aiir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                     addrSpaceAttr))
      numericAS = targetAsAttr.getValue();
    return aiir::LLVM::LLVMPointerType::get(type.getContext(), numericAS);
  });
  converter.addConversion([&](cir::VPtrType type) -> aiir::Type {
    assert(!cir::MissingFeatures::addressSpace());
    return aiir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](cir::ArrayType type) -> aiir::Type {
    aiir::Type ty =
        convertTypeForMemory(converter, dataLayout, type.getElementType());
    return aiir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](cir::VectorType type) -> aiir::Type {
    const aiir::Type ty = converter.convertType(type.getElementType());
    return aiir::VectorType::get(type.getSize(), ty, {type.getIsScalable()});
  });
  converter.addConversion([&](cir::BoolType type) -> aiir::Type {
    return aiir::IntegerType::get(type.getContext(), 1,
                                  aiir::IntegerType::Signless);
  });
  converter.addConversion([&](cir::IntType type) -> aiir::Type {
    // LLVM doesn't work with signed types, so we drop the CIR signs here.
    return aiir::IntegerType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([&](cir::SingleType type) -> aiir::Type {
    return aiir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> aiir::Type {
    return aiir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> aiir::Type {
    return aiir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP128Type type) -> aiir::Type {
    return aiir::Float128Type::get(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> aiir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP16Type type) -> aiir::Type {
    return aiir::Float16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> aiir::Type {
    return aiir::BFloat16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::ComplexType type) -> aiir::Type {
    // A complex type is lowered to an LLVM struct that contains the real and
    // imaginary part as data fields.
    aiir::Type elementTy = converter.convertType(type.getElementType());
    aiir::Type structFields[2] = {elementTy, elementTy};
    return aiir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                  structFields);
  });
  converter.addConversion([&](cir::FuncType type) -> std::optional<aiir::Type> {
    auto result = converter.convertType(type.getReturnType());
    llvm::SmallVector<aiir::Type> arguments;
    arguments.reserve(type.getNumInputs());
    if (converter.convertTypes(type.getInputs(), arguments).failed())
      return std::nullopt;
    auto varArg = type.isVarArg();
    return aiir::LLVM::LLVMFunctionType::get(result, arguments, varArg);
  });
  converter.addConversion([&](cir::RecordType type) -> aiir::Type {
    // Convert struct members.
    llvm::SmallVector<aiir::Type> llvmMembers;
    switch (type.getKind()) {
    case cir::RecordType::Class:
    case cir::RecordType::Struct:
      for (aiir::Type ty : type.getMembers())
        llvmMembers.push_back(convertTypeForMemory(converter, dataLayout, ty));
      break;
    // Unions are lowered as only the largest member.
    case cir::RecordType::Union:
      if (type.getMembers().empty())
        break;
      if (auto largestMember = type.getLargestMember(dataLayout))
        llvmMembers.push_back(
            convertTypeForMemory(converter, dataLayout, largestMember));
      if (type.getPadded()) {
        auto last = *type.getMembers().rbegin();
        llvmMembers.push_back(
            convertTypeForMemory(converter, dataLayout, last));
      }
      break;
    }

    // Record has a name: lower as an identified record.
    aiir::LLVM::LLVMStructType llvmStruct;
    if (type.getName()) {
      llvmStruct = aiir::LLVM::LLVMStructType::getIdentified(
          type.getContext(), type.getPrefixedName());
      if (llvmStruct.setBody(llvmMembers, type.getPacked()).failed())
        llvm_unreachable("Failed to set body of record");
    } else { // Record has no name: lower as literal record.
      llvmStruct = aiir::LLVM::LLVMStructType::getLiteral(
          type.getContext(), llvmMembers, type.getPacked());
    }

    return llvmStruct;
  });
  converter.addConversion([&](cir::VoidType type) -> aiir::Type {
    return aiir::LLVM::LLVMVoidType::get(type.getContext());
  });
}

static void buildCtorDtorList(
    aiir::ModuleOp module, StringRef globalXtorName, StringRef llvmXtorName,
    llvm::function_ref<std::pair<StringRef, int>(aiir::Attribute)> createXtor) {
  llvm::SmallVector<std::pair<StringRef, int>> globalXtors;
  for (const aiir::NamedAttribute namedAttr : module->getAttrs()) {
    if (namedAttr.getName() == globalXtorName) {
      for (auto attr : aiir::cast<aiir::ArrayAttr>(namedAttr.getValue()))
        globalXtors.emplace_back(createXtor(attr));
      break;
    }
  }

  if (globalXtors.empty())
    return;

  aiir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Create a global array llvm.global_ctors with element type of
  // struct { i32, ptr, ptr }
  auto ctorPFTy = aiir::LLVM::LLVMPointerType::get(builder.getContext());
  llvm::SmallVector<aiir::Type> ctorStructFields;
  ctorStructFields.push_back(builder.getI32Type());
  ctorStructFields.push_back(ctorPFTy);
  ctorStructFields.push_back(ctorPFTy);

  auto ctorStructTy = aiir::LLVM::LLVMStructType::getLiteral(
      builder.getContext(), ctorStructFields);
  auto ctorStructArrayTy =
      aiir::LLVM::LLVMArrayType::get(ctorStructTy, globalXtors.size());

  aiir::Location loc = module.getLoc();
  auto newGlobalOp = aiir::LLVM::GlobalOp::create(
      builder, loc, ctorStructArrayTy, /*constant=*/false,
      aiir::LLVM::Linkage::Appending, llvmXtorName, aiir::Attribute());

  builder.createBlock(&newGlobalOp.getRegion());
  builder.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());

  aiir::Value result =
      aiir::LLVM::UndefOp::create(builder, loc, ctorStructArrayTy);

  for (auto [index, fn] : llvm::enumerate(globalXtors)) {
    aiir::Value structInit =
        aiir::LLVM::UndefOp::create(builder, loc, ctorStructTy);
    aiir::Value initPriority = aiir::LLVM::ConstantOp::create(
        builder, loc, ctorStructFields[0], fn.second);
    aiir::Value initFuncAddr = aiir::LLVM::AddressOfOp::create(
        builder, loc, ctorStructFields[1], fn.first);
    aiir::Value initAssociate =
        aiir::LLVM::ZeroOp::create(builder, loc, ctorStructFields[2]);
    // Literal zero makes the InsertValueOp::create ambiguous.
    llvm::SmallVector<int64_t> zero{0};
    structInit = aiir::LLVM::InsertValueOp::create(builder, loc, structInit,
                                                   initPriority, zero);
    structInit = aiir::LLVM::InsertValueOp::create(builder, loc, structInit,
                                                   initFuncAddr, 1);
    // TODO: handle associated data for initializers.
    structInit = aiir::LLVM::InsertValueOp::create(builder, loc, structInit,
                                                   initAssociate, 2);
    result = aiir::LLVM::InsertValueOp::create(builder, loc, result, structInit,
                                               index);
  }

  aiir::LLVM::ReturnOp::create(builder, loc, result);
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
//      %4 = cir.cast int_to_bool %arg0 : !s32i -> !cir.bool
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
static void collectUnreachable(aiir::Operation *parent,
                               llvm::SmallVector<aiir::Operation *> &ops) {

  llvm::SmallVector<aiir::Block *> unreachableBlocks;
  parent->walk([&](aiir::Block *blk) { // check
    if (blk->hasNoPredecessors() && !blk->isEntryBlock())
      unreachableBlocks.push_back(blk);
  });

  std::set<aiir::Block *> visited;
  for (aiir::Block *root : unreachableBlocks) {
    // We create a work list for each unreachable block.
    // Thus we traverse operations in some order.
    std::deque<aiir::Block *> workList;
    workList.push_back(root);

    while (!workList.empty()) {
      aiir::Block *blk = workList.back();
      workList.pop_back();
      if (visited.count(blk))
        continue;
      visited.emplace(blk);

      for (aiir::Operation &op : *blk)
        ops.push_back(&op);

      for (aiir::Block *succ : blk->getSuccessors())
        workList.push_back(succ);
    }
  }
}

aiir::LogicalResult CIRToLLVMObjSizeOpLowering::matchAndRewrite(
    cir::ObjSizeOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type llvmResTy = getTypeConverter()->convertType(op.getType());
  aiir::Location loc = op->getLoc();

  aiir::IntegerType i1Ty = rewriter.getI1Type();

  auto i1Val = [&rewriter, &loc, &i1Ty](bool val) {
    return aiir::LLVM::ConstantOp::create(rewriter, loc, i1Ty, val);
  };

  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.objectsize", llvmResTy,
                                   {
                                       adaptor.getPtr(),
                                       i1Val(op.getMin()),
                                       i1Val(op.getNullunknown()),
                                       i1Val(op.getDynamic()),
                                   });

  return aiir::LogicalResult::success();
}

void ConvertCIRToLLVMPass::resolveBlockAddressOp(
    LLVMBlockAddressInfo &blockInfoAddr) {

  aiir::ModuleOp module = getOperation();
  aiir::OpBuilder opBuilder(module.getContext());
  for (auto &[blockAddOp, blockInfo] :
       blockInfoAddr.getUnresolvedBlockAddress()) {
    aiir::LLVM::BlockTagOp resolvedLabel =
        blockInfoAddr.lookupBlockTag(blockInfo);
    assert(resolvedLabel && "expected BlockTagOp to already be emitted");
    aiir::FlatSymbolRefAttr fnSym = blockInfo.getFunc();
    auto blkAddTag = aiir::LLVM::BlockAddressAttr::get(
        opBuilder.getContext(), fnSym, resolvedLabel.getTagAttr());
    blockAddOp.setBlockAddrAttr(blkAddTag);
  }
  blockInfoAddr.clearUnresolvedMap();
}

void ConvertCIRToLLVMPass::processCIRAttrs(aiir::ModuleOp module) {
  // Lower the module attributes to LLVM equivalents.
  if (aiir::Attribute tripleAttr =
          module->getAttr(cir::CIRDialect::getTripleAttrName()))
    module->setAttr(aiir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    tripleAttr);

  if (aiir::Attribute asmAttr =
          module->getAttr(cir::CIRDialect::getModuleLevelAsmAttrName()))
    module->setAttr(aiir::LLVM::LLVMDialect::getModuleLevelAsmAttrName(),
                    asmAttr);
}

void ConvertCIRToLLVMPass::runOnOperation() {
  llvm::TimeTraceScope scope("Convert CIR to LLVM Pass");

  aiir::ModuleOp module = getOperation();
  aiir::DataLayout dl(module);
  aiir::LLVMTypeConverter converter(&getContext());
  prepareTypeConverter(converter, dl);

  /// Tracks the state required to lower CIR `LabelOp` and `BlockAddressOp`.
  /// Maps labels to their corresponding `BlockTagOp` and keeps bookkeeping
  /// of unresolved `BlockAddressOp`s until they are matched with the
  /// corresponding `BlockTagOp` in `resolveBlockAddressOp`.
  LLVMBlockAddressInfo blockInfoAddr;
  aiir::RewritePatternSet patterns(&getContext());
  patterns.add<CIRToLLVMBlockAddressOpLowering, CIRToLLVMLabelOpLowering>(
      converter, patterns.getContext(), dl, blockInfoAddr);

  patterns.add<
#define GET_LLVM_LOWERING_PATTERNS_LIST
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef GET_LLVM_LOWERING_PATTERNS_LIST
      >(converter, patterns.getContext(), dl);

  processCIRAttrs(module);

  aiir::ConversionTarget target(getContext());
  target.addLegalOp<aiir::ModuleOp>();
  target.addLegalDialect<aiir::LLVM::LLVMDialect>();
  target.addIllegalDialect<aiir::BuiltinDialect, cir::CIRDialect,
                           aiir::func::FuncDialect>();

  llvm::SmallVector<aiir::Operation *> ops;
  ops.push_back(module);
  collectUnreachable(module, ops);

  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();

  // Emit the llvm.global_ctors array.
  buildCtorDtorList(module, cir::CIRDialect::getGlobalCtorsAttrName(),
                    "llvm.global_ctors", [](aiir::Attribute attr) {
                      auto ctorAttr = aiir::cast<cir::GlobalCtorAttr>(attr);
                      return std::make_pair(ctorAttr.getName(),
                                            ctorAttr.getPriority());
                    });
  // Emit the llvm.global_dtors array.
  buildCtorDtorList(module, cir::CIRDialect::getGlobalDtorsAttrName(),
                    "llvm.global_dtors", [](aiir::Attribute attr) {
                      auto dtorAttr = aiir::cast<cir::GlobalDtorAttr>(attr);
                      return std::make_pair(dtorAttr.getName(),
                                            dtorAttr.getPriority());
                    });
  resolveBlockAddressOp(blockInfoAddr);
}

aiir::LogicalResult CIRToLLVMBrOpLowering::matchAndRewrite(
    cir::BrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::BrOp>(op, adaptor.getOperands(),
                                                op.getDest());
  return aiir::LogicalResult::success();
}

aiir::LogicalResult CIRToLLVMGetMemberOpLowering::matchAndRewrite(
    cir::GetMemberOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type llResTy = getTypeConverter()->convertType(op.getType());
  const auto recordTy =
      aiir::cast<cir::RecordType>(op.getAddrTy().getPointee());
  assert(recordTy && "expected record type");

  switch (recordTy.getKind()) {
  case cir::RecordType::Class:
  case cir::RecordType::Struct: {
    // Since the base address is a pointer to an aggregate, the first offset
    // is always zero. The second offset tell us which member it will access.
    llvm::SmallVector<aiir::LLVM::GEPArg, 2> offset{0, op.getIndex()};
    const aiir::Type elementTy = getTypeConverter()->convertType(recordTy);
    rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(op, llResTy, elementTy,
                                                   adaptor.getAddr(), offset);
    return aiir::success();
  }
  case cir::RecordType::Union:
    // Union members share the address space, so we just need a bitcast to
    // conform to type-checking.
    rewriter.replaceOpWithNewOp<aiir::LLVM::BitcastOp>(op, llResTy,
                                                       adaptor.getAddr());
    return aiir::success();
  }
}

aiir::LogicalResult CIRToLLVMExtractMemberOpLowering::matchAndRewrite(
    cir::ExtractMemberOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  std::int64_t indices[1] = {static_cast<std::int64_t>(op.getIndex())};

  aiir::Type recordTy = op.getRecord().getType();
  auto cirRecordTy = aiir::cast<cir::RecordType>(recordTy);
  switch (cirRecordTy.getKind()) {
  case cir::RecordType::Struct:
  case cir::RecordType::Class:
    rewriter.replaceOpWithNewOp<aiir::LLVM::ExtractValueOp>(
        op, adaptor.getRecord(), indices);
    return aiir::success();

  case cir::RecordType::Union:
    op.emitError("cir.extract_member cannot extract member from a union");
    return aiir::failure();
  }
  llvm_unreachable("Unexpected record kind");
}

aiir::LogicalResult CIRToLLVMInsertMemberOpLowering::matchAndRewrite(
    cir::InsertMemberOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  std::int64_t indecies[1] = {static_cast<std::int64_t>(op.getIndex())};
  aiir::Type recordTy = op.getRecord().getType();

  if (auto cirRecordTy = aiir::dyn_cast<cir::RecordType>(recordTy)) {
    if (cirRecordTy.getKind() == cir::RecordType::Union) {
      op.emitError("cir.update_member cannot update member of a union");
      return aiir::failure();
    }
  }

  rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(
      op, adaptor.getRecord(), adaptor.getValue(), indecies);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMUnreachableOpLowering::matchAndRewrite(
    cir::UnreachableOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::UnreachableOp>(op);
  return aiir::success();
}

void createLLVMFuncOpIfNotExist(aiir::ConversionPatternRewriter &rewriter,
                                aiir::Operation *srcOp, llvm::StringRef fnName,
                                aiir::Type fnTy) {
  auto modOp = srcOp->getParentOfType<aiir::ModuleOp>();
  auto enclosingFnOp = srcOp->getParentOfType<aiir::LLVM::LLVMFuncOp>();
  aiir::Operation *sourceSymbol =
      aiir::SymbolTable::lookupSymbolIn(modOp, fnName);
  if (!sourceSymbol) {
    aiir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(enclosingFnOp);
    aiir::LLVM::LLVMFuncOp::create(rewriter, srcOp->getLoc(), fnName, fnTy);
  }
}

aiir::LogicalResult CIRToLLVMThrowOpLowering::matchAndRewrite(
    cir::ThrowOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Location loc = op.getLoc();
  auto voidTy = aiir::LLVM::LLVMVoidType::get(getContext());

  if (op.rethrows()) {
    auto funcTy = aiir::LLVM::LLVMFunctionType::get(voidTy, {});

    // Get or create `declare void @__cxa_rethrow()`
    const llvm::StringRef functionName = "__cxa_rethrow";
    createLLVMFuncOpIfNotExist(rewriter, op, functionName, funcTy);

    auto cxaRethrow = aiir::LLVM::CallOp::create(
        rewriter, loc, aiir::TypeRange{}, functionName);

    rewriter.replaceOp(op, cxaRethrow);
    return aiir::success();
  }

  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto fnTy = aiir::LLVM::LLVMFunctionType::get(
      voidTy, {llvmPtrTy, llvmPtrTy, llvmPtrTy});

  // Get or create `declare void @__cxa_throw(ptr, ptr, ptr)`
  const llvm::StringRef fnName = "__cxa_throw";
  createLLVMFuncOpIfNotExist(rewriter, op, fnName, fnTy);

  aiir::Value typeInfo = aiir::LLVM::AddressOfOp::create(
      rewriter, loc, aiir::LLVM::LLVMPointerType::get(rewriter.getContext()),
      adaptor.getTypeInfoAttr());

  aiir::Value dtor;
  if (op.getDtor()) {
    dtor = aiir::LLVM::AddressOfOp::create(rewriter, loc, llvmPtrTy,
                                           adaptor.getDtorAttr());
  } else {
    dtor = aiir::LLVM::ZeroOp::create(rewriter, loc, llvmPtrTy);
  }

  auto cxaThrowCall = aiir::LLVM::CallOp::create(
      rewriter, loc, aiir::TypeRange{}, fnName,
      aiir::ValueRange{adaptor.getExceptionPtr(), typeInfo, dtor});

  rewriter.replaceOp(op, cxaThrowCall);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAllocExceptionOpLowering::matchAndRewrite(
    cir::AllocExceptionOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // Get or create `declare ptr @__cxa_allocate_exception(i64)`
  StringRef fnName = "__cxa_allocate_exception";
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto int64Ty = aiir::IntegerType::get(rewriter.getContext(), 64);
  auto fnTy = aiir::LLVM::LLVMFunctionType::get(llvmPtrTy, {int64Ty});

  createLLVMFuncOpIfNotExist(rewriter, op, fnName, fnTy);
  auto exceptionSize = aiir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                                      adaptor.getSizeAttr());

  auto allocaExceptionCall = aiir::LLVM::CallOp::create(
      rewriter, op.getLoc(), aiir::TypeRange{llvmPtrTy}, fnName,
      aiir::ValueRange{exceptionSize});

  rewriter.replaceOp(op, allocaExceptionCall);
  return aiir::success();
}

static aiir::LLVM::LLVMStructType
getLLVMLandingPadStructTy(aiir::ConversionPatternRewriter &rewriter) {
  // Create the landing pad type: struct { ptr, i32 }
  aiir::AIIRContext *ctx = rewriter.getContext();
  auto llvmPtr = aiir::LLVM::LLVMPointerType::get(ctx);
  llvm::SmallVector<aiir::Type> structFields = {llvmPtr, rewriter.getI32Type()};
  return aiir::LLVM::LLVMStructType::getLiteral(ctx, structFields);
}

aiir::LogicalResult CIRToLLVMEhInflightOpLowering::matchAndRewrite(
    cir::EhInflightOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto llvmFn = op->getParentOfType<aiir::LLVM::LLVMFuncOp>();
  assert(llvmFn && "expected LLVM function parent");
  aiir::Block *entryBlock = &llvmFn.getRegion().front();
  assert(entryBlock->isEntryBlock());

  aiir::ArrayAttr catchListAttr = op.getCatchTypeListAttr();
  aiir::SmallVector<aiir::Value> catchSymAddrs;

  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  aiir::Location loc = op.getLoc();

  // %landingpad = landingpad { ptr, i32 }
  // Note that since llvm.landingpad has to be the first operation on the
  // block, any needed value for its operands has to be added somewhere else.
  if (catchListAttr) {
    //   catch ptr @_ZTIi
    //   catch ptr @_ZTIPKc
    for (aiir::Attribute catchAttr : catchListAttr) {
      auto symAttr = cast<aiir::FlatSymbolRefAttr>(catchAttr);
      // Generate `llvm.aiir.addressof` for each symbol, and place those
      // operations in the LLVM function entry basic block.
      aiir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);
      aiir::Value addrOp = aiir::LLVM::AddressOfOp::create(
          rewriter, loc, llvmPtrTy, symAttr.getValue());
      catchSymAddrs.push_back(addrOp);
    }
  } else if (!op.getCleanup()) {
    // We need to emit catch-all only if cleanup is not set, because when we
    // have catch-all handler, there is no case when we set would unwind past
    // the handler
    aiir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);
    aiir::Value nullOp = aiir::LLVM::ZeroOp::create(rewriter, loc, llvmPtrTy);
    catchSymAddrs.push_back(nullOp);
  }

  // %slot = extractvalue { ptr, i32 } %x, 0
  // %selector = extractvalue { ptr, i32 } %x, 1
  aiir::LLVM::LLVMStructType llvmLandingPadStructTy =
      getLLVMLandingPadStructTy(rewriter);
  auto landingPadOp = aiir::LLVM::LandingpadOp::create(
      rewriter, loc, llvmLandingPadStructTy, catchSymAddrs);

  if (op.getCleanup())
    landingPadOp.setCleanup(true);

  aiir::Value slot =
      aiir::LLVM::ExtractValueOp::create(rewriter, loc, landingPadOp, 0);
  aiir::Value selector =
      aiir::LLVM::ExtractValueOp::create(rewriter, loc, landingPadOp, 1);
  rewriter.replaceOp(op, aiir::ValueRange{slot, selector});

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMResumeFlatOpLowering::matchAndRewrite(
    cir::ResumeFlatOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // %lpad.val = insertvalue { ptr, i32 } poison, ptr %exception_ptr, 0
  // %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %selector, 1
  // resume { ptr, i32 } %lpad.val2
  aiir::Type llvmLandingPadStructTy = getLLVMLandingPadStructTy(rewriter);
  aiir::Value poison = aiir::LLVM::PoisonOp::create(rewriter, op.getLoc(),
                                                    llvmLandingPadStructTy);

  SmallVector<int64_t> slotIdx = {0};
  aiir::Value slot = aiir::LLVM::InsertValueOp::create(
      rewriter, op.getLoc(), poison, adaptor.getExceptionPtr(), slotIdx);

  SmallVector<int64_t> selectorIdx = {1};
  aiir::Value selector = aiir::LLVM::InsertValueOp::create(
      rewriter, op.getLoc(), slot, adaptor.getTypeId(), selectorIdx);

  rewriter.replaceOpWithNewOp<aiir::LLVM::ResumeOp>(op, selector);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMEhTypeIdOpLowering::matchAndRewrite(
    cir::EhTypeIdOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value addrOp = aiir::LLVM::AddressOfOp::create(
      rewriter, op.getLoc(),
      aiir::LLVM::LLVMPointerType::get(rewriter.getContext()),
      op.getTypeSymAttr());
  rewriter.replaceOpWithNewOp<aiir::LLVM::EhTypeidForOp>(
      op, rewriter.getI32Type(), addrOp);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMEhSetjmpOpLowering::matchAndRewrite(
    cir::EhSetjmpOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type returnType = typeConverter->convertType(op.getType());
  aiir::LLVM::CallIntrinsicOp newOp =
      createCallLLVMIntrinsicOp(rewriter, op.getLoc(), "llvm.eh.sjlj.setjmp",
                                returnType, adaptor.getEnv());
  rewriter.replaceOp(op, newOp);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMEhLongjmpOpLowering::matchAndRewrite(
    cir::EhLongjmpOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.eh.sjlj.longjmp",
                                   /*resultTy=*/{}, adaptor.getOperands());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMTrapOpLowering::matchAndRewrite(
    cir::TrapOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Location loc = op->getLoc();
  rewriter.eraseOp(op);

  aiir::LLVM::Trap::create(rewriter, loc);

  // Note that the call to llvm.trap is not a terminator in LLVM dialect.
  // So we must emit an additional llvm.unreachable to terminate the current
  // block.
  aiir::LLVM::UnreachableOp::create(rewriter, loc);

  return aiir::success();
}

static aiir::Value
getValueForVTableSymbol(aiir::Operation *op,
                        aiir::ConversionPatternRewriter &rewriter,
                        const aiir::TypeConverter *converter,
                        aiir::FlatSymbolRefAttr nameAttr, aiir::Type &eltType) {
  auto module = op->getParentOfType<aiir::ModuleOp>();
  aiir::Operation *symbol = aiir::SymbolTable::lookupSymbolIn(module, nameAttr);
  if (auto llvmSymbol = aiir::dyn_cast<aiir::LLVM::GlobalOp>(symbol)) {
    eltType = llvmSymbol.getType();
  } else if (auto cirSymbol = aiir::dyn_cast<cir::GlobalOp>(symbol)) {
    eltType = converter->convertType(cirSymbol.getSymType());
  } else {
    op->emitError() << "unexpected symbol type for " << symbol;
    return {};
  }

  return aiir::LLVM::AddressOfOp::create(
      rewriter, op->getLoc(),
      aiir::LLVM::LLVMPointerType::get(op->getContext()), nameAttr.getValue());
}

aiir::LogicalResult CIRToLLVMVTableAddrPointOpLowering::matchAndRewrite(
    cir::VTableAddrPointOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::TypeConverter *converter = getTypeConverter();
  aiir::Type targetType = converter->convertType(op.getType());
  llvm::SmallVector<aiir::LLVM::GEPArg> offsets;
  aiir::Type eltType;
  aiir::Value symAddr = getValueForVTableSymbol(op, rewriter, converter,
                                                op.getNameAttr(), eltType);
  if (!symAddr)
    return op.emitError() << "Unable to get value for vtable symbol";

  offsets = llvm::SmallVector<aiir::LLVM::GEPArg>{
      0, op.getAddressPointAttr().getIndex(),
      op.getAddressPointAttr().getOffset()};

  assert(eltType && "Shouldn't ever be missing an eltType here");
  aiir::LLVM::GEPNoWrapFlags inboundsNuw =
      aiir::LLVM::GEPNoWrapFlags::inbounds | aiir::LLVM::GEPNoWrapFlags::nuw;
  rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(op, targetType, eltType,
                                                 symAddr, offsets, inboundsNuw);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVTableGetVPtrOpLowering::matchAndRewrite(
    cir::VTableGetVPtrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // cir.vtable.get_vptr is equivalent to a bitcast from the source object
  // pointer to the vptr type. Since the LLVM dialect uses opaque pointers
  // we can just replace uses of this operation with the original pointer.
  aiir::Value srcVal = adaptor.getSrc();
  rewriter.replaceOp(op, srcVal);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVTableGetVirtualFnAddrOpLowering::matchAndRewrite(
    cir::VTableGetVirtualFnAddrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type targetType = getTypeConverter()->convertType(op.getType());
  auto eltType = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  llvm::SmallVector<aiir::LLVM::GEPArg> offsets =
      llvm::SmallVector<aiir::LLVM::GEPArg>{op.getIndex()};
  rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
      op, targetType, eltType, adaptor.getVptr(), offsets,
      aiir::LLVM::GEPNoWrapFlags::inbounds);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVTTAddrPointOpLowering::matchAndRewrite(
    cir::VTTAddrPointOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type resultType = getTypeConverter()->convertType(op.getType());
  llvm::SmallVector<aiir::LLVM::GEPArg> offsets;
  aiir::Type eltType;
  aiir::Value llvmAddr = adaptor.getSymAddr();

  if (op.getSymAddr()) {
    if (op.getOffset() == 0) {
      rewriter.replaceOp(op, {llvmAddr});
      return aiir::success();
    }

    offsets.push_back(adaptor.getOffset());
    eltType = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  } else {
    llvmAddr = getValueForVTableSymbol(op, rewriter, getTypeConverter(),
                                       op.getNameAttr(), eltType);
    assert(eltType && "Shouldn't ever be missing an eltType here");
    offsets.push_back(0);
    offsets.push_back(adaptor.getOffset());
  }
  rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
      op, resultType, eltType, llvmAddr, offsets,
      aiir::LLVM::GEPNoWrapFlags::inbounds);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMStackSaveOpLowering::matchAndRewrite(
    cir::StackSaveOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  const aiir::Type ptrTy = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<aiir::LLVM::StackSaveOp>(op, ptrTy);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMStackRestoreOpLowering::matchAndRewrite(
    cir::StackRestoreOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::StackRestoreOp>(op, adaptor.getPtr());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecCreateOpLowering::matchAndRewrite(
    cir::VecCreateOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // Start with an 'undef' value for the vector.  Then 'insertelement' for
  // each of the vector elements.
  const auto vecTy = aiir::cast<cir::VectorType>(op.getType());
  const aiir::Type llvmTy = typeConverter->convertType(vecTy);
  const aiir::Location loc = op.getLoc();
  aiir::Value result = aiir::LLVM::PoisonOp::create(rewriter, loc, llvmTy);
  assert(vecTy.getSize() == op.getElements().size() &&
         "cir.vec.create op count doesn't match vector type elements count");

  for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
    const aiir::Value indexValue =
        aiir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), i);
    result = aiir::LLVM::InsertElementOp::create(
        rewriter, loc, result, adaptor.getElements()[i], indexValue);
  }

  rewriter.replaceOp(op, result);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecExtractOpLowering::matchAndRewrite(
    cir::VecExtractOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::ExtractElementOp>(
      op, adaptor.getVec(), adaptor.getIndex());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecInsertOpLowering::matchAndRewrite(
    cir::VecInsertOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::InsertElementOp>(
      op, adaptor.getVec(), adaptor.getValue(), adaptor.getIndex());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecCmpOpLowering::matchAndRewrite(
    cir::VecCmpOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type elementType = elementTypeIfVector(op.getLhs().getType());
  aiir::Value bitResult;
  if (auto intType = aiir::dyn_cast<cir::IntType>(elementType)) {
    bitResult = aiir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(),
        convertCmpKindToICmpPredicate(op.getKind(), intType.isSigned()),
        adaptor.getLhs(), adaptor.getRhs());
  } else if (aiir::isa<cir::FPTypeInterface>(elementType)) {
    bitResult = aiir::LLVM::FCmpOp::create(
        rewriter, op.getLoc(), convertCmpKindToFCmpPredicate(op.getKind()),
        adaptor.getLhs(), adaptor.getRhs());
  } else {
    return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
  }

  // LLVM IR vector comparison returns a vector of i1. This one-bit vector
  // must be sign-extended to the correct result type, unless a vector of i1 is
  // the type we need.
  if (cast<cir::IntType>(cast<cir::VectorType>(op.getType()).getElementType())
          .getWidth() > 1)
    rewriter.replaceOpWithNewOp<aiir::LLVM::SExtOp>(
        op, typeConverter->convertType(op.getType()), bitResult);
  else
    rewriter.replaceOp(op, bitResult);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecSplatOpLowering::matchAndRewrite(
    cir::VecSplatOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // Vector splat can be implemented with an `insertelement` and a
  // `shufflevector`, which is better than an `insertelement` for each
  // element in the vector. Start with an undef vector. Insert the value into
  // the first element. Then use a `shufflevector` with a mask of all 0 to
  // fill out the entire vector with that value.
  cir::VectorType vecTy = op.getType();
  aiir::Type llvmTy = typeConverter->convertType(vecTy);
  aiir::Location loc = op.getLoc();
  aiir::Value poison = aiir::LLVM::PoisonOp::create(rewriter, loc, llvmTy);

  aiir::Value elementValue = adaptor.getValue();
  if (elementValue.getDefiningOp<aiir::LLVM::PoisonOp>()) {
    // If the splat value is poison, then we can just use poison value
    // for the entire vector.
    rewriter.replaceOp(op, poison);
    return aiir::success();
  }

  if (auto constValue = elementValue.getDefiningOp<aiir::LLVM::ConstantOp>()) {
    if (auto intAttr = dyn_cast<aiir::IntegerAttr>(constValue.getValue())) {
      aiir::DenseIntElementsAttr denseVec = aiir::DenseIntElementsAttr::get(
          aiir::cast<aiir::ShapedType>(llvmTy), intAttr.getValue());
      rewriter.replaceOpWithNewOp<aiir::LLVM::ConstantOp>(
          op, denseVec.getType(), denseVec);
      return aiir::success();
    }

    if (auto fpAttr = dyn_cast<aiir::FloatAttr>(constValue.getValue())) {
      aiir::DenseFPElementsAttr denseVec = aiir::DenseFPElementsAttr::get(
          aiir::cast<aiir::ShapedType>(llvmTy), fpAttr.getValue());
      rewriter.replaceOpWithNewOp<aiir::LLVM::ConstantOp>(
          op, denseVec.getType(), denseVec);
      return aiir::success();
    }
  }

  aiir::Value indexValue =
      aiir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 0);
  aiir::Value oneElement = aiir::LLVM::InsertElementOp::create(
      rewriter, loc, poison, elementValue, indexValue);
  SmallVector<int32_t> zeroValues(vecTy.getSize(), 0);
  rewriter.replaceOpWithNewOp<aiir::LLVM::ShuffleVectorOp>(op, oneElement,
                                                           poison, zeroValues);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecShuffleOpLowering::matchAndRewrite(
    cir::VecShuffleOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // LLVM::ShuffleVectorOp takes an ArrayRef of int for the list of indices.
  // Convert the ClangIR ArrayAttr of IntAttr constants into a
  // SmallVector<int>.
  SmallVector<int, 8> indices;
  std::transform(
      op.getIndices().begin(), op.getIndices().end(),
      std::back_inserter(indices), [](aiir::Attribute intAttr) {
        return aiir::cast<cir::IntAttr>(intAttr).getValue().getSExtValue();
      });
  rewriter.replaceOpWithNewOp<aiir::LLVM::ShuffleVectorOp>(
      op, adaptor.getVec1(), adaptor.getVec2(), indices);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecShuffleDynamicOpLowering::matchAndRewrite(
    cir::VecShuffleDynamicOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // LLVM IR does not have an operation that corresponds to this form of
  // the built-in.
  //     __builtin_shufflevector(V, I)
  // is implemented as this pseudocode, where the for loop is unrolled
  // and N is the number of elements:
  //
  // result = undef
  // maskbits = NextPowerOf2(N - 1)
  // masked = I & maskbits
  // for (i in 0 <= i < N)
  //    result[i] = V[masked[i]]
  aiir::Location loc = op.getLoc();
  aiir::Value input = adaptor.getVec();
  aiir::Type llvmIndexVecType =
      getTypeConverter()->convertType(op.getIndices().getType());
  aiir::Type llvmIndexType = getTypeConverter()->convertType(
      elementTypeIfVector(op.getIndices().getType()));
  uint64_t numElements =
      aiir::cast<cir::VectorType>(op.getVec().getType()).getSize();

  uint64_t maskBits = llvm::NextPowerOf2(numElements - 1) - 1;
  aiir::Value maskValue = aiir::LLVM::ConstantOp::create(
      rewriter, loc, llvmIndexType,
      rewriter.getIntegerAttr(llvmIndexType, maskBits));
  aiir::Value maskVector =
      aiir::LLVM::UndefOp::create(rewriter, loc, llvmIndexVecType);

  for (uint64_t i = 0; i < numElements; ++i) {
    aiir::Value idxValue =
        aiir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), i);
    maskVector = aiir::LLVM::InsertElementOp::create(rewriter, loc, maskVector,
                                                     maskValue, idxValue);
  }

  aiir::Value maskedIndices = aiir::LLVM::AndOp::create(
      rewriter, loc, llvmIndexVecType, adaptor.getIndices(), maskVector);
  aiir::Value result = aiir::LLVM::UndefOp::create(
      rewriter, loc, getTypeConverter()->convertType(op.getVec().getType()));
  for (uint64_t i = 0; i < numElements; ++i) {
    aiir::Value iValue =
        aiir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), i);
    aiir::Value indexValue = aiir::LLVM::ExtractElementOp::create(
        rewriter, loc, maskedIndices, iValue);
    aiir::Value valueAtIndex =
        aiir::LLVM::ExtractElementOp::create(rewriter, loc, input, indexValue);
    result = aiir::LLVM::InsertElementOp::create(rewriter, loc, result,
                                                 valueAtIndex, iValue);
  }
  rewriter.replaceOp(op, result);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVecTernaryOpLowering::matchAndRewrite(
    cir::VecTernaryOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  // Convert `cond` into a vector of i1, then use that in a `select` op.
  aiir::Value bitVec = aiir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(), aiir::LLVM::ICmpPredicate::ne, adaptor.getCond(),
      aiir::LLVM::ZeroOp::create(
          rewriter, op.getCond().getLoc(),
          typeConverter->convertType(op.getCond().getType())));
  rewriter.replaceOpWithNewOp<aiir::LLVM::SelectOp>(
      op, bitVec, adaptor.getLhs(), adaptor.getRhs());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMComplexAddOpLowering::matchAndRewrite(
    cir::ComplexAddOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value lhs = adaptor.getLhs();
  aiir::Value rhs = adaptor.getRhs();
  aiir::Location loc = op.getLoc();

  auto complexType = aiir::cast<cir::ComplexType>(op.getLhs().getType());
  aiir::Type complexElemTy =
      getTypeConverter()->convertType(complexType.getElementType());
  auto lhsReal = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, lhs, ArrayRef(int64_t{0}));
  auto lhsImag = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, lhs, ArrayRef(int64_t{1}));
  auto rhsReal = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, rhs, ArrayRef(int64_t{0}));
  auto rhsImag = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, rhs, ArrayRef(int64_t{1}));

  aiir::Value newReal;
  aiir::Value newImag;
  if (complexElemTy.isInteger()) {
    newReal = aiir::LLVM::AddOp::create(rewriter, loc, complexElemTy, lhsReal,
                                        rhsReal);
    newImag = aiir::LLVM::AddOp::create(rewriter, loc, complexElemTy, lhsImag,
                                        rhsImag);
  } else {
    assert(!cir::MissingFeatures::fastMathFlags());
    assert(!cir::MissingFeatures::fpConstraints());
    newReal = aiir::LLVM::FAddOp::create(rewriter, loc, complexElemTy, lhsReal,
                                         rhsReal);
    newImag = aiir::LLVM::FAddOp::create(rewriter, loc, complexElemTy, lhsImag,
                                         rhsImag);
  }

  aiir::Type complexLLVMTy =
      getTypeConverter()->convertType(op.getResult().getType());
  auto initialComplex =
      aiir::LLVM::PoisonOp::create(rewriter, op->getLoc(), complexLLVMTy);

  auto realComplex = aiir::LLVM::InsertValueOp::create(
      rewriter, op->getLoc(), initialComplex, newReal, ArrayRef(int64_t{0}));

  rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(
      op, realComplex, newImag, ArrayRef(int64_t{1}));

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMComplexCreateOpLowering::matchAndRewrite(
    cir::ComplexCreateOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type complexLLVMTy =
      getTypeConverter()->convertType(op.getResult().getType());
  auto initialComplex =
      aiir::LLVM::UndefOp::create(rewriter, op->getLoc(), complexLLVMTy);

  auto realComplex = aiir::LLVM::InsertValueOp::create(
      rewriter, op->getLoc(), initialComplex, adaptor.getReal(),
      ArrayRef(int64_t{0}));

  auto complex = aiir::LLVM::InsertValueOp::create(
      rewriter, op->getLoc(), realComplex, adaptor.getImag(),
      ArrayRef(int64_t{1}));

  rewriter.replaceOp(op, complex);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMComplexRealOpLowering::matchAndRewrite(
    cir::ComplexRealOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resultLLVMTy = getTypeConverter()->convertType(op.getType());
  aiir::Value operand = adaptor.getOperand();
  if (aiir::isa<cir::ComplexType>(op.getOperand().getType())) {
    operand = aiir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), resultLLVMTy, operand,
        llvm::ArrayRef<std::int64_t>{0});
  }
  rewriter.replaceOp(op, operand);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMComplexSubOpLowering::matchAndRewrite(
    cir::ComplexSubOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value lhs = adaptor.getLhs();
  aiir::Value rhs = adaptor.getRhs();
  aiir::Location loc = op.getLoc();

  auto complexType = aiir::cast<cir::ComplexType>(op.getLhs().getType());
  aiir::Type complexElemTy =
      getTypeConverter()->convertType(complexType.getElementType());
  auto lhsReal = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, lhs, ArrayRef(int64_t{0}));
  auto lhsImag = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, lhs, ArrayRef(int64_t{1}));
  auto rhsReal = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, rhs, ArrayRef(int64_t{0}));
  auto rhsImag = aiir::LLVM::ExtractValueOp::create(
      rewriter, loc, complexElemTy, rhs, ArrayRef(int64_t{1}));

  aiir::Value newReal;
  aiir::Value newImag;
  if (complexElemTy.isInteger()) {
    newReal = aiir::LLVM::SubOp::create(rewriter, loc, complexElemTy, lhsReal,
                                        rhsReal);
    newImag = aiir::LLVM::SubOp::create(rewriter, loc, complexElemTy, lhsImag,
                                        rhsImag);
  } else {
    assert(!cir::MissingFeatures::fastMathFlags());
    assert(!cir::MissingFeatures::fpConstraints());
    newReal = aiir::LLVM::FSubOp::create(rewriter, loc, complexElemTy, lhsReal,
                                         rhsReal);
    newImag = aiir::LLVM::FSubOp::create(rewriter, loc, complexElemTy, lhsImag,
                                         rhsImag);
  }

  aiir::Type complexLLVMTy =
      getTypeConverter()->convertType(op.getResult().getType());
  auto initialComplex =
      aiir::LLVM::PoisonOp::create(rewriter, op->getLoc(), complexLLVMTy);

  auto realComplex = aiir::LLVM::InsertValueOp::create(
      rewriter, op->getLoc(), initialComplex, newReal, ArrayRef(int64_t{0}));

  rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(
      op, realComplex, newImag, ArrayRef(int64_t{1}));

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMComplexImagOpLowering::matchAndRewrite(
    cir::ComplexImagOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resultLLVMTy = getTypeConverter()->convertType(op.getType());
  aiir::Value operand = adaptor.getOperand();
  aiir::Location loc = op.getLoc();

  if (aiir::isa<cir::ComplexType>(op.getOperand().getType())) {
    operand = aiir::LLVM::ExtractValueOp::create(
        rewriter, loc, resultLLVMTy, operand, llvm::ArrayRef<std::int64_t>{1});
  } else {
    aiir::TypedAttr zeroAttr = rewriter.getZeroAttr(resultLLVMTy);
    operand =
        aiir::LLVM::ConstantOp::create(rewriter, loc, resultLLVMTy, zeroAttr);
  }

  rewriter.replaceOp(op, operand);
  return aiir::success();
}

aiir::IntegerType computeBitfieldIntType(aiir::Type storageType,
                                         aiir::AIIRContext *context,
                                         unsigned &storageSize) {
  return TypeSwitch<aiir::Type, aiir::IntegerType>(storageType)
      .Case<cir::ArrayType>([&](cir::ArrayType atTy) {
        storageSize = atTy.getSize() * 8;
        return aiir::IntegerType::get(context, storageSize);
      })
      .Case<cir::IntType>([&](cir::IntType intTy) {
        storageSize = intTy.getWidth();
        return aiir::IntegerType::get(context, storageSize);
      })
      .Default([](aiir::Type) -> aiir::IntegerType {
        llvm_unreachable(
            "Either ArrayType or IntType expected for bitfields storage");
      });
}

aiir::LogicalResult CIRToLLVMSetBitfieldOpLowering::matchAndRewrite(
    cir::SetBitfieldOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  cir::BitfieldInfoAttr info = op.getBitfieldInfo();
  uint64_t size = info.getSize();
  uint64_t offset = info.getOffset();
  aiir::Type storageType = info.getStorageType();
  aiir::AIIRContext *context = storageType.getContext();

  unsigned storageSize = 0;

  aiir::IntegerType intType =
      computeBitfieldIntType(storageType, context, storageSize);

  aiir::Value srcVal = createIntCast(rewriter, adaptor.getSrc(), intType);
  unsigned srcWidth = storageSize;
  aiir::Value resultVal = srcVal;

  if (storageSize != size) {
    assert(storageSize > size && "Invalid bitfield size.");

    aiir::Value val = aiir::LLVM::LoadOp::create(
        rewriter, op.getLoc(), intType, adaptor.getAddr(), op.getAlignment(),
        op.getIsVolatile());

    srcVal =
        createAnd(rewriter, srcVal, llvm::APInt::getLowBitsSet(srcWidth, size));
    resultVal = srcVal;
    srcVal = createShL(rewriter, srcVal, offset);

    // Mask out the original value.
    val = createAnd(rewriter, val,
                    ~llvm::APInt::getBitsSet(srcWidth, offset, offset + size));

    // Or together the unchanged values and the source value.
    srcVal = aiir::LLVM::OrOp::create(rewriter, op.getLoc(), val, srcVal);
  }

  aiir::LLVM::StoreOp::create(rewriter, op.getLoc(), srcVal, adaptor.getAddr(),
                              op.getAlignment(), op.getIsVolatile());

  aiir::Type resultTy = getTypeConverter()->convertType(op.getType());

  if (info.getIsSigned()) {
    assert(size <= storageSize);
    unsigned highBits = storageSize - size;

    if (highBits) {
      resultVal = createShL(rewriter, resultVal, highBits);
      resultVal = createAShR(rewriter, resultVal, highBits);
    }
  }

  resultVal = createIntCast(rewriter, resultVal,
                            aiir::cast<aiir::IntegerType>(resultTy),
                            info.getIsSigned());

  rewriter.replaceOp(op, resultVal);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMComplexImagPtrOpLowering::matchAndRewrite(
    cir::ComplexImagPtrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  cir::PointerType operandTy = op.getOperand().getType();
  aiir::Type resultLLVMTy = getTypeConverter()->convertType(op.getType());
  aiir::Type elementLLVMTy =
      getTypeConverter()->convertType(operandTy.getPointee());

  aiir::LLVM::GEPArg gepIndices[2] = {{0}, {1}};
  aiir::LLVM::GEPNoWrapFlags inboundsNuw =
      aiir::LLVM::GEPNoWrapFlags::inbounds | aiir::LLVM::GEPNoWrapFlags::nuw;
  rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
      op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
      inboundsNuw);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMComplexRealPtrOpLowering::matchAndRewrite(
    cir::ComplexRealPtrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  cir::PointerType operandTy = op.getOperand().getType();
  aiir::Type resultLLVMTy = getTypeConverter()->convertType(op.getType());
  aiir::Type elementLLVMTy =
      getTypeConverter()->convertType(operandTy.getPointee());

  aiir::LLVM::GEPArg gepIndices[2] = {0, 0};
  aiir::LLVM::GEPNoWrapFlags inboundsNuw =
      aiir::LLVM::GEPNoWrapFlags::inbounds | aiir::LLVM::GEPNoWrapFlags::nuw;
  rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
      op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
      inboundsNuw);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMGetBitfieldOpLowering::matchAndRewrite(
    cir::GetBitfieldOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {

  aiir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  cir::BitfieldInfoAttr info = op.getBitfieldInfo();
  uint64_t size = info.getSize();
  uint64_t offset = info.getOffset();
  aiir::Type storageType = info.getStorageType();
  aiir::AIIRContext *context = storageType.getContext();
  unsigned storageSize = 0;

  aiir::IntegerType intType =
      computeBitfieldIntType(storageType, context, storageSize);

  aiir::Value val = aiir::LLVM::LoadOp::create(
      rewriter, op.getLoc(), intType, adaptor.getAddr(), op.getAlignment(),
      op.getIsVolatile());
  val = aiir::LLVM::BitcastOp::create(rewriter, op.getLoc(), intType, val);

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

  aiir::Type resTy = getTypeConverter()->convertType(op.getType());
  aiir::Value newOp = createIntCast(
      rewriter, val, aiir::cast<aiir::IntegerType>(resTy), info.getIsSigned());
  rewriter.replaceOp(op, newOp);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMIsConstantOpLowering::matchAndRewrite(
    cir::IsConstantOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<aiir::LLVM::IsConstantOp>(op, adaptor.getVal());
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMInlineAsmOpLowering::matchAndRewrite(
    cir::InlineAsmOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type llResTy;
  if (op.getNumResults())
    llResTy = getTypeConverter()->convertType(op.getType(0));

  cir::AsmFlavor dialect = op.getAsmFlavor();
  aiir::LLVM::AsmDialect llDialect = dialect == cir::AsmFlavor::x86_att
                                         ? aiir::LLVM::AsmDialect::AD_ATT
                                         : aiir::LLVM::AsmDialect::AD_Intel;

  SmallVector<aiir::Attribute> opAttrs;
  StringRef llvmAttrName = aiir::LLVM::InlineAsmOp::getElementTypeAttrName();

  // this is for the lowering to LLVM from LLVM dialect. Otherwise, if we
  // don't have the result (i.e. void type as a result of operation), the
  // element type attribute will be attached to the whole instruction, but not
  // to the operand
  if (!op.getNumResults())
    opAttrs.push_back(aiir::Attribute());

  SmallVector<aiir::Value> llvmOperands;
  SmallVector<aiir::Value> cirOperands;
  for (auto const &[llvmOp, cirOp] :
       zip(adaptor.getAsmOperands(), op.getAsmOperands())) {
    append_range(llvmOperands, llvmOp);
    append_range(cirOperands, cirOp);
  }

  // so far we infer the llvm dialect element type attr from
  // CIR operand type.
  for (auto const &[cirOpAttr, cirOp] :
       zip(op.getOperandAttrs(), cirOperands)) {
    if (!cirOpAttr) {
      opAttrs.push_back(aiir::Attribute());
      continue;
    }

    llvm::SmallVector<aiir::NamedAttribute, 1> attrs;
    cir::PointerType typ = aiir::cast<cir::PointerType>(cirOp.getType());
    aiir::TypeAttr typAttr = aiir::TypeAttr::get(convertTypeForMemory(
        *getTypeConverter(), dataLayout, typ.getPointee()));

    attrs.push_back(rewriter.getNamedAttr(llvmAttrName, typAttr));
    aiir::DictionaryAttr newDict = rewriter.getDictionaryAttr(attrs);
    opAttrs.push_back(newDict);
  }

  rewriter.replaceOpWithNewOp<aiir::LLVM::InlineAsmOp>(
      op, llResTy, llvmOperands, op.getAsmStringAttr(), op.getConstraintsAttr(),
      op.getSideEffectsAttr(),
      /*is_align_stack*/ aiir::UnitAttr(),
      /*tail_call_kind*/
      aiir::LLVM::TailCallKindAttr::get(
          getContext(), aiir::LLVM::tailcallkind::TailCallKind::None),
      aiir::LLVM::AsmDialectAttr::get(getContext(), llDialect),
      rewriter.getArrayAttr(opAttrs));

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVAStartOpLowering::matchAndRewrite(
    cir::VAStartOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = aiir::LLVM::LLVMPointerType::get(getContext());
  auto vaList = aiir::LLVM::BitcastOp::create(rewriter, op.getLoc(), opaquePtr,
                                              adaptor.getArgList());
  rewriter.replaceOpWithNewOp<aiir::LLVM::VaStartOp>(op, vaList);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVAEndOpLowering::matchAndRewrite(
    cir::VAEndOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = aiir::LLVM::LLVMPointerType::get(getContext());
  auto vaList = aiir::LLVM::BitcastOp::create(rewriter, op.getLoc(), opaquePtr,
                                              adaptor.getArgList());
  rewriter.replaceOpWithNewOp<aiir::LLVM::VaEndOp>(op, vaList);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVACopyOpLowering::matchAndRewrite(
    cir::VACopyOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = aiir::LLVM::LLVMPointerType::get(getContext());
  auto dstList = aiir::LLVM::BitcastOp::create(rewriter, op.getLoc(), opaquePtr,
                                               adaptor.getDstList());
  auto srcList = aiir::LLVM::BitcastOp::create(rewriter, op.getLoc(), opaquePtr,
                                               adaptor.getSrcList());
  rewriter.replaceOpWithNewOp<aiir::LLVM::VaCopyOp>(op, dstList, srcList);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMVAArgOpLowering::matchAndRewrite(
    cir::VAArgOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  assert(!cir::MissingFeatures::vaArgABILowering());
  auto opaquePtr = aiir::LLVM::LLVMPointerType::get(getContext());
  auto vaList = aiir::LLVM::BitcastOp::create(rewriter, op.getLoc(), opaquePtr,
                                              adaptor.getArgList());

  aiir::Type llvmType =
      getTypeConverter()->convertType(op->getResultTypes().front());
  if (!llvmType)
    return aiir::failure();

  rewriter.replaceOpWithNewOp<aiir::LLVM::VaArgOp>(op, llvmType, vaList);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMLabelOpLowering::matchAndRewrite(
    cir::LabelOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::AIIRContext *ctx = rewriter.getContext();
  aiir::Block *block = op->getBlock();
  // A BlockTagOp cannot reside in the entry block. The address of the entry
  // block cannot be taken
  if (block->isEntryBlock()) {
    aiir::Block *newBlock =
        rewriter.splitBlock(op->getBlock(), aiir::Block::iterator(op));
    rewriter.setInsertionPointToEnd(block);
    aiir::LLVM::BrOp::create(rewriter, op.getLoc(), newBlock);
  }
  auto tagAttr =
      aiir::LLVM::BlockTagAttr::get(ctx, blockInfoAddr.getTagIndex());
  rewriter.setInsertionPoint(op);

  auto blockTagOp =
      aiir::LLVM::BlockTagOp::create(rewriter, op->getLoc(), tagAttr);
  aiir::LLVM::LLVMFuncOp func = op->getParentOfType<aiir::LLVM::LLVMFuncOp>();
  auto blockInfoAttr =
      cir::BlockAddrInfoAttr::get(ctx, func.getSymName(), op.getLabel());
  blockInfoAddr.mapBlockTag(blockInfoAttr, blockTagOp);
  rewriter.eraseOp(op);

  return aiir::success();
}

aiir::LogicalResult CIRToLLVMBlockAddressOpLowering::matchAndRewrite(
    cir::BlockAddressOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::AIIRContext *ctx = rewriter.getContext();

  aiir::LLVM::BlockTagOp matchLabel =
      blockInfoAddr.lookupBlockTag(op.getBlockAddrInfoAttr());
  aiir::LLVM::BlockTagAttr tagAttr;
  if (!matchLabel)
    // If the BlockTagOp has not been emitted yet, use  a placeholder.
    // This will later be replaced with the correct tag index during
    // `resolveBlockAddressOp`.
    tagAttr = {};
  else
    tagAttr = matchLabel.getTag();

  auto blkAddr = aiir::LLVM::BlockAddressAttr::get(
      rewriter.getContext(), op.getBlockAddrInfoAttr().getFunc(), tagAttr);
  rewriter.setInsertionPoint(op);
  auto newOp = aiir::LLVM::BlockAddressOp::create(
      rewriter, op.getLoc(), aiir::LLVM::LLVMPointerType::get(ctx), blkAddr);
  if (!matchLabel)
    blockInfoAddr.addUnresolvedBlockAddress(newOp, op.getBlockAddrInfoAttr());
  rewriter.replaceOp(op, newOp);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMIndirectBrOpLowering::matchAndRewrite(
    cir::IndirectBrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {

  llvm::SmallVector<aiir::Block *, 8> successors;
  llvm::SmallVector<aiir::ValueRange, 8> succOperands;
  bool poison = op.getPoison();
  for (aiir::Block *succ : op->getSuccessors())
    successors.push_back(succ);

  for (aiir::ValueRange operand : op.getSuccOperands()) {
    succOperands.push_back(operand);
  }

  auto llvmPtrType = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  aiir::Value targetAddr;
  if (!poison) {
    targetAddr = aiir::LLVM::BitcastOp::create(rewriter, op.getLoc(),
                                               llvmPtrType, adaptor.getAddr());
  } else {
    targetAddr =
        aiir::LLVM::PoisonOp::create(rewriter, op->getLoc(), llvmPtrType);
    // Remove the block argument to avoid generating an empty PHI during
    // lowering.
    op->getBlock()->eraseArgument(0);
  }

  auto newOp = aiir::LLVM::IndirectBrOp::create(
      rewriter, op.getLoc(), targetAddr, succOperands, successors);
  rewriter.replaceOp(op, newOp);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMAwaitOpLowering::matchAndRewrite(
    cir::AwaitOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  return aiir::failure();
}

aiir::LogicalResult CIRToLLVMCpuIdOpLowering::matchAndRewrite(
    cir::CpuIdOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type i32Ty = rewriter.getI32Type();
  aiir::Type i64Ty = rewriter.getI64Type();
  aiir::Type i32PtrTy = aiir::LLVM::LLVMPointerType::get(i32Ty.getContext(), 0);

  aiir::Type cpuidRetTy = aiir::LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {i32Ty, i32Ty, i32Ty, i32Ty});

  aiir::Value functionId = adaptor.getFunctionId();
  aiir::Value subFunctionId = adaptor.getSubFunctionId();

  StringRef asmString, constraints;
  aiir::ModuleOp moduleOp = op->getParentOfType<aiir::ModuleOp>();
  llvm::Triple triple(
      aiir::cast<aiir::StringAttr>(
          moduleOp->getAttr(cir::CIRDialect::getTripleAttrName()))
          .getValue());
  if (triple.getArch() == llvm::Triple::x86) {
    asmString = "cpuid";
    constraints = "={ax},={bx},={cx},={dx},{ax},{cx}";
  } else {
    // x86-64 uses %rbx as the base register, so preserve it.
    asmString = "xchgq %rbx, ${1:q}\n"
                "cpuid\n"
                "xchgq %rbx, ${1:q}";
    constraints = "={ax},=r,={cx},={dx},0,2";
  }

  aiir::Value inlineAsm =
      aiir::LLVM::InlineAsmOp::create(
          rewriter, op.getLoc(), cpuidRetTy, {functionId, subFunctionId},
          rewriter.getStringAttr(asmString),
          rewriter.getStringAttr(constraints),
          /*has_side_effects=*/aiir::UnitAttr{},
          /*is_align_stack=*/aiir::UnitAttr{},
          /*tail_call_kind=*/aiir::LLVM::TailCallKindAttr{},
          /*asm_dialect=*/aiir::LLVM::AsmDialectAttr{},
          /*operand_attrs=*/aiir::ArrayAttr{})
          .getResult(0);

  aiir::Value basePtr = adaptor.getCpuInfo();

  aiir::DataLayout layout(op->getParentOfType<aiir::ModuleOp>());
  unsigned alignment = layout.getTypeABIAlignment(i32Ty);
  for (unsigned i = 0; i < 4; i++) {
    aiir::Value extracted =
        aiir::LLVM::ExtractValueOp::create(rewriter, op.getLoc(), inlineAsm, i)
            .getResult();
    aiir::Value index = aiir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(i));
    llvm::SmallVector<aiir::Value, 1> gepIndices = {index};
    aiir::Value storePtr = aiir::LLVM::GEPOp::create(
                               rewriter, op.getLoc(), i32PtrTy, i32Ty, basePtr,
                               gepIndices, aiir::LLVM::GEPNoWrapFlags::none)
                               .getResult();
    aiir::LLVM::StoreOp::create(rewriter, op.getLoc(), extracted, storePtr,
                                alignment);
  }

  rewriter.eraseOp(op);
  return aiir::success();
}

aiir::LogicalResult CIRToLLVMMemChrOpLowering::matchAndRewrite(
    cir::MemChrOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  aiir::Type srcTy = getTypeConverter()->convertType(op.getSrc().getType());
  aiir::Type patternTy =
      getTypeConverter()->convertType(op.getPattern().getType());
  aiir::Type lenTy = getTypeConverter()->convertType(op.getLen().getType());
  auto fnTy =
      aiir::LLVM::LLVMFunctionType::get(llvmPtrTy, {srcTy, patternTy, lenTy},
                                        /*isVarArg=*/false);
  llvm::StringRef fnName = "memchr";
  createLLVMFuncOpIfNotExist(rewriter, op, fnName, fnTy);
  rewriter.replaceOpWithNewOp<aiir::LLVM::CallOp>(
      op, aiir::TypeRange{llvmPtrTy}, fnName,
      aiir::ValueRange{adaptor.getSrc(), adaptor.getPattern(),
                       adaptor.getLen()});
  return aiir::success();
}

std::unique_ptr<aiir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

void populateCIRToLLVMPasses(aiir::OpPassManager &pm) {
  aiir::populateCIRPreLoweringPasses(pm);
  pm.addPass(createConvertCIRToLLVMPass());
}

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(aiir::ModuleOp aiirModule, LLVMContext &llvmCtx,
                             StringRef aiirSaveTempsOutFile) {
  llvm::TimeTraceScope scope("lower from CIR to LLVM directly");

  aiir::AIIRContext *aiirCtx = aiirModule.getContext();

  aiir::PassManager pm(aiirCtx);
  populateCIRToLLVMPasses(pm);

  (void)aiir::applyPassManagerCLOptions(pm);

  if (aiir::failed(pm.run(aiirModule))) {
    // FIXME: Handle any errors where they occurs and return a nullptr here.
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");
  }

  if (!aiirSaveTempsOutFile.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream out(aiirSaveTempsOutFile, ec);
    if (!ec)
      aiirModule->print(out);
  }

  aiir::registerBuiltinDialectTranslation(*aiirCtx);
  aiir::registerLLVMDialectTranslation(*aiirCtx);
  aiir::registerOpenMPDialectTranslation(*aiirCtx);
  aiir::registerCIRDialectTranslation(*aiirCtx);

  llvm::TimeTraceScope translateScope("translateModuleToLLVMIR");

  StringRef moduleName = aiirModule.getName().value_or("CIRToLLVMModule");
  std::unique_ptr<llvm::Module> llvmModule =
      aiir::translateModuleToLLVMIR(aiirModule, llvmCtx, moduleName);

  if (!llvmModule) {
    // FIXME: Handle any errors where they occurs and return a nullptr here.
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");
  }

  return llvmModule;
}
} // namespace direct
} // namespace cir
