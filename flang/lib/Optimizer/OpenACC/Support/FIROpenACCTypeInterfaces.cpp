//===-- FIROpenACCTypeInterfaces.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of external dialect interfaces for FIR.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Support/FIROpenACCTypeInterfaces.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "flang/Optimizer/Support/Utils.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> useAccReductionCombine(
    "openacc-use-reduction-combine",
    llvm::cl::desc("Whether to generate acc.reduction_combine. Does not "
                   "control reduction for MIN/MAX and logical reductions."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> useAccReductionCombineAll(
    "openacc-use-reduction-combine-all",
    llvm::cl::desc("Whether to generate acc.reduction_combine for all types "
                   "and operators"),
    llvm::cl::init(false));

namespace fir::acc {

template <typename Ty>
aiir::TypedValue<aiir::acc::PointerLikeType>
OpenACCMappableModel<Ty>::getVarPtr(aiir::Type type, aiir::Value var) const {
  if (auto ptr =
          aiir::dyn_cast<aiir::TypedValue<aiir::acc::PointerLikeType>>(var))
    return ptr;

  if (auto load = aiir::dyn_cast_if_present<fir::LoadOp>(var.getDefiningOp())) {
    // All FIR reference types implement the PointerLikeType interface.
    return aiir::cast<aiir::TypedValue<aiir::acc::PointerLikeType>>(
        load.getMemref());
  }

  return {};
}

template aiir::TypedValue<aiir::acc::PointerLikeType>
OpenACCMappableModel<fir::BaseBoxType>::getVarPtr(aiir::Type type,
                                                  aiir::Value var) const;

template aiir::TypedValue<aiir::acc::PointerLikeType>
OpenACCMappableModel<fir::ReferenceType>::getVarPtr(aiir::Type type,
                                                    aiir::Value var) const;

template aiir::TypedValue<aiir::acc::PointerLikeType>
OpenACCMappableModel<fir::HeapType>::getVarPtr(aiir::Type type,
                                               aiir::Value var) const;

template aiir::TypedValue<aiir::acc::PointerLikeType>
OpenACCMappableModel<fir::PointerType>::getVarPtr(aiir::Type type,
                                                  aiir::Value var) const;

template <typename Ty>
std::optional<llvm::TypeSize> OpenACCMappableModel<Ty>::getSizeInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const {
  // TODO: Bounds operation affect the size - add support to take them
  // into account.
  if (!accBounds.empty())
    return {};

  // Class-type is either a polymorphic or unlimited polymorphic. In the latter
  // case, the size is not computable. But in the former it should be - however,
  // fir::getTypeSizeAndAlignment does not support polymorphic types.
  if (aiir::isa<fir::ClassType>(type)) {
    return {};
  }

  // When requesting the size of a box entity or a reference, the intent
  // is to get the size of the data that it is referring to.
  aiir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  assert(eleTy && "expect to be able to unwrap the element type");

  // If the type enclosed is a mappable type, then have it provide the size.
  if (auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(eleTy))
    return mappableTy.getSizeInBytes(var, accBounds, dataLayout);

  // Dynamic extents or unknown ranks generally do not have compile-time
  // computable dimensions.
  auto seqType = aiir::dyn_cast<fir::SequenceType>(eleTy);
  if (seqType && (seqType.hasDynamicExtents() || seqType.hasUnknownShape()))
    return {};

  // Attempt to find an operation that a lookup for KindMapping can be done
  // from.
  aiir::Operation *kindMapSrcOp = var.getDefiningOp();
  if (!kindMapSrcOp) {
    kindMapSrcOp = var.getParentRegion()->getParentOp();
    if (!kindMapSrcOp)
      return {};
  }
  auto kindMap = fir::getKindMapping(kindMapSrcOp);

  auto sizeAndAlignment =
      fir::getTypeSizeAndAlignment(var.getLoc(), eleTy, dataLayout, kindMap);
  if (!sizeAndAlignment.has_value())
    return {};

  return {llvm::TypeSize::getFixed(sizeAndAlignment->first)};
}

template std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::BaseBoxType>::getSizeInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::ReferenceType>::getSizeInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::HeapType>::getSizeInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::PointerType>::getSizeInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template <typename Ty>
std::optional<int64_t> OpenACCMappableModel<Ty>::getOffsetInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const {
  // TODO: Bounds operation affect the offset - add support to take them
  // into account.
  if (!accBounds.empty())
    return {};

  // Class-type does not behave like a normal box because it does not hold an
  // element type. Thus special handle it here.
  if (aiir::isa<fir::ClassType>(type)) {
    // The pointer to the class-type is always at the start address.
    return {0};
  }

  aiir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  assert(eleTy && "expect to be able to unwrap the element type");

  // If the type enclosed is a mappable type, then have it provide the offset.
  if (auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(eleTy))
    return mappableTy.getOffsetInBytes(var, accBounds, dataLayout);

  // Dynamic extents (aka descriptor-based arrays) - may have a offset.
  // For example, a negative stride may mean a negative offset to compute the
  // start of array.
  auto seqType = aiir::dyn_cast<fir::SequenceType>(eleTy);
  if (seqType && (seqType.hasDynamicExtents() || seqType.hasUnknownShape()))
    return {};

  // If the size is computable and since there are no bounds or dynamic extents,
  // then the offset relative to pointer must be zero.
  if (getSizeInBytes(type, var, accBounds, dataLayout).has_value()) {
    return {0};
  }

  // The offset is not evident because it is relative to the pointer being held.
  // And we don't have any further details about this type.
  return {};
}

template std::optional<int64_t>
OpenACCMappableModel<fir::BaseBoxType>::getOffsetInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template std::optional<int64_t>
OpenACCMappableModel<fir::ReferenceType>::getOffsetInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template std::optional<int64_t>
OpenACCMappableModel<fir::HeapType>::getOffsetInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template std::optional<int64_t>
OpenACCMappableModel<fir::PointerType>::getOffsetInBytes(
    aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
    const aiir::DataLayout &dataLayout) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::hasUnknownDimensions(aiir::Type type) const {
  assert(fir::isa_ref_type(type) && "expected FIR reference type");
  return fir::hasDynamicSize(fir::unwrapRefType(type));
}

template bool OpenACCMappableModel<fir::ReferenceType>::hasUnknownDimensions(
    aiir::Type type) const;

template bool OpenACCMappableModel<fir::HeapType>::hasUnknownDimensions(
    aiir::Type type) const;

template bool OpenACCMappableModel<fir::PointerType>::hasUnknownDimensions(
    aiir::Type type) const;

template <>
bool OpenACCMappableModel<fir::BaseBoxType>::hasUnknownDimensions(
    aiir::Type type) const {
  // Descriptor-based entities have dimensions encoded.
  return false;
}

static llvm::SmallVector<aiir::Value>
generateSeqTyAccBounds(fir::SequenceType seqType, aiir::Value var,
                       aiir::OpBuilder &builder) {
  assert((aiir::isa<aiir::acc::PointerLikeType>(var.getType()) ||
          aiir::isa<aiir::acc::MappableType>(var.getType())) &&
         "must be pointer-like or mappable");
  fir::FirOpBuilder firBuilder(builder, var.getDefiningOp());
  aiir::Location loc = var.getLoc();

  // If [hl]fir.declare is visible, extract the bounds from the declaration's
  // shape (if it is provided).
  if (aiir::isa<hlfir::DeclareOp, fir::DeclareOp>(var.getDefiningOp())) {
    aiir::Value zero =
        firBuilder.createIntegerConstant(loc, builder.getIndexType(), 0);
    aiir::Value one =
        firBuilder.createIntegerConstant(loc, builder.getIndexType(), 1);

    aiir::Value shape;
    if (auto declareOp =
            aiir::dyn_cast_if_present<fir::DeclareOp>(var.getDefiningOp()))
      shape = declareOp.getShape();
    else if (auto declareOp = aiir::dyn_cast_if_present<hlfir::DeclareOp>(
                 var.getDefiningOp()))
      shape = declareOp.getShape();

    const bool strideIncludeLowerExtent = true;

    llvm::SmallVector<aiir::Value> accBounds;
    aiir::Operation *anyShapeOp = shape ? shape.getDefiningOp() : nullptr;
    if (auto shapeOp = aiir::dyn_cast_if_present<fir::ShapeOp>(anyShapeOp)) {
      aiir::Value cummulativeExtent = one;
      for (auto extent : shapeOp.getExtents()) {
        aiir::Value upperbound =
            aiir::arith::SubIOp::create(builder, loc, extent, one);
        aiir::Value stride = one;
        if (strideIncludeLowerExtent) {
          stride = cummulativeExtent;
          cummulativeExtent = aiir::arith::MulIOp::create(
              builder, loc, cummulativeExtent, extent);
        }
        auto accBound = aiir::acc::DataBoundsOp::create(
            builder, loc, aiir::acc::DataBoundsType::get(builder.getContext()),
            /*lowerbound=*/zero, /*upperbound=*/upperbound,
            /*extent=*/extent, /*stride=*/stride, /*strideInBytes=*/false,
            /*startIdx=*/one);
        accBounds.push_back(accBound);
      }
    } else if (auto shapeShiftOp =
                   aiir::dyn_cast_if_present<fir::ShapeShiftOp>(anyShapeOp)) {
      aiir::Value lowerbound;
      aiir::Value cummulativeExtent = one;
      for (auto [idx, val] : llvm::enumerate(shapeShiftOp.getPairs())) {
        if (idx % 2 == 0) {
          lowerbound = val;
        } else {
          aiir::Value extent = val;
          aiir::Value upperbound =
              aiir::arith::SubIOp::create(builder, loc, extent, one);
          aiir::Value stride = one;
          if (strideIncludeLowerExtent) {
            stride = cummulativeExtent;
            cummulativeExtent = aiir::arith::MulIOp::create(
                builder, loc, cummulativeExtent, extent);
          }
          auto accBound = aiir::acc::DataBoundsOp::create(
              builder, loc,
              aiir::acc::DataBoundsType::get(builder.getContext()),
              /*lowerbound=*/zero, /*upperbound=*/upperbound,
              /*extent=*/extent, /*stride=*/stride, /*strideInBytes=*/false,
              /*startIdx=*/lowerbound);
          accBounds.push_back(accBound);
        }
      }
    }

    if (!accBounds.empty())
      return accBounds;
  }

  if (seqType.hasDynamicExtents() || seqType.hasUnknownShape()) {
    aiir::Value box;
    bool mayBeOptional = false;
    if (auto boxAddr =
            aiir::dyn_cast_if_present<fir::BoxAddrOp>(var.getDefiningOp())) {
      box = boxAddr.getVal();
      // Since fir.box_addr already accesses the box, we do not care
      // checking if it is optional.
    } else if (aiir::isa<fir::BaseBoxType>(var.getType())) {
      box = var;
      mayBeOptional = fir::mayBeAbsentBox(box);
    }

    if (box) {
      auto res =
          hlfir::translateToExtendedValue(loc, firBuilder, hlfir::Entity(box));
      fir::ExtendedValue exv = res.first;
      aiir::Value boxRef = box;
      if (auto boxPtr =
              aiir::cast<aiir::acc::MappableType>(box.getType()).getVarPtr(box))
        boxRef = boxPtr;

      aiir::Value isPresent =
          !mayBeOptional ? aiir::Value{}
                         : fir::IsPresentOp::create(builder, loc,
                                                    builder.getI1Type(), box);

      fir::factory::AddrAndBoundsInfo info(box, boxRef, isPresent,
                                           box.getType());
      return fir::factory::genBoundsOpsFromBox<aiir::acc::DataBoundsOp,
                                               aiir::acc::DataBoundsType>(
          firBuilder, loc, exv, info);
    }

    assert(false && "array with unknown dimension expected to have descriptor");
    return {};
  }

  // TODO: Detect assumed-size case.
  const bool isAssumedSize = false;
  auto valToCheck = var;
  if (auto boxAddr =
          aiir::dyn_cast_if_present<fir::BoxAddrOp>(var.getDefiningOp())) {
    valToCheck = boxAddr.getVal();
  }
  auto res = hlfir::translateToExtendedValue(loc, firBuilder,
                                             hlfir::Entity(valToCheck));
  fir::ExtendedValue exv = res.first;
  return fir::factory::genBaseBoundsOps<aiir::acc::DataBoundsOp,
                                        aiir::acc::DataBoundsType>(
      firBuilder, loc, exv,
      /*isAssumedSize=*/isAssumedSize);
}

template <typename Ty>
llvm::SmallVector<aiir::Value>
OpenACCMappableModel<Ty>::generateAccBounds(aiir::Type type, aiir::Value var,
                                            aiir::OpBuilder &builder) const {
  // acc bounds only make sense for arrays - thus look for sequence type.
  aiir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  if (auto seqTy = aiir::dyn_cast_if_present<fir::SequenceType>(eleTy)) {
    return generateSeqTyAccBounds(seqTy, var, builder);
  }

  return {};
}

template llvm::SmallVector<aiir::Value>
OpenACCMappableModel<fir::BaseBoxType>::generateAccBounds(
    aiir::Type type, aiir::Value var, aiir::OpBuilder &builder) const;

template llvm::SmallVector<aiir::Value>
OpenACCMappableModel<fir::ReferenceType>::generateAccBounds(
    aiir::Type type, aiir::Value var, aiir::OpBuilder &builder) const;

template llvm::SmallVector<aiir::Value>
OpenACCMappableModel<fir::HeapType>::generateAccBounds(
    aiir::Type type, aiir::Value var, aiir::OpBuilder &builder) const;

template llvm::SmallVector<aiir::Value>
OpenACCMappableModel<fir::PointerType>::generateAccBounds(
    aiir::Type type, aiir::Value var, aiir::OpBuilder &builder) const;

static aiir::Value
getBaseRef(aiir::TypedValue<aiir::acc::PointerLikeType> varPtr) {
  // If there is no defining op - the unwrapped reference is the base one.
  aiir::Operation *op = varPtr.getDefiningOp();
  if (!op)
    return varPtr;

  // Look to find if this value originates from an interior pointer
  // calculation op.
  aiir::Value baseRef =
      llvm::TypeSwitch<aiir::Operation *, aiir::Value>(op)
          .Case([&](fir::DeclareOp op) {
            // If this declare binds a view with an underlying storage operand,
            // treat that storage as the base reference. Otherwise, fall back
            // to the declared memref.
            if (auto storage = op.getStorage())
              return storage;
            return aiir::Value(varPtr);
          })
          .Case([&](hlfir::DesignateOp op) {
            // Get the base object.
            return op.getMemref();
          })
          .Case<fir::ArrayCoorOp, fir::cg::XArrayCoorOp>([&](auto op) {
            // Get the base array on which the coordinate is being applied.
            return op.getMemref();
          })
          .Case([&](fir::CoordinateOp op) {
            // For coordinate operation which is applied on derived type
            // object, get the base object.
            return op.getRef();
          })
          .Case([&](fir::ConvertOp op) -> aiir::Value {
            // Strip the conversion and recursively check the operand
            if (auto ptrLikeOperand = aiir::dyn_cast_if_present<
                    aiir::TypedValue<aiir::acc::PointerLikeType>>(
                    op.getValue()))
              return getBaseRef(ptrLikeOperand);
            return varPtr;
          })
          .Default([&](aiir::Operation *) { return varPtr; });

  return baseRef;
}

static bool isScalarLike(aiir::Type type) {
  return fir::isa_trivial(type) || fir::isa_ref_type(type);
}

static bool isArrayLike(aiir::Type type) {
  return aiir::isa<fir::SequenceType>(type);
}

static bool isCompositeLike(aiir::Type type) {
  // class(*) is not a composite type since it does not have a determined type.
  if (fir::isUnlimitedPolymorphicType(type))
    return false;

  return aiir::isa<fir::RecordType, fir::ClassType, aiir::TupleType>(type);
}

static aiir::acc::VariableTypeCategory
categorizeElemType(aiir::Type enclosingTy, aiir::Type eleTy, aiir::Value var) {
  // If the type enclosed is a mappable type, then have it provide the type
  // category.
  if (auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(eleTy))
    return mappableTy.getTypeCategory(var);

  // For all arrays, despite whether they are allocatable, pointer, assumed,
  // etc, we'd like to categorize them as "array".
  if (isArrayLike(eleTy))
    return aiir::acc::VariableTypeCategory::array;

  if (isCompositeLike(eleTy))
    return aiir::acc::VariableTypeCategory::composite;
  if (aiir::isa<fir::BoxType>(enclosingTy)) {
    // Even if we have a scalar type - simply because it is wrapped in a box
    // we want to categorize it as "nonscalar". Anything else would've been
    // non-scalar anyway.
    return aiir::acc::VariableTypeCategory::nonscalar;
  }
  if (isScalarLike(eleTy))
    return aiir::acc::VariableTypeCategory::scalar;
  if (aiir::isa<fir::CharacterType, aiir::FunctionType>(eleTy))
    return aiir::acc::VariableTypeCategory::nonscalar;
  // Assumed-type (type(*))does not have a determined type that can be
  // categorized.
  if (aiir::isa<aiir::NoneType>(eleTy))
    return aiir::acc::VariableTypeCategory::uncategorized;
  // "pointers" - in the sense of raw address point-of-view, are considered
  // scalars.
  if (aiir::isa<fir::LLVMPointerType>(eleTy))
    return aiir::acc::VariableTypeCategory::scalar;

  // Without further checking, this type cannot be categorized.
  return aiir::acc::VariableTypeCategory::uncategorized;
}

template <typename Ty>
aiir::acc::VariableTypeCategory
OpenACCMappableModel<Ty>::getTypeCategory(aiir::Type type,
                                          aiir::Value var) const {
  // FIR uses operations to compute interior pointers.
  // So for example, an array element or composite field access to a float
  // value would both be represented as !fir.ref<f32>. We do not want to treat
  // such a reference as a scalar. Thus unwrap interior pointer calculations.
  aiir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  if (eleTy && isScalarLike(eleTy)) {
    if (auto ptrLikeVar = aiir::dyn_cast_if_present<
            aiir::TypedValue<aiir::acc::PointerLikeType>>(var)) {
      auto baseRef = getBaseRef(ptrLikeVar);
      if (baseRef != var) {
        type = baseRef.getType();
        if (auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(type))
          return mappableTy.getTypeCategory(baseRef);
      }
    }
  }

  // Class-type does not behave like a normal box because it does not hold an
  // element type. Thus special handle it here.
  if (aiir::isa<fir::ClassType>(type)) {
    // class(*) is not a composite type since it does not have a determined
    // type.
    if (fir::isUnlimitedPolymorphicType(type))
      return aiir::acc::VariableTypeCategory::uncategorized;
    return aiir::acc::VariableTypeCategory::composite;
  }

  assert(eleTy && "expect to be able to unwrap the element type");
  return categorizeElemType(type, eleTy, var);
}

template aiir::acc::VariableTypeCategory
OpenACCMappableModel<fir::BaseBoxType>::getTypeCategory(aiir::Type type,
                                                        aiir::Value var) const;

template aiir::acc::VariableTypeCategory
OpenACCMappableModel<fir::ReferenceType>::getTypeCategory(
    aiir::Type type, aiir::Value var) const;

template aiir::acc::VariableTypeCategory
OpenACCMappableModel<fir::HeapType>::getTypeCategory(aiir::Type type,
                                                     aiir::Value var) const;

template aiir::acc::VariableTypeCategory
OpenACCMappableModel<fir::PointerType>::getTypeCategory(aiir::Type type,
                                                        aiir::Value var) const;

template <typename Ty>
aiir::acc::VariableInfoAttr OpenACCMappableModel<Ty>::genPrivateVariableInfo(
    aiir::Type type, aiir::TypedValue<aiir::acc::MappableType> var) const {
  hlfir::Entity entity{var};
  return fir::OpenACCFortranVariableInfoAttr::get(var.getContext(),
                                                  entity.mayBeOptional());
}

template aiir::acc::VariableInfoAttr
OpenACCMappableModel<fir::BaseBoxType>::genPrivateVariableInfo(
    aiir::Type type, aiir::TypedValue<aiir::acc::MappableType> var) const;

template aiir::acc::VariableInfoAttr
OpenACCMappableModel<fir::ReferenceType>::genPrivateVariableInfo(
    aiir::Type type, aiir::TypedValue<aiir::acc::MappableType> var) const;

template aiir::acc::VariableInfoAttr
OpenACCMappableModel<fir::HeapType>::genPrivateVariableInfo(
    aiir::Type type, aiir::TypedValue<aiir::acc::MappableType> var) const;

template aiir::acc::VariableInfoAttr
OpenACCMappableModel<fir::PointerType>::genPrivateVariableInfo(
    aiir::Type type, aiir::TypedValue<aiir::acc::MappableType> var) const;

static aiir::acc::VariableTypeCategory
categorizePointee(aiir::Type pointer,
                  aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
                  aiir::Type varType) {
  // FIR uses operations to compute interior pointers.
  // So for example, an array element or composite field access to a float
  // value would both be represented as !fir.ref<f32>. We do not want to treat
  // such a reference as a scalar. Thus unwrap interior pointer calculations.
  auto baseRef = getBaseRef(varPtr);

  if (auto mappableTy =
          aiir::dyn_cast<aiir::acc::MappableType>(baseRef.getType()))
    return mappableTy.getTypeCategory(baseRef);

  // It must be a pointer-like type since it is not a MappableType.
  auto ptrLikeTy = aiir::cast<aiir::acc::PointerLikeType>(baseRef.getType());
  aiir::Type eleTy = ptrLikeTy.getElementType();
  return categorizeElemType(pointer, eleTy, varPtr);
}

template <>
aiir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::ReferenceType>::getPointeeTypeCategory(
    aiir::Type pointer, aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
    aiir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

template <>
aiir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::PointerType>::getPointeeTypeCategory(
    aiir::Type pointer, aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
    aiir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

template <>
aiir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::HeapType>::getPointeeTypeCategory(
    aiir::Type pointer, aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
    aiir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

template <>
aiir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::LLVMPointerType>::getPointeeTypeCategory(
    aiir::Type pointer, aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
    aiir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

static hlfir::Entity
genDesignateWithTriplets(fir::FirOpBuilder &builder, aiir::Location loc,
                         hlfir::Entity &entity,
                         hlfir::DesignateOp::Subscripts &triplets,
                         aiir::Value shape, aiir::ValueRange extents) {
  llvm::SmallVector<aiir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, entity, lenParams);

  // Compute result type of array section.
  fir::SequenceType::Shape resultTypeShape;
  bool shapeIsConstant = true;
  for (aiir::Value extent : extents) {
    if (std::optional<std::int64_t> cst_extent =
            fir::getIntIfConstant(extent)) {
      resultTypeShape.push_back(*cst_extent);
    } else {
      resultTypeShape.push_back(fir::SequenceType::getUnknownExtent());
      shapeIsConstant = false;
    }
  }
  assert(!resultTypeShape.empty() &&
         "expect private sections to always represented as arrays");
  aiir::Type eleTy = entity.getFortranElementType();
  auto seqTy = fir::SequenceType::get(resultTypeShape, eleTy);
  bool isVolatile = fir::isa_volatile_type(entity.getType());
  bool resultNeedsBox =
      llvm::isa<fir::BaseBoxType>(entity.getType()) || !shapeIsConstant;
  bool isPolymorphic = fir::isPolymorphicType(entity.getType());
  aiir::Type resultType;
  if (isPolymorphic) {
    resultType = fir::ClassType::get(seqTy, isVolatile);
  } else if (resultNeedsBox) {
    resultType = fir::BoxType::get(seqTy, isVolatile);
  } else {
    resultType = fir::ReferenceType::get(seqTy, isVolatile);
  }

  // Generate section with hlfir.designate.
  auto designate = hlfir::DesignateOp::create(
      builder, loc, resultType, entity, /*component=*/"",
      /*componentShape=*/aiir::Value{}, triplets,
      /*substring=*/aiir::ValueRange{}, /*complexPartAttr=*/std::nullopt, shape,
      lenParams);
  return hlfir::Entity{designate.getResult()};
}

// Designate uses triplets based on object lower bounds while acc.bounds are
// zero based. This helper shift the bounds to create the designate triplets.
static hlfir::DesignateOp::Subscripts
genTripletsFromAccBounds(fir::FirOpBuilder &builder, aiir::Location loc,
                         const llvm::SmallVector<aiir::Value> &accBounds,
                         hlfir::Entity entity) {
  assert(entity.getRank() * 3 == static_cast<int>(accBounds.size()) &&
         "must get lb,ub,step for each dimension");
  hlfir::DesignateOp::Subscripts triplets;
  for (unsigned i = 0; i < accBounds.size(); i += 3) {
    aiir::Value lb = hlfir::genLBound(loc, builder, entity, i / 3);
    lb = builder.createConvert(loc, accBounds[i].getType(), lb);
    assert(accBounds[i].getType() == accBounds[i + 1].getType() &&
           "mix of integer types in triplets");
    aiir::Value sliceLB =
        builder.createOrFold<aiir::arith::AddIOp>(loc, accBounds[i], lb);
    aiir::Value sliceUB =
        builder.createOrFold<aiir::arith::AddIOp>(loc, accBounds[i + 1], lb);
    triplets.emplace_back(
        hlfir::DesignateOp::Triplet{sliceLB, sliceUB, accBounds[i + 2]});
  }
  return triplets;
}

static std::pair<aiir::Value, llvm::SmallVector<aiir::Value>>
computeSectionShapeAndExtents(fir::FirOpBuilder &builder, aiir::Location loc,
                              aiir::ValueRange bounds) {
  llvm::SmallVector<aiir::Value> extents;
  // Compute the fir.shape of the array section and the triplets to create
  // hlfir.designate.
  aiir::Type idxTy = builder.getIndexType();
  for (unsigned i = 0; i + 2 < bounds.size(); i += 3)
    extents.push_back(builder.genExtentFromTriplet(
        loc, bounds[i], bounds[i + 1], bounds[i + 2], idxTy, /*fold=*/true));
  aiir::Value shape = fir::ShapeOp::create(builder, loc, extents);
  return {shape, extents};
}

static std::pair<hlfir::Entity, hlfir::Entity>
genArraySectionsInRecipe(fir::FirOpBuilder &builder, aiir::Location loc,
                         aiir::ValueRange bounds, hlfir::Entity lhs,
                         hlfir::Entity rhs) {
  assert(lhs.getRank() * 3 == static_cast<int>(bounds.size()) &&
         "must get lb,ub,step for each dimension");
  lhs = hlfir::derefPointersAndAllocatables(loc, builder, lhs);
  rhs = hlfir::derefPointersAndAllocatables(loc, builder, rhs);
  // Get the list of lb,ub,step values for the sections that can be used inside
  // the recipe region.
  auto [shape, extents] = computeSectionShapeAndExtents(builder, loc, bounds);
  hlfir::DesignateOp::Subscripts rhsTriplets =
      genTripletsFromAccBounds(builder, loc, bounds, rhs);
  hlfir::DesignateOp::Subscripts lhsTriplets;
  // Share the bounds when both rhs/lhs are known to be 1-based to avoid noise
  // in the IR for the most common cases.
  if (!lhs.mayHaveNonDefaultLowerBounds() &&
      !rhs.mayHaveNonDefaultLowerBounds())
    lhsTriplets = rhsTriplets;
  else
    lhsTriplets = genTripletsFromAccBounds(builder, loc, bounds, lhs);
  hlfir::Entity leftSection =
      genDesignateWithTriplets(builder, loc, lhs, lhsTriplets, shape, extents);
  hlfir::Entity rightSection =
      genDesignateWithTriplets(builder, loc, rhs, rhsTriplets, shape, extents);
  return {leftSection, rightSection};
}

static bool boundsAreAllConstants(aiir::ValueRange bounds) {
  for (aiir::Value bound : bounds)
    if (!fir::getIntIfConstant(bound).has_value())
      return false;
  return true;
}

template <typename Ty>
aiir::Value OpenACCMappableModel<Ty>::generatePrivateInit(
    aiir::Type type, aiir::OpBuilder &aiirBuilder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::MappableType> var, llvm::StringRef varName,
    aiir::ValueRange bounds, aiir::Value initVal, aiir::acc::VariableInfoAttr,
    bool &needsDestroy) const {
  aiir::ModuleOp mod = aiirBuilder.getInsertionBlock()
                           ->getParent()
                           ->getParentOfType<aiir::ModuleOp>();
  assert(mod && "failed to retrieve ModuleOp");
  fir::FirOpBuilder builder(aiirBuilder, mod);

  hlfir::Entity inputVar = hlfir::Entity{var};
  if (inputVar.isPolymorphic())
    TODO(loc, "OpenACC: polymorphic variable privatization");
  if (auto recType =
          llvm::dyn_cast<fir::RecordType>(inputVar.getFortranElementType())) {
    // Need to make deep copies of allocatable components.
    if (fir::isRecordWithAllocatableMember(recType))
      TODO(loc,
           "OpenACC: privatizing derived type with allocatable components");
    // Need to decide if user assignment/final routine should be called.
    if (fir::isRecordWithFinalRoutine(recType, mod).value_or(false))
      TODO(loc, "OpenACC: privatizing derived type with user assignment or "
                "final routine ");
    // Pointer components needs to be initialized to NULL() for private-like
    // recipes.
    if (fir::isRecordWithDescriptorMember(recType))
      TODO(loc, "OpenACC: privatizing derived type with pointer components");
  }
  bool isPointerOrAllocatable = inputVar.isMutableBox();
  hlfir::Entity dereferencedVar =
      hlfir::derefPointersAndAllocatables(loc, builder, inputVar);

  // Step 1: Gather the address, shape, extents, and lengths parameters of the
  // entity being privatized. Designate the array section if only a section is
  // privatized, otherwise just use the original variable.
  hlfir::Entity privatizedVar = dereferencedVar;
  aiir::Value tempShape;
  llvm::SmallVector<aiir::Value> tempExtents;
  // TODO: while it seems best to allocate as little memory as possible and
  // allocate only the storage for the section, this may actually have drawbacks
  // when the array has static size and can be privatized with an alloca while
  // the section size is dynamic and requires an dynamic allocmem.  Hence, we
  // currently allocate the full array storage in such cases. This could be
  // improved via some kind of threshold if the base array size is large enough
  // to justify doing a dynamic allocation with the hope that it is much
  // smaller.
  bool allocateSection = false;
  bool isDynamicSectionOfStaticSizeArray =
      !bounds.empty() &&
      !fir::hasDynamicSize(dereferencedVar.getElementOrSequenceType()) &&
      !boundsAreAllConstants(bounds);
  if (!bounds.empty() && !isDynamicSectionOfStaticSizeArray) {
    allocateSection = true;
    hlfir::DesignateOp::Subscripts triplets;
    std::tie(tempShape, tempExtents) =
        computeSectionShapeAndExtents(builder, loc, bounds);
    triplets = genTripletsFromAccBounds(builder, loc, bounds, dereferencedVar);
    privatizedVar = genDesignateWithTriplets(builder, loc, dereferencedVar,
                                             triplets, tempShape, tempExtents);
  } else if (privatizedVar.getRank() > 0) {
    aiir::Value shape = hlfir::genShape(loc, builder, privatizedVar);
    tempExtents = hlfir::getExplicitExtentsFromShape(shape, builder);
    tempShape = fir::ShapeOp::create(builder, loc, tempExtents);
  }
  llvm::SmallVector<aiir::Value> typeParams;
  hlfir::genLengthParameters(loc, builder, privatizedVar, typeParams);
  aiir::Type baseType = privatizedVar.getElementOrSequenceType();
  // Step2: Create a temporary allocation for the privatized part.
  aiir::Value alloc;
  if (fir::hasDynamicSize(baseType) ||
      (isPointerOrAllocatable && bounds.empty())) {
    // Note: heap allocation is forced for whole pointers/allocatable so that
    // the private POINTER/ALLOCATABLE can be deallocated/reallocated on the
    // device inside the compute region. It may not be a requirement, and this
    // could be revisited. In practice, this only matters for scalars since
    // array POINTER and ALLOCATABLE always have dynamic size. Constant sections
    // of POINTER/ALLOCATABLE can use alloca since only part of the data is
    // privatized (it makes no sense to deallocate them).
    alloc = builder.createHeapTemporary(loc, baseType, varName, tempExtents,
                                        typeParams);
    needsDestroy = true;
  } else {
    alloc = builder.createTemporary(loc, baseType, varName, tempExtents,
                                    typeParams);
  }
  // Step3: Assign the initial value to the privatized part if any.
  if (initVal) {
    aiir::Value tempEntity = alloc;
    if (fir::hasDynamicSize(baseType))
      tempEntity =
          fir::EmboxOp::create(builder, loc, fir::BoxType::get(baseType), alloc,
                               tempShape, /*slice=*/aiir::Value{}, typeParams);
    hlfir::genNoAliasAssignment(
        loc, builder, hlfir::Entity{initVal}, hlfir::Entity{tempEntity},
        /*emitWorkshareLoop=*/false, /*temporaryLHS=*/true);
  }

  // Making a dynamic allocation of the size of the whole base instead of the
  // section in case of section would lead to improper deallocation because
  // generatePrivateDestroy always deallocates the start of the section when
  // there is a section.
  assert(!(needsDestroy && !bounds.empty() && !allocateSection) &&
         "dynamic allocation of the whole base in case of section is not "
         "expected");

  if (inputVar.getType() == alloc.getType() && !allocateSection)
    return alloc;

  // Step4: reconstruct the input variable from the privatized part:
  // - get a mock base address if the privatized part is a section (so that any
  // addressing of the input variable can be replaced by the same addressing of
  // the privatized part even though the allocated part for the private does not
  // cover all the input variable storage. This is relying on OpenACC
  // constraint that any addressing of such privatized variable inside the
  // construct region can only address the variable inside the privatized
  // section).
  // - reconstruct a descriptor with the same bounds and type parameters as the
  // input if needed.
  // - store this new descriptor in a temporary allocation if the input variable
  // is a POINTER/ALLOCATABLE.
  llvm::SmallVector<aiir::Value> inputVarLowerBounds, inputVarExtents;
  if (dereferencedVar.isArray()) {
    for (int dim = 0; dim < dereferencedVar.getRank(); ++dim) {
      inputVarLowerBounds.push_back(
          hlfir::genLBound(loc, builder, dereferencedVar, dim));
      inputVarExtents.push_back(
          hlfir::genExtent(loc, builder, dereferencedVar, dim));
    }
  }

  aiir::Value privateVarBaseAddr = alloc;
  if (allocateSection) {
    // To compute the mock base address without doing pointer arithmetic,
    // compute: TYPE, TEMP(ZERO_BASED_SECTION_LB:) MOCK_BASE = TEMP(0)
    // This addresses the section "backwards" (0 <= ZERO_BASED_SECTION_LB). This
    // is currently OK, but care should be taken to avoid tripping bound checks
    // if added in the future.
    aiir::Type inputBaseAddrType =
        dereferencedVar.getBoxType().getBaseAddressType();
    aiir::Value tempBaseAddr =
        builder.createConvert(loc, inputBaseAddrType, alloc);
    aiir::Value zero =
        builder.createIntegerConstant(loc, builder.getIndexType(), 0);
    llvm::SmallVector<aiir::Value> lowerBounds;
    llvm::SmallVector<aiir::Value> zeros;
    for (unsigned i = 0; i < bounds.size(); i += 3) {
      lowerBounds.push_back(bounds[i]);
      zeros.push_back(zero);
    }
    aiir::Value offsetShapeShift =
        builder.genShape(loc, lowerBounds, inputVarExtents);
    aiir::Type eleRefType =
        builder.getRefType(privatizedVar.getFortranElementType());
    aiir::Value mockBase = fir::ArrayCoorOp::create(
        builder, loc, eleRefType, tempBaseAddr, offsetShapeShift,
        /*slice=*/aiir::Value{}, /*indices=*/zeros,
        /*typeParams=*/aiir::ValueRange{});
    privateVarBaseAddr =
        builder.createConvert(loc, inputBaseAddrType, mockBase);
  }

  aiir::Value retVal = privateVarBaseAddr;
  if (inputVar.isBoxAddressOrValue()) {
    // Recreate descriptor with same bounds as the input variable.
    aiir::Value shape;
    if (!inputVarExtents.empty())
      shape = builder.genShape(loc, inputVarLowerBounds, inputVarExtents);
    aiir::Value box = fir::EmboxOp::create(builder, loc, inputVar.getBoxType(),
                                           privateVarBaseAddr, shape,
                                           /*slice=*/aiir::Value{}, typeParams);
    if (inputVar.isMutableBox()) {
      aiir::Value boxAlloc =
          fir::AllocaOp::create(builder, loc, inputVar.getBoxType());
      fir::StoreOp::create(builder, loc, box, boxAlloc);
      retVal = boxAlloc;
    } else {
      retVal = box;
    }
  }
  return retVal;
}

template aiir::Value
OpenACCMappableModel<fir::BaseBoxType>::generatePrivateInit(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::MappableType> var, llvm::StringRef varName,
    aiir::ValueRange extents, aiir::Value initVal,
    aiir::acc::VariableInfoAttr varInfo, bool &needsDestroy) const;

template aiir::Value
OpenACCMappableModel<fir::ReferenceType>::generatePrivateInit(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::MappableType> var, llvm::StringRef varName,
    aiir::ValueRange extents, aiir::Value initVal,
    aiir::acc::VariableInfoAttr varInfo, bool &needsDestroy) const;

template aiir::Value OpenACCMappableModel<fir::HeapType>::generatePrivateInit(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::MappableType> var, llvm::StringRef varName,
    aiir::ValueRange extents, aiir::Value initVal,
    aiir::acc::VariableInfoAttr varInfo, bool &needsDestroy) const;

template aiir::Value
OpenACCMappableModel<fir::PointerType>::generatePrivateInit(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::MappableType> var, llvm::StringRef varName,
    aiir::ValueRange extents, aiir::Value initVal,
    aiir::acc::VariableInfoAttr varInfo, bool &needsDestroy) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::generateCopy(
    aiir::Type type, aiir::OpBuilder &aiirBuilder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::MappableType> src,
    aiir::TypedValue<aiir::acc::MappableType> dest, aiir::ValueRange bounds,
    aiir::acc::VariableInfoAttr varInfo) const {
  aiir::ModuleOp mod =
      aiirBuilder.getBlock()->getParent()->getParentOfType<aiir::ModuleOp>();
  assert(mod && "failed to retrieve parent module");
  fir::FirOpBuilder builder(aiirBuilder, mod);
  hlfir::Entity source{src};
  hlfir::Entity destination{dest};

  source = hlfir::derefPointersAndAllocatables(loc, builder, source);
  destination = hlfir::derefPointersAndAllocatables(loc, builder, destination);

  if (!bounds.empty())
    std::tie(source, destination) =
        genArraySectionsInRecipe(builder, loc, bounds, source, destination);
  // The source and the destination of the firstprivate copy cannot alias,
  // the destination is already properly allocated, so a simple assignment
  // can be generated right away to avoid ending-up with runtime calls
  // for arrays of numerical, logical and, character types.
  //
  // The temporary_lhs flag allows indicating that user defined assignments
  // should not be called while copying components, and that the LHS and RHS
  // are known to not alias since the LHS is a created object.
  //
  // TODO: detect cases where user defined assignment is needed and add a TODO.
  // using temporary_lhs allows more aggressive optimizations of simple derived
  // types. Existing compilers supporting OpenACC do not call user defined
  // assignments, some use case is needed to decide what to do.
  source = hlfir::loadTrivialScalar(loc, builder, source);
  hlfir::AssignOp::create(builder, loc, source, destination, /*realloc=*/false,
                          /*keep_lhs_length_if_realloc=*/false,
                          /*temporary_lhs=*/true);
  return true;
}

template bool OpenACCMappableModel<fir::BaseBoxType>::generateCopy(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::VariableInfoAttr) const;
template bool OpenACCMappableModel<fir::ReferenceType>::generateCopy(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::VariableInfoAttr) const;
template bool OpenACCMappableModel<fir::PointerType>::generateCopy(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::VariableInfoAttr) const;
template bool OpenACCMappableModel<fir::HeapType>::generateCopy(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::VariableInfoAttr) const;

template <typename Op>
static aiir::Value genLogicalCombiner(fir::FirOpBuilder &builder,
                                      aiir::Location loc, aiir::Value value1,
                                      aiir::Value value2) {
  aiir::Type i1 = builder.getI1Type();
  aiir::Value v1 = fir::ConvertOp::create(builder, loc, i1, value1);
  aiir::Value v2 = fir::ConvertOp::create(builder, loc, i1, value2);
  aiir::Value combined = Op::create(builder, loc, v1, v2);
  return fir::ConvertOp::create(builder, loc, value1.getType(), combined);
}

static aiir::Value genComparisonCombiner(fir::FirOpBuilder &builder,
                                         aiir::Location loc,
                                         aiir::arith::CmpIPredicate pred,
                                         aiir::Value value1,
                                         aiir::Value value2) {
  aiir::Type i1 = builder.getI1Type();
  aiir::Value v1 = fir::ConvertOp::create(builder, loc, i1, value1);
  aiir::Value v2 = fir::ConvertOp::create(builder, loc, i1, value2);
  aiir::Value add = aiir::arith::CmpIOp::create(builder, loc, pred, v1, v2);
  return fir::ConvertOp::create(builder, loc, value1.getType(), add);
}

static aiir::Value genScalarCombiner(fir::FirOpBuilder &builder,
                                     aiir::Location loc,
                                     aiir::acc::ReductionOperator op,
                                     aiir::Type ty, aiir::Value value1,
                                     aiir::Value value2) {
  value1 = builder.loadIfRef(loc, value1);
  value2 = builder.loadIfRef(loc, value2);
  if (op == aiir::acc::ReductionOperator::AccAdd) {
    if (ty.isIntOrIndex())
      return aiir::arith::AddIOp::create(builder, loc, value1, value2);
    if (aiir::isa<aiir::FloatType>(ty))
      return aiir::arith::AddFOp::create(builder, loc, value1, value2);
    if (auto cmplxTy = aiir::dyn_cast_or_null<aiir::ComplexType>(ty))
      return fir::AddcOp::create(builder, loc, value1, value2);
    TODO(loc, "reduction add type");
  }

  if (op == aiir::acc::ReductionOperator::AccMul) {
    if (ty.isIntOrIndex())
      return aiir::arith::MulIOp::create(builder, loc, value1, value2);
    if (aiir::isa<aiir::FloatType>(ty))
      return aiir::arith::MulFOp::create(builder, loc, value1, value2);
    if (aiir::isa<aiir::ComplexType>(ty))
      return fir::MulcOp::create(builder, loc, value1, value2);
    TODO(loc, "reduction mul type");
  }

  if (op == aiir::acc::ReductionOperator::AccMin ||
      op == aiir::acc::ReductionOperator::AccMinimumf ||
      op == aiir::acc::ReductionOperator::AccMinnumf) {
    Fortran::common::FPMaxminBehavior savedMode = builder.getFPMaxminBehavior();
    if (op == aiir::acc::ReductionOperator::AccMinimumf)
      builder.setFPMaxminBehavior(Fortran::common::FPMaxminBehavior::Extremum);
    else if (op == aiir::acc::ReductionOperator::AccMinnumf)
      builder.setFPMaxminBehavior(
          Fortran::common::FPMaxminBehavior::ExtremeNum);

    aiir::Value result = fir::genMin(builder, loc, {value1, value2});
    builder.setFPMaxminBehavior(savedMode);
    return result;
  }

  if (op == aiir::acc::ReductionOperator::AccMax ||
      op == aiir::acc::ReductionOperator::AccMaximumf ||
      op == aiir::acc::ReductionOperator::AccMaxnumf) {
    Fortran::common::FPMaxminBehavior savedMode = builder.getFPMaxminBehavior();
    if (op == aiir::acc::ReductionOperator::AccMaximumf)
      builder.setFPMaxminBehavior(Fortran::common::FPMaxminBehavior::Extremum);
    else if (op == aiir::acc::ReductionOperator::AccMaxnumf)
      builder.setFPMaxminBehavior(
          Fortran::common::FPMaxminBehavior::ExtremeNum);

    aiir::Value result = fir::genMax(builder, loc, {value1, value2});
    builder.setFPMaxminBehavior(savedMode);
    return result;
  }

  if (op == aiir::acc::ReductionOperator::AccIand)
    return aiir::arith::AndIOp::create(builder, loc, value1, value2);

  if (op == aiir::acc::ReductionOperator::AccIor)
    return aiir::arith::OrIOp::create(builder, loc, value1, value2);

  if (op == aiir::acc::ReductionOperator::AccXor)
    return aiir::arith::XOrIOp::create(builder, loc, value1, value2);

  if (op == aiir::acc::ReductionOperator::AccLand)
    return genLogicalCombiner<aiir::arith::AndIOp>(builder, loc, value1,
                                                   value2);

  if (op == aiir::acc::ReductionOperator::AccLor)
    return genLogicalCombiner<aiir::arith::OrIOp>(builder, loc, value1, value2);

  if (op == aiir::acc::ReductionOperator::AccEqv)
    return genComparisonCombiner(builder, loc, aiir::arith::CmpIPredicate::eq,
                                 value1, value2);

  if (op == aiir::acc::ReductionOperator::AccNeqv)
    return genComparisonCombiner(builder, loc, aiir::arith::CmpIPredicate::ne,
                                 value1, value2);

  TODO(loc, "reduction operator");
}

static bool useAccReductionCombineOp(aiir::Type elementType,
                                     aiir::acc::ReductionOperator op) {
  if (useAccReductionCombineAll)
    return true;
  if (!useAccReductionCombine)
    return false;
  // LOGICAL operators do not have aiir operators and requires FIR specific
  // logic to interpret the TRUE and FALSE values from the storage (implemented
  // in fir.convert to i1).
  if (!llvm::isa<aiir::IntegerType, aiir::FloatType, aiir::ComplexType>(
          elementType))
    return false;
  // MIN/MAX for floating point can have different edge-case behaviors (NANs).
  // Currently the aiir operator does not match the behavior implemented by
  // flang.
  return op != aiir::acc::ReductionOperator::AccMax &&
         op != aiir::acc::ReductionOperator::AccMin;
}

template <typename Ty>
bool OpenACCMappableModel<Ty>::generateCombiner(
    aiir::Type type, aiir::OpBuilder &aiirBuilder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::MappableType> dest,
    aiir::TypedValue<aiir::acc::MappableType> source, aiir::ValueRange bounds,
    aiir::acc::ReductionOperator op, aiir::Attribute fastmathFlags) const {
  aiir::ModuleOp mod =
      aiirBuilder.getBlock()->getParent()->getParentOfType<aiir::ModuleOp>();
  assert(mod && "failed to retrieve parent module");
  fir::FirOpBuilder builder(aiirBuilder, mod);
  if (fastmathFlags)
    if (auto fastMathAttr =
            aiir::dyn_cast<aiir::arith::FastMathFlagsAttr>(fastmathFlags))
      builder.setFastMathFlags(fastMathAttr.getValue());
  // Generate loops that combine and assign the inputs into dest (or array
  // section of the inputs when there are bounds).
  hlfir::Entity srcSection{source};
  hlfir::Entity destSection{dest};
  if (!bounds.empty()) {
    std::tie(srcSection, destSection) =
        genArraySectionsInRecipe(builder, loc, bounds, srcSection, destSection);
  }

  aiir::Type elementType = fir::getFortranElementType(dest.getType());
  auto genKernel =
      [&](aiir::Location l, fir::FirOpBuilder &b, hlfir::Entity destElementAddr,
          hlfir::Entity srcElementAddr, aiir::ArrayAttr accessGroups) -> void {
    assert(!accessGroups && "access groups not expected in acc reductions");
    if (useAccReductionCombineOp(elementType, op)) {
      aiir::acc::ReductionCombineOp::create(builder, loc, destElementAddr,
                                            srcElementAddr, op);
      return;
    }
    hlfir::Entity srcElementValue =
        hlfir::loadTrivialScalar(loc, builder, srcElementAddr);
    hlfir::Entity destElementValue =
        hlfir::loadTrivialScalar(loc, builder, destElementAddr);
    hlfir::Entity combined(genScalarCombiner(
        builder, loc, op, elementType, destElementValue, srcElementValue));
    hlfir::AssignOp::create(builder, loc, combined, destElementAddr,
                            /*realloc=*/false,
                            /*keep_lhs_length_if_realloc=*/false,
                            /*temporary_lhs=*/false);
  };
  hlfir::genNoAliasAssignment(loc, builder, srcSection, destSection,
                              /*emitWorkshareLoop=*/false,
                              /*temporaryLHS=*/false, genKernel);
  return true;
}

template bool OpenACCMappableModel<fir::BaseBoxType>::generateCombiner(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::ReductionOperator op, aiir::Attribute) const;
template bool OpenACCMappableModel<fir::ReferenceType>::generateCombiner(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::ReductionOperator op, aiir::Attribute) const;
template bool OpenACCMappableModel<fir::PointerType>::generateCombiner(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::ReductionOperator op, aiir::Attribute) const;
template bool OpenACCMappableModel<fir::HeapType>::generateCombiner(
    aiir::Type, aiir::OpBuilder &, aiir::Location,
    aiir::TypedValue<aiir::acc::MappableType>,
    aiir::TypedValue<aiir::acc::MappableType>, aiir::ValueRange,
    aiir::acc::ReductionOperator op, aiir::Attribute) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::generatePrivateDestroy(
    aiir::Type type, aiir::OpBuilder &aiirBuilder, aiir::Location loc,
    aiir::Value privatized, aiir::ValueRange bounds,
    aiir::acc::VariableInfoAttr varInfo) const {
  hlfir::Entity inputVar = hlfir::Entity{privatized};
  aiir::ModuleOp mod =
      aiirBuilder.getBlock()->getParent()->getParentOfType<aiir::ModuleOp>();
  assert(mod && "failed to retrieve parent module");
  fir::FirOpBuilder builder(aiirBuilder, mod);
  auto genFreeRawAddress = [&](hlfir::Entity entity) {
    aiir::Value addr = hlfir::genVariableRawAddress(loc, builder, entity);
    aiir::Type heapType =
        fir::HeapType::get(fir::unwrapRefType(addr.getType()));
    if (heapType != addr.getType())
      addr = fir::ConvertOp::create(builder, loc, heapType, addr);
    fir::FreeMemOp::create(builder, loc, addr);
  };
  if (bounds.empty()) {
    genFreeRawAddress(inputVar);
    return true;
  }
  // The input variable is an array section, the base address is not the real
  // allocation. Compute the section base address and deallocate that.
  hlfir::Entity dereferencedVar =
      hlfir::derefPointersAndAllocatables(loc, builder, inputVar);
  hlfir::DesignateOp::Subscripts triplets;
  auto [tempShape, tempExtents] =
      computeSectionShapeAndExtents(builder, loc, bounds);
  (void)tempExtents;
  triplets = genTripletsFromAccBounds(builder, loc, bounds, dereferencedVar);
  hlfir::Entity arraySection = genDesignateWithTriplets(
      builder, loc, dereferencedVar, triplets, tempShape, tempExtents);
  genFreeRawAddress(arraySection);
  return true;
}

template bool OpenACCMappableModel<fir::BaseBoxType>::generatePrivateDestroy(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value privatized, aiir::ValueRange bounds,
    aiir::acc::VariableInfoAttr varInfo) const;
template bool OpenACCMappableModel<fir::ReferenceType>::generatePrivateDestroy(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value privatized, aiir::ValueRange bounds,
    aiir::acc::VariableInfoAttr varInfo) const;
template bool OpenACCMappableModel<fir::HeapType>::generatePrivateDestroy(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value privatized, aiir::ValueRange bounds,
    aiir::acc::VariableInfoAttr varInfo) const;
template bool OpenACCMappableModel<fir::PointerType>::generatePrivateDestroy(
    aiir::Type type, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value privatized, aiir::ValueRange bounds,
    aiir::acc::VariableInfoAttr varInfo) const;

template <typename Ty>
aiir::Value OpenACCPointerLikeModel<Ty>::genAllocate(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    llvm::StringRef varName, aiir::Type varType, aiir::Value originalVar,
    bool &needsFree) const {

  // Unwrap to get the pointee type.
  aiir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
  assert(pointeeTy && "expected pointee type to be extractable");

  // Box types are descriptors that contain both metadata and a pointer to data.
  // The `genAllocate` API is designed for simple allocations and cannot
  // properly handle the dual nature of boxes. Using `generatePrivateInit`
  // instead can allocate both the descriptor and its referenced data. For use
  // cases that require an empty descriptor storage, potentially this could be
  // implemented here.
  if (fir::isa_box_type(pointeeTy))
    return {};

  // Unlimited polymorphic (class(*)) cannot be handled - size unknown
  if (fir::isUnlimitedPolymorphicType(pointeeTy))
    return {};

  // Return null for dynamic size types because the size of the
  // allocation cannot be determined simply from the type.
  if (fir::hasDynamicSize(pointeeTy))
    return {};

  // Use heap allocation for fir.heap, stack allocation for others (fir.ref,
  // fir.ptr, fir.llvm_ptr). For fir.ptr, which is supposed to represent a
  // Fortran pointer type, it feels a bit odd to "allocate" since it is meant
  // to point to an existing entity - but one can imagine where a pointee is
  // privatized - thus it makes sense to issue an allocate.
  aiir::Value allocation;
  if (std::is_same_v<Ty, fir::HeapType>) {
    needsFree = true;
    allocation = fir::AllocMemOp::create(builder, loc, pointeeTy);
  } else {
    needsFree = false;
    allocation = fir::AllocaOp::create(builder, loc, pointeeTy);
  }

  // Convert to the requested pointer type if needed.
  // This means converting from a fir.ref to either a fir.llvm_ptr or a fir.ptr.
  // fir.heap is already correct type in this case.
  if (allocation.getType() != pointer) {
    assert(!(std::is_same_v<Ty, fir::HeapType>) &&
           "fir.heap is already correct type because of allocmem");
    return fir::ConvertOp::create(builder, loc, pointer, allocation);
  }

  return allocation;
}

template aiir::Value OpenACCPointerLikeModel<fir::ReferenceType>::genAllocate(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    llvm::StringRef varName, aiir::Type varType, aiir::Value originalVar,
    bool &needsFree) const;

template aiir::Value OpenACCPointerLikeModel<fir::PointerType>::genAllocate(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    llvm::StringRef varName, aiir::Type varType, aiir::Value originalVar,
    bool &needsFree) const;

template aiir::Value OpenACCPointerLikeModel<fir::HeapType>::genAllocate(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    llvm::StringRef varName, aiir::Type varType, aiir::Value originalVar,
    bool &needsFree) const;

template aiir::Value OpenACCPointerLikeModel<fir::LLVMPointerType>::genAllocate(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    llvm::StringRef varName, aiir::Type varType, aiir::Value originalVar,
    bool &needsFree) const;

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::genFree(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> varToFree,
    aiir::Value allocRes, aiir::Type varType) const {

  // Unwrap to get the pointee type.
  aiir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
  assert(pointeeTy && "expected pointee type to be extractable");

  // Box types contain both a descriptor and data. The `genFree` API
  // handles simple deallocations and cannot properly manage both parts.
  // Using `generatePrivateDestroy` instead can free both the descriptor and
  // its referenced data.
  if (fir::isa_box_type(pointeeTy))
    return false;

  // If pointer type is HeapType, assume it's a heap allocation
  if (std::is_same_v<Ty, fir::HeapType>) {
    fir::FreeMemOp::create(builder, loc, varToFree);
    return true;
  }

  // Use allocRes if provided to determine the allocation type
  aiir::Value valueToInspect = allocRes ? allocRes : varToFree;

  // Strip casts and declare operations to find the original allocation
  aiir::Value strippedValue = fir::acc::getOriginalDef(valueToInspect);
  aiir::Operation *originalAlloc = strippedValue.getDefiningOp();

  // If we found an AllocMemOp (heap allocation), free it
  if (aiir::isa_and_nonnull<fir::AllocMemOp>(originalAlloc)) {
    aiir::Value toFree = varToFree;
    if (!aiir::isa<fir::HeapType>(valueToInspect.getType()))
      toFree = fir::ConvertOp::create(
          builder, loc,
          fir::HeapType::get(varToFree.getType().getElementType()), toFree);
    fir::FreeMemOp::create(builder, loc, toFree);
    return true;
  }

  // If we found an AllocaOp (stack allocation), no deallocation needed
  if (aiir::isa_and_nonnull<fir::AllocaOp>(originalAlloc))
    return true;

  // Unable to determine allocation type
  return false;
}

template bool OpenACCPointerLikeModel<fir::ReferenceType>::genFree(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> varToFree,
    aiir::Value allocRes, aiir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::PointerType>::genFree(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> varToFree,
    aiir::Value allocRes, aiir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::HeapType>::genFree(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> varToFree,
    aiir::Value allocRes, aiir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::genFree(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> varToFree,
    aiir::Value allocRes, aiir::Type varType) const;

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::genCopy(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> destination,
    aiir::TypedValue<aiir::acc::PointerLikeType> source,
    aiir::Type varType) const {

  // Check that source and destination types match
  if (source.getType() != destination.getType())
    return false;

  // Unwrap to get the pointee type.
  aiir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
  assert(pointeeTy && "expected pointee type to be extractable");

  // Box types contain both a descriptor and referenced data. The genCopy API
  // handles simple copies and cannot properly manage both parts.
  if (fir::isa_box_type(pointeeTy))
    return false;

  // Unlimited polymorphic (class(*)) cannot be handled because source and
  // destination types are not known.
  if (fir::isUnlimitedPolymorphicType(pointeeTy))
    return false;

  // Return false for dynamic size types because the copy logic
  // cannot be determined simply from the type.
  if (fir::hasDynamicSize(pointeeTy))
    return false;

  if (fir::isa_trivial(pointeeTy)) {
    auto loadVal = fir::LoadOp::create(builder, loc, source);
    fir::StoreOp::create(builder, loc, loadVal, destination);
  } else {
    hlfir::AssignOp::create(builder, loc, source, destination);
  }
  return true;
}

template bool OpenACCPointerLikeModel<fir::ReferenceType>::genCopy(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> destination,
    aiir::TypedValue<aiir::acc::PointerLikeType> source,
    aiir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::PointerType>::genCopy(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> destination,
    aiir::TypedValue<aiir::acc::PointerLikeType> source,
    aiir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::HeapType>::genCopy(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> destination,
    aiir::TypedValue<aiir::acc::PointerLikeType> source,
    aiir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::genCopy(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> destination,
    aiir::TypedValue<aiir::acc::PointerLikeType> source,
    aiir::Type varType) const;

template <typename Ty>
aiir::Value OpenACCPointerLikeModel<Ty>::genLoad(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> srcPtr,
    aiir::Type valueType) const {

  // Unwrap to get the pointee type.
  aiir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
  assert(pointeeTy && "expected pointee type to be extractable");

  // Box types contain both a descriptor and referenced data. The genLoad API
  // handles simple loads and cannot properly manage both parts.
  if (fir::isa_box_type(pointeeTy))
    return {};

  // Unlimited polymorphic (class(*)) cannot be handled because type is unknown.
  if (fir::isUnlimitedPolymorphicType(pointeeTy))
    return {};

  // Return empty for dynamic size types because the load logic
  // cannot be determined simply from the type.
  if (fir::hasDynamicSize(pointeeTy))
    return {};

  aiir::Value loadedValue = fir::LoadOp::create(builder, loc, srcPtr);

  // If valueType is provided and differs from the loaded type, insert a convert
  if (valueType && loadedValue.getType() != valueType)
    return fir::ConvertOp::create(builder, loc, valueType, loadedValue);

  return loadedValue;
}

template aiir::Value OpenACCPointerLikeModel<fir::ReferenceType>::genLoad(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> srcPtr,
    aiir::Type valueType) const;

template aiir::Value OpenACCPointerLikeModel<fir::PointerType>::genLoad(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> srcPtr,
    aiir::Type valueType) const;

template aiir::Value OpenACCPointerLikeModel<fir::HeapType>::genLoad(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> srcPtr,
    aiir::Type valueType) const;

template aiir::Value OpenACCPointerLikeModel<fir::LLVMPointerType>::genLoad(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::TypedValue<aiir::acc::PointerLikeType> srcPtr,
    aiir::Type valueType) const;

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::genStore(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value valueToStore,
    aiir::TypedValue<aiir::acc::PointerLikeType> destPtr) const {

  // Unwrap to get the pointee type.
  aiir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
  assert(pointeeTy && "expected pointee type to be extractable");

  // Box types contain both a descriptor and referenced data. The genStore API
  // handles simple stores and cannot properly manage both parts.
  if (fir::isa_box_type(pointeeTy))
    return false;

  // Unlimited polymorphic (class(*)) cannot be handled because type is unknown.
  if (fir::isUnlimitedPolymorphicType(pointeeTy))
    return false;

  // Return false for dynamic size types because the store logic
  // cannot be determined simply from the type.
  if (fir::hasDynamicSize(pointeeTy))
    return false;

  // Get the type from the value being stored
  aiir::Type valueType = valueToStore.getType();
  aiir::Value convertedValue = valueToStore;

  // If the value type differs from the pointee type, insert a convert
  if (valueType != pointeeTy)
    convertedValue =
        fir::ConvertOp::create(builder, loc, pointeeTy, valueToStore);

  fir::StoreOp::create(builder, loc, convertedValue, destPtr);
  return true;
}

template bool OpenACCPointerLikeModel<fir::ReferenceType>::genStore(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value valueToStore,
    aiir::TypedValue<aiir::acc::PointerLikeType> destPtr) const;

template bool OpenACCPointerLikeModel<fir::PointerType>::genStore(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value valueToStore,
    aiir::TypedValue<aiir::acc::PointerLikeType> destPtr) const;

template bool OpenACCPointerLikeModel<fir::HeapType>::genStore(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value valueToStore,
    aiir::TypedValue<aiir::acc::PointerLikeType> destPtr) const;

template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::genStore(
    aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
    aiir::Value valueToStore,
    aiir::TypedValue<aiir::acc::PointerLikeType> destPtr) const;

/// Check CUDA attributes on a function argument.
static bool hasCUDADeviceAttrOnFuncArg(aiir::BlockArgument blockArg) {
  auto *owner = blockArg.getOwner();
  if (!owner)
    return false;

  auto *parentOp = owner->getParentOp();
  if (!parentOp)
    return false;

  if (auto funcLike = aiir::dyn_cast<aiir::FunctionOpInterface>(parentOp)) {
    unsigned argIndex = blockArg.getArgNumber();
    if (argIndex < funcLike.getNumArguments())
      if (auto attr = funcLike.getArgAttr(argIndex, cuf::getDataAttrName()))
        if (auto cudaAttr = aiir::dyn_cast<cuf::DataAttributeAttr>(attr))
          return cuf::isDeviceDataAttribute(cudaAttr.getValue());
  }
  return false;
}

/// Shared implementation for checking if a value represents device data.
static bool isDeviceDataImpl(aiir::Value var) {
  // Strip casts to find the underlying value.
  aiir::Value currentVal =
      fir::acc::getOriginalDef(var, /*stripDeclare=*/false);

  if (auto blockArg = aiir::dyn_cast<aiir::BlockArgument>(currentVal))
    return hasCUDADeviceAttrOnFuncArg(blockArg);

  aiir::Operation *defOp = currentVal.getDefiningOp();
  assert(defOp && "expected defining op for non-block-argument value");

  // Check for CUDA attributes on the defining operation.
  if (cuf::hasDeviceDataAttr(defOp))
    return true;

  // Handle operations that access a partial entity - check if the base entity
  // is device data.
  if (auto partialAccess =
          aiir::dyn_cast<aiir::acc::PartialEntityAccessOpInterface>(defOp))
    if (aiir::Value base = partialAccess.getBaseEntity())
      return isDeviceDataImpl(base);

  // Handle fir.embox, fir.rebox, and similar ops via
  // FortranObjectViewOpInterface to check if the underlying source is device
  // data.
  if (auto viewOp = aiir::dyn_cast<fir::FortranObjectViewOpInterface>(defOp))
    if (aiir::Value source = viewOp.getViewSource(defOp->getResult(0)))
      return isDeviceDataImpl(source);

  // Handle address_of - check the referenced global.
  if (auto addrOfIface =
          aiir::dyn_cast<aiir::acc::AddressOfGlobalOpInterface>(defOp)) {
    auto symbol = addrOfIface.getSymbol();
    if (auto global = aiir::SymbolTable::lookupNearestSymbolFrom<
            aiir::acc::GlobalVariableOpInterface>(defOp, symbol))
      return global.isDeviceData();
    return false;
  }

  return false;
}

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::isDeviceData(aiir::Type pointer,
                                               aiir::Value var) const {
  return isDeviceDataImpl(var);
}

template bool OpenACCPointerLikeModel<fir::ReferenceType>::isDeviceData(
    aiir::Type, aiir::Value) const;
template bool
    OpenACCPointerLikeModel<fir::PointerType>::isDeviceData(aiir::Type,
                                                            aiir::Value) const;
template bool
    OpenACCPointerLikeModel<fir::HeapType>::isDeviceData(aiir::Type,
                                                         aiir::Value) const;
template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::isDeviceData(
    aiir::Type, aiir::Value) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::isDeviceData(aiir::Type type,
                                            aiir::Value var) const {
  return isDeviceDataImpl(var);
}

template bool
    OpenACCMappableModel<fir::BaseBoxType>::isDeviceData(aiir::Type,
                                                         aiir::Value) const;
template bool
    OpenACCMappableModel<fir::ReferenceType>::isDeviceData(aiir::Type,
                                                           aiir::Value) const;
template bool
    OpenACCMappableModel<fir::HeapType>::isDeviceData(aiir::Type,
                                                      aiir::Value) const;
template bool
    OpenACCMappableModel<fir::PointerType>::isDeviceData(aiir::Type,
                                                         aiir::Value) const;

std::optional<aiir::arith::AtomicRMWKind>
OpenACCReducibleLogicalModel::getAtomicRMWKind(
    aiir::Type type, aiir::acc::ReductionOperator redOp) const {
  switch (redOp) {
  case aiir::acc::ReductionOperator::AccLand:
    return aiir::arith::AtomicRMWKind::andi;
  case aiir::acc::ReductionOperator::AccLor:
    return aiir::arith::AtomicRMWKind::ori;
  case aiir::acc::ReductionOperator::AccEqv:
  case aiir::acc::ReductionOperator::AccNeqv:
    // Eqv and Neqv are valid for logical types but don't have a direct
    // AtomicRMWKind mapping yet.
    return std::nullopt;
  default:
    // Other reduction operators are not valid for logical types.
    return std::nullopt;
  }
}

} // namespace fir::acc
