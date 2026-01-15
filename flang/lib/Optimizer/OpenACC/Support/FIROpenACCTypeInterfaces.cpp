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
#include "flang/Optimizer/Support/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir::acc {

template <typename Ty>
mlir::TypedValue<mlir::acc::PointerLikeType>
OpenACCMappableModel<Ty>::getVarPtr(mlir::Type type, mlir::Value var) const {
  if (auto ptr =
          mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(var))
    return ptr;

  if (auto load = mlir::dyn_cast_if_present<fir::LoadOp>(var.getDefiningOp())) {
    // All FIR reference types implement the PointerLikeType interface.
    return mlir::cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(
        load.getMemref());
  }

  return {};
}

template mlir::TypedValue<mlir::acc::PointerLikeType>
OpenACCMappableModel<fir::BaseBoxType>::getVarPtr(mlir::Type type,
                                                  mlir::Value var) const;

template mlir::TypedValue<mlir::acc::PointerLikeType>
OpenACCMappableModel<fir::ReferenceType>::getVarPtr(mlir::Type type,
                                                    mlir::Value var) const;

template mlir::TypedValue<mlir::acc::PointerLikeType>
OpenACCMappableModel<fir::HeapType>::getVarPtr(mlir::Type type,
                                               mlir::Value var) const;

template mlir::TypedValue<mlir::acc::PointerLikeType>
OpenACCMappableModel<fir::PointerType>::getVarPtr(mlir::Type type,
                                                  mlir::Value var) const;

template <typename Ty>
std::optional<llvm::TypeSize> OpenACCMappableModel<Ty>::getSizeInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const {
  // TODO: Bounds operation affect the size - add support to take them
  // into account.
  if (!accBounds.empty())
    return {};

  // Class-type is either a polymorphic or unlimited polymorphic. In the latter
  // case, the size is not computable. But in the former it should be - however,
  // fir::getTypeSizeAndAlignment does not support polymorphic types.
  if (mlir::isa<fir::ClassType>(type)) {
    return {};
  }

  // When requesting the size of a box entity or a reference, the intent
  // is to get the size of the data that it is referring to.
  mlir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  assert(eleTy && "expect to be able to unwrap the element type");

  // If the type enclosed is a mappable type, then have it provide the size.
  if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(eleTy))
    return mappableTy.getSizeInBytes(var, accBounds, dataLayout);

  // Dynamic extents or unknown ranks generally do not have compile-time
  // computable dimensions.
  auto seqType = mlir::dyn_cast<fir::SequenceType>(eleTy);
  if (seqType && (seqType.hasDynamicExtents() || seqType.hasUnknownShape()))
    return {};

  // Attempt to find an operation that a lookup for KindMapping can be done
  // from.
  mlir::Operation *kindMapSrcOp = var.getDefiningOp();
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
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::ReferenceType>::getSizeInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::HeapType>::getSizeInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::PointerType>::getSizeInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template <typename Ty>
std::optional<int64_t> OpenACCMappableModel<Ty>::getOffsetInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const {
  // TODO: Bounds operation affect the offset - add support to take them
  // into account.
  if (!accBounds.empty())
    return {};

  // Class-type does not behave like a normal box because it does not hold an
  // element type. Thus special handle it here.
  if (mlir::isa<fir::ClassType>(type)) {
    // The pointer to the class-type is always at the start address.
    return {0};
  }

  mlir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  assert(eleTy && "expect to be able to unwrap the element type");

  // If the type enclosed is a mappable type, then have it provide the offset.
  if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(eleTy))
    return mappableTy.getOffsetInBytes(var, accBounds, dataLayout);

  // Dynamic extents (aka descriptor-based arrays) - may have a offset.
  // For example, a negative stride may mean a negative offset to compute the
  // start of array.
  auto seqType = mlir::dyn_cast<fir::SequenceType>(eleTy);
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
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template std::optional<int64_t>
OpenACCMappableModel<fir::ReferenceType>::getOffsetInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template std::optional<int64_t>
OpenACCMappableModel<fir::HeapType>::getOffsetInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template std::optional<int64_t>
OpenACCMappableModel<fir::PointerType>::getOffsetInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::hasUnknownDimensions(mlir::Type type) const {
  assert(fir::isa_ref_type(type) && "expected FIR reference type");
  return fir::hasDynamicSize(fir::unwrapRefType(type));
}

template bool OpenACCMappableModel<fir::ReferenceType>::hasUnknownDimensions(
    mlir::Type type) const;

template bool OpenACCMappableModel<fir::HeapType>::hasUnknownDimensions(
    mlir::Type type) const;

template bool OpenACCMappableModel<fir::PointerType>::hasUnknownDimensions(
    mlir::Type type) const;

template <>
bool OpenACCMappableModel<fir::BaseBoxType>::hasUnknownDimensions(
    mlir::Type type) const {
  // Descriptor-based entities have dimensions encoded.
  return false;
}

static llvm::SmallVector<mlir::Value>
generateSeqTyAccBounds(fir::SequenceType seqType, mlir::Value var,
                       mlir::OpBuilder &builder) {
  assert((mlir::isa<mlir::acc::PointerLikeType>(var.getType()) ||
          mlir::isa<mlir::acc::MappableType>(var.getType())) &&
         "must be pointer-like or mappable");
  fir::FirOpBuilder firBuilder(builder, var.getDefiningOp());
  mlir::Location loc = var.getLoc();

  if (seqType.hasDynamicExtents() || seqType.hasUnknownShape()) {
    if (auto boxAddr =
            mlir::dyn_cast_if_present<fir::BoxAddrOp>(var.getDefiningOp())) {
      mlir::Value box = boxAddr.getVal();
      auto res =
          hlfir::translateToExtendedValue(loc, firBuilder, hlfir::Entity(box));
      fir::ExtendedValue exv = res.first;
      mlir::Value boxRef = box;
      if (auto boxPtr = mlir::cast<mlir::acc::MappableType>(box.getType())
                            .getVarPtr(box)) {
        boxRef = boxPtr;
      }
      // TODO: Handle Fortran optional.
      const mlir::Value isPresent;
      fir::factory::AddrAndBoundsInfo info(box, boxRef, isPresent,
                                           box.getType());
      return fir::factory::genBoundsOpsFromBox<mlir::acc::DataBoundsOp,
                                               mlir::acc::DataBoundsType>(
          firBuilder, loc, exv, info);
    }

    if (mlir::isa<hlfir::DeclareOp, fir::DeclareOp>(var.getDefiningOp())) {
      mlir::Value zero =
          firBuilder.createIntegerConstant(loc, builder.getIndexType(), 0);
      mlir::Value one =
          firBuilder.createIntegerConstant(loc, builder.getIndexType(), 1);

      mlir::Value shape;
      if (auto declareOp =
              mlir::dyn_cast_if_present<fir::DeclareOp>(var.getDefiningOp()))
        shape = declareOp.getShape();
      else if (auto declareOp = mlir::dyn_cast_if_present<hlfir::DeclareOp>(
                   var.getDefiningOp()))
        shape = declareOp.getShape();

      const bool strideIncludeLowerExtent = true;

      llvm::SmallVector<mlir::Value> accBounds;
      if (auto shapeOp =
              mlir::dyn_cast_if_present<fir::ShapeOp>(shape.getDefiningOp())) {
        mlir::Value cummulativeExtent = one;
        for (auto extent : shapeOp.getExtents()) {
          mlir::Value upperbound =
              mlir::arith::SubIOp::create(builder, loc, extent, one);
          mlir::Value stride = one;
          if (strideIncludeLowerExtent) {
            stride = cummulativeExtent;
            cummulativeExtent = mlir::arith::MulIOp::create(
                builder, loc, cummulativeExtent, extent);
          }
          auto accBound = mlir::acc::DataBoundsOp::create(
              builder, loc,
              mlir::acc::DataBoundsType::get(builder.getContext()),
              /*lowerbound=*/zero, /*upperbound=*/upperbound,
              /*extent=*/extent, /*stride=*/stride, /*strideInBytes=*/false,
              /*startIdx=*/one);
          accBounds.push_back(accBound);
        }
      } else if (auto shapeShiftOp =
                     mlir::dyn_cast_if_present<fir::ShapeShiftOp>(
                         shape.getDefiningOp())) {
        mlir::Value lowerbound;
        mlir::Value cummulativeExtent = one;
        for (auto [idx, val] : llvm::enumerate(shapeShiftOp.getPairs())) {
          if (idx % 2 == 0) {
            lowerbound = val;
          } else {
            mlir::Value extent = val;
            mlir::Value upperbound =
                mlir::arith::SubIOp::create(builder, loc, extent, one);
            mlir::Value stride = one;
            if (strideIncludeLowerExtent) {
              stride = cummulativeExtent;
              cummulativeExtent = mlir::arith::MulIOp::create(
                  builder, loc, cummulativeExtent, extent);
            }
            auto accBound = mlir::acc::DataBoundsOp::create(
                builder, loc,
                mlir::acc::DataBoundsType::get(builder.getContext()),
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

    assert(false && "array with unknown dimension expected to have descriptor");
    return {};
  }

  // TODO: Detect assumed-size case.
  const bool isAssumedSize = false;
  auto valToCheck = var;
  if (auto boxAddr =
          mlir::dyn_cast_if_present<fir::BoxAddrOp>(var.getDefiningOp())) {
    valToCheck = boxAddr.getVal();
  }
  auto res = hlfir::translateToExtendedValue(loc, firBuilder,
                                             hlfir::Entity(valToCheck));
  fir::ExtendedValue exv = res.first;
  return fir::factory::genBaseBoundsOps<mlir::acc::DataBoundsOp,
                                        mlir::acc::DataBoundsType>(
      firBuilder, loc, exv,
      /*isAssumedSize=*/isAssumedSize);
}

template <typename Ty>
llvm::SmallVector<mlir::Value>
OpenACCMappableModel<Ty>::generateAccBounds(mlir::Type type, mlir::Value var,
                                            mlir::OpBuilder &builder) const {
  // acc bounds only make sense for arrays - thus look for sequence type.
  mlir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  if (auto seqTy = mlir::dyn_cast_if_present<fir::SequenceType>(eleTy)) {
    return generateSeqTyAccBounds(seqTy, var, builder);
  }

  return {};
}

template llvm::SmallVector<mlir::Value>
OpenACCMappableModel<fir::BaseBoxType>::generateAccBounds(
    mlir::Type type, mlir::Value var, mlir::OpBuilder &builder) const;

template llvm::SmallVector<mlir::Value>
OpenACCMappableModel<fir::ReferenceType>::generateAccBounds(
    mlir::Type type, mlir::Value var, mlir::OpBuilder &builder) const;

template llvm::SmallVector<mlir::Value>
OpenACCMappableModel<fir::HeapType>::generateAccBounds(
    mlir::Type type, mlir::Value var, mlir::OpBuilder &builder) const;

template llvm::SmallVector<mlir::Value>
OpenACCMappableModel<fir::PointerType>::generateAccBounds(
    mlir::Type type, mlir::Value var, mlir::OpBuilder &builder) const;

static mlir::Value
getBaseRef(mlir::TypedValue<mlir::acc::PointerLikeType> varPtr) {
  // If there is no defining op - the unwrapped reference is the base one.
  mlir::Operation *op = varPtr.getDefiningOp();
  if (!op)
    return varPtr;

  // Look to find if this value originates from an interior pointer
  // calculation op.
  mlir::Value baseRef =
      llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
          .Case<fir::DeclareOp>([&](auto op) {
            // If this declare binds a view with an underlying storage operand,
            // treat that storage as the base reference. Otherwise, fall back
            // to the declared memref.
            if (auto storage = op.getStorage())
              return storage;
            return mlir::Value(varPtr);
          })
          .Case<hlfir::DesignateOp>([&](auto op) {
            // Get the base object.
            return op.getMemref();
          })
          .Case<fir::ArrayCoorOp, fir::cg::XArrayCoorOp>([&](auto op) {
            // Get the base array on which the coordinate is being applied.
            return op.getMemref();
          })
          .Case<fir::CoordinateOp>([&](auto op) {
            // For coordinate operation which is applied on derived type
            // object, get the base object.
            return op.getRef();
          })
          .Case<fir::ConvertOp>([&](auto op) -> mlir::Value {
            // Strip the conversion and recursively check the operand
            if (auto ptrLikeOperand = mlir::dyn_cast_if_present<
                    mlir::TypedValue<mlir::acc::PointerLikeType>>(
                    op.getValue()))
              return getBaseRef(ptrLikeOperand);
            return varPtr;
          })
          .Default([&](mlir::Operation *) { return varPtr; });

  return baseRef;
}

static bool isScalarLike(mlir::Type type) {
  return fir::isa_trivial(type) || fir::isa_ref_type(type);
}

static bool isArrayLike(mlir::Type type) {
  return mlir::isa<fir::SequenceType>(type);
}

static bool isCompositeLike(mlir::Type type) {
  // class(*) is not a composite type since it does not have a determined type.
  if (fir::isUnlimitedPolymorphicType(type))
    return false;

  return mlir::isa<fir::RecordType, fir::ClassType, mlir::TupleType>(type);
}

static mlir::acc::VariableTypeCategory
categorizeElemType(mlir::Type enclosingTy, mlir::Type eleTy, mlir::Value var) {
  // If the type enclosed is a mappable type, then have it provide the type
  // category.
  if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(eleTy))
    return mappableTy.getTypeCategory(var);

  // For all arrays, despite whether they are allocatable, pointer, assumed,
  // etc, we'd like to categorize them as "array".
  if (isArrayLike(eleTy))
    return mlir::acc::VariableTypeCategory::array;

  if (isCompositeLike(eleTy))
    return mlir::acc::VariableTypeCategory::composite;
  if (mlir::isa<fir::BoxType>(enclosingTy)) {
    // Even if we have a scalar type - simply because it is wrapped in a box
    // we want to categorize it as "nonscalar". Anything else would've been
    // non-scalar anyway.
    return mlir::acc::VariableTypeCategory::nonscalar;
  }
  if (isScalarLike(eleTy))
    return mlir::acc::VariableTypeCategory::scalar;
  if (mlir::isa<fir::CharacterType, mlir::FunctionType>(eleTy))
    return mlir::acc::VariableTypeCategory::nonscalar;
  // Assumed-type (type(*))does not have a determined type that can be
  // categorized.
  if (mlir::isa<mlir::NoneType>(eleTy))
    return mlir::acc::VariableTypeCategory::uncategorized;
  // "pointers" - in the sense of raw address point-of-view, are considered
  // scalars.
  if (mlir::isa<fir::LLVMPointerType>(eleTy))
    return mlir::acc::VariableTypeCategory::scalar;

  // Without further checking, this type cannot be categorized.
  return mlir::acc::VariableTypeCategory::uncategorized;
}

template <typename Ty>
mlir::acc::VariableTypeCategory
OpenACCMappableModel<Ty>::getTypeCategory(mlir::Type type,
                                          mlir::Value var) const {
  // FIR uses operations to compute interior pointers.
  // So for example, an array element or composite field access to a float
  // value would both be represented as !fir.ref<f32>. We do not want to treat
  // such a reference as a scalar. Thus unwrap interior pointer calculations.
  mlir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);
  if (eleTy && isScalarLike(eleTy)) {
    if (auto ptrLikeVar = mlir::dyn_cast_if_present<
            mlir::TypedValue<mlir::acc::PointerLikeType>>(var)) {
      auto baseRef = getBaseRef(ptrLikeVar);
      if (baseRef != var) {
        type = baseRef.getType();
        if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(type))
          return mappableTy.getTypeCategory(baseRef);
      }
    }
  }

  // Class-type does not behave like a normal box because it does not hold an
  // element type. Thus special handle it here.
  if (mlir::isa<fir::ClassType>(type)) {
    // class(*) is not a composite type since it does not have a determined
    // type.
    if (fir::isUnlimitedPolymorphicType(type))
      return mlir::acc::VariableTypeCategory::uncategorized;
    return mlir::acc::VariableTypeCategory::composite;
  }

  assert(eleTy && "expect to be able to unwrap the element type");
  return categorizeElemType(type, eleTy, var);
}

template mlir::acc::VariableTypeCategory
OpenACCMappableModel<fir::BaseBoxType>::getTypeCategory(mlir::Type type,
                                                        mlir::Value var) const;

template mlir::acc::VariableTypeCategory
OpenACCMappableModel<fir::ReferenceType>::getTypeCategory(
    mlir::Type type, mlir::Value var) const;

template mlir::acc::VariableTypeCategory
OpenACCMappableModel<fir::HeapType>::getTypeCategory(mlir::Type type,
                                                     mlir::Value var) const;

template mlir::acc::VariableTypeCategory
OpenACCMappableModel<fir::PointerType>::getTypeCategory(mlir::Type type,
                                                        mlir::Value var) const;

static mlir::acc::VariableTypeCategory
categorizePointee(mlir::Type pointer,
                  mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
                  mlir::Type varType) {
  // FIR uses operations to compute interior pointers.
  // So for example, an array element or composite field access to a float
  // value would both be represented as !fir.ref<f32>. We do not want to treat
  // such a reference as a scalar. Thus unwrap interior pointer calculations.
  auto baseRef = getBaseRef(varPtr);

  if (auto mappableTy =
          mlir::dyn_cast<mlir::acc::MappableType>(baseRef.getType()))
    return mappableTy.getTypeCategory(baseRef);

  // It must be a pointer-like type since it is not a MappableType.
  auto ptrLikeTy = mlir::cast<mlir::acc::PointerLikeType>(baseRef.getType());
  mlir::Type eleTy = ptrLikeTy.getElementType();
  return categorizeElemType(pointer, eleTy, varPtr);
}

template <>
mlir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::ReferenceType>::getPointeeTypeCategory(
    mlir::Type pointer, mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
    mlir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

template <>
mlir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::PointerType>::getPointeeTypeCategory(
    mlir::Type pointer, mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
    mlir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

template <>
mlir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::HeapType>::getPointeeTypeCategory(
    mlir::Type pointer, mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
    mlir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

template <>
mlir::acc::VariableTypeCategory
OpenACCPointerLikeModel<fir::LLVMPointerType>::getPointeeTypeCategory(
    mlir::Type pointer, mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
    mlir::Type varType) const {
  return categorizePointee(pointer, varPtr, varType);
}

static hlfir::Entity
genDesignateWithTriplets(fir::FirOpBuilder &builder, mlir::Location loc,
                         hlfir::Entity &entity,
                         hlfir::DesignateOp::Subscripts &triplets,
                         mlir::Value shape, mlir::ValueRange extents) {
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, entity, lenParams);

  // Compute result type of array section.
  fir::SequenceType::Shape resultTypeShape;
  bool shapeIsConstant = true;
  for (mlir::Value extent : extents) {
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
  mlir::Type eleTy = entity.getFortranElementType();
  auto seqTy = fir::SequenceType::get(resultTypeShape, eleTy);
  bool isVolatile = fir::isa_volatile_type(entity.getType());
  bool resultNeedsBox =
      llvm::isa<fir::BaseBoxType>(entity.getType()) || !shapeIsConstant;
  bool isPolymorphic = fir::isPolymorphicType(entity.getType());
  mlir::Type resultType;
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
      /*componentShape=*/mlir::Value{}, triplets,
      /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt, shape,
      lenParams);
  return hlfir::Entity{designate.getResult()};
}

// Designate uses triplets based on object lower bounds while acc.bounds are
// zero based. This helper shift the bounds to create the designate triplets.
static hlfir::DesignateOp::Subscripts
genTripletsFromAccBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                         const llvm::SmallVector<mlir::Value> &accBounds,
                         hlfir::Entity entity) {
  assert(entity.getRank() * 3 == static_cast<int>(accBounds.size()) &&
         "must get lb,ub,step for each dimension");
  hlfir::DesignateOp::Subscripts triplets;
  for (unsigned i = 0; i < accBounds.size(); i += 3) {
    mlir::Value lb = hlfir::genLBound(loc, builder, entity, i / 3);
    lb = builder.createConvert(loc, accBounds[i].getType(), lb);
    assert(accBounds[i].getType() == accBounds[i + 1].getType() &&
           "mix of integer types in triplets");
    mlir::Value sliceLB =
        builder.createOrFold<mlir::arith::AddIOp>(loc, accBounds[i], lb);
    mlir::Value sliceUB =
        builder.createOrFold<mlir::arith::AddIOp>(loc, accBounds[i + 1], lb);
    triplets.emplace_back(
        hlfir::DesignateOp::Triplet{sliceLB, sliceUB, accBounds[i + 2]});
  }
  return triplets;
}

static std::pair<mlir::Value, llvm::SmallVector<mlir::Value>>
computeSectionShapeAndExtents(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::ValueRange bounds) {
  llvm::SmallVector<mlir::Value> extents;
  // Compute the fir.shape of the array section and the triplets to create
  // hlfir.designate.
  mlir::Type idxTy = builder.getIndexType();
  for (unsigned i = 0; i + 2 < bounds.size(); i += 3)
    extents.push_back(builder.genExtentFromTriplet(
        loc, bounds[i], bounds[i + 1], bounds[i + 2], idxTy, /*fold=*/true));
  mlir::Value shape = fir::ShapeOp::create(builder, loc, extents);
  return {shape, extents};
}

static std::pair<hlfir::Entity, hlfir::Entity>
genArraySectionsInRecipe(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange bounds, hlfir::Entity lhs,
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

static bool boundsAreAllConstants(mlir::ValueRange bounds) {
  for (mlir::Value bound : bounds)
    if (!fir::getIntIfConstant(bound).has_value())
      return false;
  return true;
}

template <typename Ty>
mlir::Value OpenACCMappableModel<Ty>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &mlirBuilder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange bounds, mlir::Value initVal, bool &needsDestroy) const {
  mlir::ModuleOp mod = mlirBuilder.getInsertionBlock()
                           ->getParent()
                           ->getParentOfType<mlir::ModuleOp>();
  assert(mod && "failed to retrieve ModuleOp");
  fir::FirOpBuilder builder(mlirBuilder, mod);

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
  mlir::Value tempShape;
  llvm::SmallVector<mlir::Value> tempExtents;
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
    mlir::Value shape = hlfir::genShape(loc, builder, privatizedVar);
    tempExtents = hlfir::getExplicitExtentsFromShape(shape, builder);
    tempShape = fir::ShapeOp::create(builder, loc, tempExtents);
  }
  llvm::SmallVector<mlir::Value> typeParams;
  hlfir::genLengthParameters(loc, builder, privatizedVar, typeParams);
  mlir::Type baseType = privatizedVar.getElementOrSequenceType();
  // Step2: Create a temporary allocation for the privatized part.
  mlir::Value alloc;
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
    mlir::Value tempEntity = alloc;
    if (fir::hasDynamicSize(baseType))
      tempEntity =
          fir::EmboxOp::create(builder, loc, fir::BoxType::get(baseType), alloc,
                               tempShape, /*slice=*/mlir::Value{}, typeParams);
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
  llvm::SmallVector<mlir::Value> inputVarLowerBounds, inputVarExtents;
  if (dereferencedVar.isArray()) {
    for (int dim = 0; dim < dereferencedVar.getRank(); ++dim) {
      inputVarLowerBounds.push_back(
          hlfir::genLBound(loc, builder, dereferencedVar, dim));
      inputVarExtents.push_back(
          hlfir::genExtent(loc, builder, dereferencedVar, dim));
    }
  }

  mlir::Value privateVarBaseAddr = alloc;
  if (allocateSection) {
    // To compute the mock base address without doing pointer arithmetic,
    // compute: TYPE, TEMP(ZERO_BASED_SECTION_LB:) MOCK_BASE = TEMP(0)
    // This addresses the section "backwards" (0 <= ZERO_BASED_SECTION_LB). This
    // is currently OK, but care should be taken to avoid tripping bound checks
    // if added in the future.
    mlir::Type inputBaseAddrType =
        dereferencedVar.getBoxType().getBaseAddressType();
    mlir::Value tempBaseAddr =
        builder.createConvert(loc, inputBaseAddrType, alloc);
    mlir::Value zero =
        builder.createIntegerConstant(loc, builder.getIndexType(), 0);
    llvm::SmallVector<mlir::Value> lowerBounds;
    llvm::SmallVector<mlir::Value> zeros;
    for (unsigned i = 0; i < bounds.size(); i += 3) {
      lowerBounds.push_back(bounds[i]);
      zeros.push_back(zero);
    }
    mlir::Value offsetShapeShift =
        builder.genShape(loc, lowerBounds, inputVarExtents);
    mlir::Type eleRefType =
        builder.getRefType(privatizedVar.getFortranElementType());
    mlir::Value mockBase = fir::ArrayCoorOp::create(
        builder, loc, eleRefType, tempBaseAddr, offsetShapeShift,
        /*slice=*/mlir::Value{}, /*indices=*/zeros,
        /*typeParams=*/mlir::ValueRange{});
    privateVarBaseAddr =
        builder.createConvert(loc, inputBaseAddrType, mockBase);
  }

  mlir::Value retVal = privateVarBaseAddr;
  if (inputVar.isBoxAddressOrValue()) {
    // Recreate descriptor with same bounds as the input variable.
    mlir::Value shape;
    if (!inputVarExtents.empty())
      shape = builder.genShape(loc, inputVarLowerBounds, inputVarExtents);
    mlir::Value box = fir::EmboxOp::create(builder, loc, inputVar.getBoxType(),
                                           privateVarBaseAddr, shape,
                                           /*slice=*/mlir::Value{}, typeParams);
    if (inputVar.isMutableBox()) {
      mlir::Value boxAlloc =
          fir::AllocaOp::create(builder, loc, inputVar.getBoxType());
      fir::StoreOp::create(builder, loc, box, boxAlloc);
      retVal = boxAlloc;
    } else {
      retVal = box;
    }
  }
  return retVal;
}

template mlir::Value
OpenACCMappableModel<fir::BaseBoxType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal, bool &needsDestroy) const;

template mlir::Value
OpenACCMappableModel<fir::ReferenceType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal, bool &needsDestroy) const;

template mlir::Value OpenACCMappableModel<fir::HeapType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal, bool &needsDestroy) const;

template mlir::Value
OpenACCMappableModel<fir::PointerType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal, bool &needsDestroy) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::generateCopy(
    mlir::Type type, mlir::OpBuilder &mlirBuilder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> src,
    mlir::TypedValue<mlir::acc::MappableType> dest,
    mlir::ValueRange bounds) const {
  mlir::ModuleOp mod =
      mlirBuilder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  assert(mod && "failed to retrieve parent module");
  fir::FirOpBuilder builder(mlirBuilder, mod);
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
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange) const;
template bool OpenACCMappableModel<fir::ReferenceType>::generateCopy(
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange) const;
template bool OpenACCMappableModel<fir::PointerType>::generateCopy(
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange) const;
template bool OpenACCMappableModel<fir::HeapType>::generateCopy(
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange) const;

template <typename Op>
static mlir::Value genLogicalCombiner(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value value1,
                                      mlir::Value value2) {
  mlir::Type i1 = builder.getI1Type();
  mlir::Value v1 = fir::ConvertOp::create(builder, loc, i1, value1);
  mlir::Value v2 = fir::ConvertOp::create(builder, loc, i1, value2);
  mlir::Value combined = Op::create(builder, loc, v1, v2);
  return fir::ConvertOp::create(builder, loc, value1.getType(), combined);
}

static mlir::Value genComparisonCombiner(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::arith::CmpIPredicate pred,
                                         mlir::Value value1,
                                         mlir::Value value2) {
  mlir::Type i1 = builder.getI1Type();
  mlir::Value v1 = fir::ConvertOp::create(builder, loc, i1, value1);
  mlir::Value v2 = fir::ConvertOp::create(builder, loc, i1, value2);
  mlir::Value add = mlir::arith::CmpIOp::create(builder, loc, pred, v1, v2);
  return fir::ConvertOp::create(builder, loc, value1.getType(), add);
}

static mlir::Value genScalarCombiner(fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::acc::ReductionOperator op,
                                     mlir::Type ty, mlir::Value value1,
                                     mlir::Value value2) {
  value1 = builder.loadIfRef(loc, value1);
  value2 = builder.loadIfRef(loc, value2);
  if (op == mlir::acc::ReductionOperator::AccAdd) {
    if (ty.isIntOrIndex())
      return mlir::arith::AddIOp::create(builder, loc, value1, value2);
    if (mlir::isa<mlir::FloatType>(ty))
      return mlir::arith::AddFOp::create(builder, loc, value1, value2);
    if (auto cmplxTy = mlir::dyn_cast_or_null<mlir::ComplexType>(ty))
      return fir::AddcOp::create(builder, loc, value1, value2);
    TODO(loc, "reduction add type");
  }

  if (op == mlir::acc::ReductionOperator::AccMul) {
    if (ty.isIntOrIndex())
      return mlir::arith::MulIOp::create(builder, loc, value1, value2);
    if (mlir::isa<mlir::FloatType>(ty))
      return mlir::arith::MulFOp::create(builder, loc, value1, value2);
    if (mlir::isa<mlir::ComplexType>(ty))
      return fir::MulcOp::create(builder, loc, value1, value2);
    TODO(loc, "reduction mul type");
  }

  if (op == mlir::acc::ReductionOperator::AccMin)
    return fir::genMin(builder, loc, {value1, value2});

  if (op == mlir::acc::ReductionOperator::AccMax)
    return fir::genMax(builder, loc, {value1, value2});

  if (op == mlir::acc::ReductionOperator::AccIand)
    return mlir::arith::AndIOp::create(builder, loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccIor)
    return mlir::arith::OrIOp::create(builder, loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccXor)
    return mlir::arith::XOrIOp::create(builder, loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccLand)
    return genLogicalCombiner<mlir::arith::AndIOp>(builder, loc, value1,
                                                   value2);

  if (op == mlir::acc::ReductionOperator::AccLor)
    return genLogicalCombiner<mlir::arith::OrIOp>(builder, loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccEqv)
    return genComparisonCombiner(builder, loc, mlir::arith::CmpIPredicate::eq,
                                 value1, value2);

  if (op == mlir::acc::ReductionOperator::AccNeqv)
    return genComparisonCombiner(builder, loc, mlir::arith::CmpIPredicate::ne,
                                 value1, value2);

  TODO(loc, "reduction operator");
}

template <typename Ty>
bool OpenACCMappableModel<Ty>::generateCombiner(
    mlir::Type type, mlir::OpBuilder &mlirBuilder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> dest,
    mlir::TypedValue<mlir::acc::MappableType> source, mlir::ValueRange bounds,
    mlir::acc::ReductionOperator op, mlir::Attribute fastmathFlags) const {
  mlir::ModuleOp mod =
      mlirBuilder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  assert(mod && "failed to retrieve parent module");
  fir::FirOpBuilder builder(mlirBuilder, mod);
  if (fastmathFlags)
    if (auto fastMathAttr =
            mlir::dyn_cast<mlir::arith::FastMathFlagsAttr>(fastmathFlags))
      builder.setFastMathFlags(fastMathAttr.getValue());
  // Generate loops that combine and assign the inputs into dest (or array
  // section of the inputs when there are bounds).
  hlfir::Entity srcSection{source};
  hlfir::Entity destSection{dest};
  if (!bounds.empty()) {
    std::tie(srcSection, destSection) =
        genArraySectionsInRecipe(builder, loc, bounds, srcSection, destSection);
  }

  mlir::Type elementType = fir::getFortranElementType(dest.getType());
  auto genKernel = [&](mlir::Location l, fir::FirOpBuilder &b,
                       hlfir::Entity srcElementValue,
                       hlfir::Entity destElementValue) -> hlfir::Entity {
    return hlfir::Entity{genScalarCombiner(builder, loc, op, elementType,
                                           srcElementValue, destElementValue)};
  };
  hlfir::genNoAliasAssignment(loc, builder, srcSection, destSection,
                              /*emitWorkshareLoop=*/false,
                              /*temporaryLHS=*/false, genKernel);
  return true;
}

template bool OpenACCMappableModel<fir::BaseBoxType>::generateCombiner(
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange,
    mlir::acc::ReductionOperator op, mlir::Attribute) const;
template bool OpenACCMappableModel<fir::ReferenceType>::generateCombiner(
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange,
    mlir::acc::ReductionOperator op, mlir::Attribute) const;
template bool OpenACCMappableModel<fir::PointerType>::generateCombiner(
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange,
    mlir::acc::ReductionOperator op, mlir::Attribute) const;
template bool OpenACCMappableModel<fir::HeapType>::generateCombiner(
    mlir::Type, mlir::OpBuilder &, mlir::Location,
    mlir::TypedValue<mlir::acc::MappableType>,
    mlir::TypedValue<mlir::acc::MappableType>, mlir::ValueRange,
    mlir::acc::ReductionOperator op, mlir::Attribute) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::generatePrivateDestroy(
    mlir::Type type, mlir::OpBuilder &mlirBuilder, mlir::Location loc,
    mlir::Value privatized, mlir::ValueRange bounds) const {
  hlfir::Entity inputVar = hlfir::Entity{privatized};
  mlir::ModuleOp mod =
      mlirBuilder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  assert(mod && "failed to retrieve parent module");
  fir::FirOpBuilder builder(mlirBuilder, mod);
  auto genFreeRawAddress = [&](hlfir::Entity entity) {
    mlir::Value addr = hlfir::genVariableRawAddress(loc, builder, entity);
    mlir::Type heapType =
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
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value privatized, mlir::ValueRange bounds) const;
template bool OpenACCMappableModel<fir::ReferenceType>::generatePrivateDestroy(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value privatized, mlir::ValueRange bounds) const;
template bool OpenACCMappableModel<fir::HeapType>::generatePrivateDestroy(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value privatized, mlir::ValueRange bounds) const;
template bool OpenACCMappableModel<fir::PointerType>::generatePrivateDestroy(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value privatized, mlir::ValueRange bounds) const;

template <typename Ty>
mlir::Value OpenACCPointerLikeModel<Ty>::genAllocate(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef varName, mlir::Type varType, mlir::Value originalVar,
    bool &needsFree) const {

  // Unwrap to get the pointee type.
  mlir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
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
  mlir::Value allocation;
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

template mlir::Value OpenACCPointerLikeModel<fir::ReferenceType>::genAllocate(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef varName, mlir::Type varType, mlir::Value originalVar,
    bool &needsFree) const;

template mlir::Value OpenACCPointerLikeModel<fir::PointerType>::genAllocate(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef varName, mlir::Type varType, mlir::Value originalVar,
    bool &needsFree) const;

template mlir::Value OpenACCPointerLikeModel<fir::HeapType>::genAllocate(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef varName, mlir::Type varType, mlir::Value originalVar,
    bool &needsFree) const;

template mlir::Value OpenACCPointerLikeModel<fir::LLVMPointerType>::genAllocate(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef varName, mlir::Type varType, mlir::Value originalVar,
    bool &needsFree) const;

static mlir::Value stripCasts(mlir::Value value, bool stripDeclare = true) {
  mlir::Value currentValue = value;

  while (currentValue) {
    auto *definingOp = currentValue.getDefiningOp();
    if (!definingOp)
      break;

    if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(definingOp)) {
      currentValue = convertOp.getValue();
      continue;
    }

    if (auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definingOp)) {
      currentValue = viewLike.getViewSource();
      continue;
    }

    if (stripDeclare) {
      if (auto declareOp = mlir::dyn_cast<hlfir::DeclareOp>(definingOp)) {
        currentValue = declareOp.getMemref();
        continue;
      }

      if (auto declareOp = mlir::dyn_cast<fir::DeclareOp>(definingOp)) {
        currentValue = declareOp.getMemref();
        continue;
      }
    }
    break;
  }

  return currentValue;
}

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::genFree(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
    mlir::Value allocRes, mlir::Type varType) const {

  // Unwrap to get the pointee type.
  mlir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
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
  mlir::Value valueToInspect = allocRes ? allocRes : varToFree;

  // Strip casts and declare operations to find the original allocation
  mlir::Value strippedValue = stripCasts(valueToInspect);
  mlir::Operation *originalAlloc = strippedValue.getDefiningOp();

  // If we found an AllocMemOp (heap allocation), free it
  if (mlir::isa_and_nonnull<fir::AllocMemOp>(originalAlloc)) {
    mlir::Value toFree = varToFree;
    if (!mlir::isa<fir::HeapType>(valueToInspect.getType()))
      toFree = fir::ConvertOp::create(
          builder, loc,
          fir::HeapType::get(varToFree.getType().getElementType()), toFree);
    fir::FreeMemOp::create(builder, loc, toFree);
    return true;
  }

  // If we found an AllocaOp (stack allocation), no deallocation needed
  if (mlir::isa_and_nonnull<fir::AllocaOp>(originalAlloc))
    return true;

  // Unable to determine allocation type
  return false;
}

template bool OpenACCPointerLikeModel<fir::ReferenceType>::genFree(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
    mlir::Value allocRes, mlir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::PointerType>::genFree(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
    mlir::Value allocRes, mlir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::HeapType>::genFree(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
    mlir::Value allocRes, mlir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::genFree(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
    mlir::Value allocRes, mlir::Type varType) const;

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::genCopy(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> destination,
    mlir::TypedValue<mlir::acc::PointerLikeType> source,
    mlir::Type varType) const {

  // Check that source and destination types match
  if (source.getType() != destination.getType())
    return false;

  // Unwrap to get the pointee type.
  mlir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
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
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> destination,
    mlir::TypedValue<mlir::acc::PointerLikeType> source,
    mlir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::PointerType>::genCopy(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> destination,
    mlir::TypedValue<mlir::acc::PointerLikeType> source,
    mlir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::HeapType>::genCopy(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> destination,
    mlir::TypedValue<mlir::acc::PointerLikeType> source,
    mlir::Type varType) const;

template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::genCopy(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> destination,
    mlir::TypedValue<mlir::acc::PointerLikeType> source,
    mlir::Type varType) const;

template <typename Ty>
mlir::Value OpenACCPointerLikeModel<Ty>::genLoad(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
    mlir::Type valueType) const {

  // Unwrap to get the pointee type.
  mlir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
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

  mlir::Value loadedValue = fir::LoadOp::create(builder, loc, srcPtr);

  // If valueType is provided and differs from the loaded type, insert a convert
  if (valueType && loadedValue.getType() != valueType)
    return fir::ConvertOp::create(builder, loc, valueType, loadedValue);

  return loadedValue;
}

template mlir::Value OpenACCPointerLikeModel<fir::ReferenceType>::genLoad(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
    mlir::Type valueType) const;

template mlir::Value OpenACCPointerLikeModel<fir::PointerType>::genLoad(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
    mlir::Type valueType) const;

template mlir::Value OpenACCPointerLikeModel<fir::HeapType>::genLoad(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
    mlir::Type valueType) const;

template mlir::Value OpenACCPointerLikeModel<fir::LLVMPointerType>::genLoad(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
    mlir::Type valueType) const;

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::genStore(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value valueToStore,
    mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const {

  // Unwrap to get the pointee type.
  mlir::Type pointeeTy = fir::dyn_cast_ptrEleTy(pointer);
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
  mlir::Type valueType = valueToStore.getType();
  mlir::Value convertedValue = valueToStore;

  // If the value type differs from the pointee type, insert a convert
  if (valueType != pointeeTy)
    convertedValue =
        fir::ConvertOp::create(builder, loc, pointeeTy, valueToStore);

  fir::StoreOp::create(builder, loc, convertedValue, destPtr);
  return true;
}

template bool OpenACCPointerLikeModel<fir::ReferenceType>::genStore(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value valueToStore,
    mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const;

template bool OpenACCPointerLikeModel<fir::PointerType>::genStore(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value valueToStore,
    mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const;

template bool OpenACCPointerLikeModel<fir::HeapType>::genStore(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value valueToStore,
    mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const;

template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::genStore(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value valueToStore,
    mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const;

/// Check CUDA attributes on a function argument.
static bool hasCUDADeviceAttrOnFuncArg(mlir::BlockArgument blockArg) {
  auto *owner = blockArg.getOwner();
  if (!owner)
    return false;

  auto *parentOp = owner->getParentOp();
  if (!parentOp)
    return false;

  if (auto funcLike = mlir::dyn_cast<mlir::FunctionOpInterface>(parentOp)) {
    unsigned argIndex = blockArg.getArgNumber();
    if (argIndex < funcLike.getNumArguments())
      if (auto attr = funcLike.getArgAttr(argIndex, cuf::getDataAttrName()))
        if (auto cudaAttr = mlir::dyn_cast<cuf::DataAttributeAttr>(attr))
          return cuf::isDeviceDataAttribute(cudaAttr.getValue());
  }
  return false;
}

/// Shared implementation for checking if a value represents device data.
static bool isDeviceDataImpl(mlir::Value var) {
  // Strip casts to find the underlying value.
  mlir::Value currentVal = stripCasts(var, /*stripDeclare=*/false);

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(currentVal))
    return hasCUDADeviceAttrOnFuncArg(blockArg);

  mlir::Operation *defOp = currentVal.getDefiningOp();
  assert(defOp && "expected defining op for non-block-argument value");

  // Check for CUDA attributes on the defining operation.
  if (cuf::hasDeviceDataAttr(defOp))
    return true;

  // Handle operations that access a partial entity - check if the base entity
  // is device data.
  if (auto partialAccess =
          mlir::dyn_cast<mlir::acc::PartialEntityAccessOpInterface>(defOp))
    if (mlir::Value base = partialAccess.getBaseEntity())
      return isDeviceDataImpl(base);

  // Handle fir.embox, fir.rebox, and similar ops via
  // FortranObjectViewOpInterface to check if the underlying source is device
  // data.
  if (auto viewOp = mlir::dyn_cast<fir::FortranObjectViewOpInterface>(defOp))
    if (mlir::Value source = viewOp.getViewSource(defOp->getResult(0)))
      return isDeviceDataImpl(source);

  // Handle address_of - check the referenced global.
  if (auto addrOfIface =
          mlir::dyn_cast<mlir::acc::AddressOfGlobalOpInterface>(defOp)) {
    auto symbol = addrOfIface.getSymbol();
    if (auto global = mlir::SymbolTable::lookupNearestSymbolFrom<
            mlir::acc::GlobalVariableOpInterface>(defOp, symbol))
      return global.isDeviceData();
    return false;
  }

  return false;
}

template <typename Ty>
bool OpenACCPointerLikeModel<Ty>::isDeviceData(mlir::Type pointer,
                                               mlir::Value var) const {
  return isDeviceDataImpl(var);
}

template bool OpenACCPointerLikeModel<fir::ReferenceType>::isDeviceData(
    mlir::Type, mlir::Value) const;
template bool
    OpenACCPointerLikeModel<fir::PointerType>::isDeviceData(mlir::Type,
                                                            mlir::Value) const;
template bool
    OpenACCPointerLikeModel<fir::HeapType>::isDeviceData(mlir::Type,
                                                         mlir::Value) const;
template bool OpenACCPointerLikeModel<fir::LLVMPointerType>::isDeviceData(
    mlir::Type, mlir::Value) const;

template <typename Ty>
bool OpenACCMappableModel<Ty>::isDeviceData(mlir::Type type,
                                            mlir::Value var) const {
  return isDeviceDataImpl(var);
}

template bool
    OpenACCMappableModel<fir::BaseBoxType>::isDeviceData(mlir::Type,
                                                         mlir::Value) const;
template bool
    OpenACCMappableModel<fir::ReferenceType>::isDeviceData(mlir::Type,
                                                           mlir::Value) const;
template bool
    OpenACCMappableModel<fir::HeapType>::isDeviceData(mlir::Type,
                                                      mlir::Value) const;
template bool
    OpenACCMappableModel<fir::PointerType>::isDeviceData(mlir::Type,
                                                         mlir::Value) const;

} // namespace fir::acc
