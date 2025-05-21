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

#include "flang/Optimizer/OpenACC/FIROpenACCTypeInterfaces.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir::acc {

static mlir::TypedValue<mlir::acc::PointerLikeType>
getPtrFromVar(mlir::Value var) {
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

template <>
mlir::TypedValue<mlir::acc::PointerLikeType>
OpenACCMappableModel<fir::SequenceType>::getVarPtr(mlir::Type type,
                                                   mlir::Value var) const {
  return getPtrFromVar(var);
}

template <>
mlir::TypedValue<mlir::acc::PointerLikeType>
OpenACCMappableModel<fir::BaseBoxType>::getVarPtr(mlir::Type type,
                                                  mlir::Value var) const {
  return getPtrFromVar(var);
}

template <>
std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::SequenceType>::getSizeInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const {
  // TODO: Bounds operation affect the total size - add support to take them
  // into account.
  if (!accBounds.empty())
    return {};

  // Dynamic extents or unknown ranks generally do not have compile-time
  // computable dimensions.
  auto seqType = mlir::cast<fir::SequenceType>(type);
  if (seqType.hasDynamicExtents() || seqType.hasUnknownShape())
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
      fir::getTypeSizeAndAlignment(var.getLoc(), type, dataLayout, kindMap);
  if (!sizeAndAlignment.has_value())
    return {};

  return {llvm::TypeSize::getFixed(sizeAndAlignment->first)};
}

template <>
std::optional<llvm::TypeSize>
OpenACCMappableModel<fir::BaseBoxType>::getSizeInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const {
  // If we have a box value instead of box reference, the intent is to
  // get the size of the data not the box itself.
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(var.getType())) {
    if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(
            fir::unwrapRefType(boxTy.getEleTy()))) {
      return mappableTy.getSizeInBytes(var, accBounds, dataLayout);
    }
  }
  // Size for boxes is not computable until it gets materialized.
  return {};
}

template <>
std::optional<int64_t>
OpenACCMappableModel<fir::SequenceType>::getOffsetInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const {
  // TODO: Bounds operation affect the offset- add support to take them
  // into account.
  if (!accBounds.empty())
    return {};

  // Dynamic extents (aka descriptor-based arrays) - may have a offset.
  // For example, a negative stride may mean a negative offset to compute the
  // start of array.
  auto seqType = mlir::cast<fir::SequenceType>(type);
  if (seqType.hasDynamicExtents() || seqType.hasUnknownShape())
    return {};

  // We have non-dynamic extents - but if for some reason the size is not
  // computable - assume offset is not either. Otherwise, it is an offset of
  // zero.
  if (getSizeInBytes(type, var, accBounds, dataLayout).has_value()) {
    return {0};
  }
  return {};
}

template <>
std::optional<int64_t> OpenACCMappableModel<fir::BaseBoxType>::getOffsetInBytes(
    mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
    const mlir::DataLayout &dataLayout) const {
  // If we have a box value instead of box reference, the intent is to
  // get the offset of the data not the offset of the box itself.
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(var.getType())) {
    if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(
            fir::unwrapRefType(boxTy.getEleTy()))) {
      return mappableTy.getOffsetInBytes(var, accBounds, dataLayout);
    }
  }
  // Until boxes get materialized, the offset is not evident because it is
  // relative to the pointer being held.
  return {};
}

template <>
llvm::SmallVector<mlir::Value>
OpenACCMappableModel<fir::SequenceType>::generateAccBounds(
    mlir::Type type, mlir::Value var, mlir::OpBuilder &builder) const {
  assert((mlir::isa<mlir::acc::PointerLikeType>(var.getType()) ||
          mlir::isa<mlir::acc::MappableType>(var.getType())) &&
         "must be pointer-like or mappable");

  fir::FirOpBuilder firBuilder(builder, var.getDefiningOp());
  auto seqType = mlir::cast<fir::SequenceType>(type);
  mlir::Location loc = var.getLoc();

  mlir::Value varPtr =
      mlir::isa<mlir::acc::PointerLikeType>(var.getType())
          ? var
          : mlir::cast<mlir::acc::MappableType>(var.getType()).getVarPtr(var);

  if (seqType.hasDynamicExtents() || seqType.hasUnknownShape()) {
    if (auto boxAddr =
            mlir::dyn_cast_if_present<fir::BoxAddrOp>(varPtr.getDefiningOp())) {
      mlir::Value box = boxAddr.getVal();
      auto res =
          hlfir::translateToExtendedValue(loc, firBuilder, hlfir::Entity(box));
      fir::ExtendedValue exv = res.first;
      mlir::Value boxRef = box;
      if (auto boxPtr = getPtrFromVar(box)) {
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

    if (mlir::isa<hlfir::DeclareOp, fir::DeclareOp>(varPtr.getDefiningOp())) {
      mlir::Value zero =
          firBuilder.createIntegerConstant(loc, builder.getIndexType(), 0);
      mlir::Value one =
          firBuilder.createIntegerConstant(loc, builder.getIndexType(), 1);

      mlir::Value shape;
      if (auto declareOp =
              mlir::dyn_cast_if_present<fir::DeclareOp>(varPtr.getDefiningOp()))
        shape = declareOp.getShape();
      else if (auto declareOp = mlir::dyn_cast_if_present<hlfir::DeclareOp>(
                   varPtr.getDefiningOp()))
        shape = declareOp.getShape();

      const bool strideIncludeLowerExtent = true;

      llvm::SmallVector<mlir::Value> accBounds;
      if (auto shapeOp =
              mlir::dyn_cast_if_present<fir::ShapeOp>(shape.getDefiningOp())) {
        mlir::Value cummulativeExtent = one;
        for (auto extent : shapeOp.getExtents()) {
          mlir::Value upperbound =
              builder.create<mlir::arith::SubIOp>(loc, extent, one);
          mlir::Value stride = one;
          if (strideIncludeLowerExtent) {
            stride = cummulativeExtent;
            cummulativeExtent = builder.create<mlir::arith::MulIOp>(
                loc, cummulativeExtent, extent);
          }
          auto accBound = builder.create<mlir::acc::DataBoundsOp>(
              loc, mlir::acc::DataBoundsType::get(builder.getContext()),
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
                builder.create<mlir::arith::SubIOp>(loc, extent, one);
            upperbound = builder.create<mlir::arith::AddIOp>(loc, lowerbound,
                                                             upperbound);
            mlir::Value stride = one;
            if (strideIncludeLowerExtent) {
              stride = cummulativeExtent;
              cummulativeExtent = builder.create<mlir::arith::MulIOp>(
                  loc, cummulativeExtent, extent);
            }
            auto accBound = builder.create<mlir::acc::DataBoundsOp>(
                loc, mlir::acc::DataBoundsType::get(builder.getContext()),
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
  auto valToCheck = varPtr;
  if (auto boxAddr =
          mlir::dyn_cast_if_present<fir::BoxAddrOp>(varPtr.getDefiningOp())) {
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

template <>
llvm::SmallVector<mlir::Value>
OpenACCMappableModel<fir::BaseBoxType>::generateAccBounds(
    mlir::Type type, mlir::Value var, mlir::OpBuilder &builder) const {
  // If we have a box value instead of box reference, the intent is to
  // get the bounds of the data not the bounds of the box itself.
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(var.getType())) {
    if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(
            fir::unwrapRefType(boxTy.getEleTy()))) {
      mlir::Value data = builder.create<fir::BoxAddrOp>(var.getLoc(), var);
      return mappableTy.generateAccBounds(data, builder);
    }
  }
  // Box references are not arrays - thus generating acc.bounds does not make
  // sense.
  return {};
}

static bool isScalarLike(mlir::Type type) {
  return fir::isa_trivial(type) || fir::isa_ref_type(type);
}

static bool isArrayLike(mlir::Type type) {
  return mlir::isa<fir::SequenceType>(type);
}

static bool isCompositeLike(mlir::Type type) {
  return mlir::isa<fir::RecordType, fir::ClassType, mlir::TupleType>(type);
}

template <>
mlir::acc::VariableTypeCategory
OpenACCMappableModel<fir::SequenceType>::getTypeCategory(
    mlir::Type type, mlir::Value var) const {
  return mlir::acc::VariableTypeCategory::array;
}

template <>
mlir::acc::VariableTypeCategory
OpenACCMappableModel<fir::BaseBoxType>::getTypeCategory(mlir::Type type,
                                                        mlir::Value var) const {

  mlir::Type eleTy = fir::dyn_cast_ptrOrBoxEleTy(type);

  // If the type enclosed by the box is a mappable type, then have it
  // provide the type category.
  if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(eleTy))
    return mappableTy.getTypeCategory(var);

  // For all arrays, despite whether they are allocatable, pointer, assumed,
  // etc, we'd like to categorize them as "array".
  if (isArrayLike(eleTy))
    return mlir::acc::VariableTypeCategory::array;

  // We got here because we don't have an array nor a mappable type. At this
  // point, we know we have a type that fits the "aggregate" definition since it
  // is a type with a descriptor. Try to refine it by checking if it matches the
  // "composite" definition.
  if (isCompositeLike(eleTy))
    return mlir::acc::VariableTypeCategory::composite;

  // Even if we have a scalar type - simply because it is wrapped in a box
  // we want to categorize it as "nonscalar". Anything else would've been
  // non-scalar anyway.
  return mlir::acc::VariableTypeCategory::nonscalar;
}

static mlir::TypedValue<mlir::acc::PointerLikeType>
getBaseRef(mlir::TypedValue<mlir::acc::PointerLikeType> varPtr) {
  // If there is no defining op - the unwrapped reference is the base one.
  mlir::Operation *op = varPtr.getDefiningOp();
  if (!op)
    return varPtr;

  // Look to find if this value originates from an interior pointer
  // calculation op.
  mlir::Value baseRef =
      llvm::TypeSwitch<mlir::Operation *, mlir::Value>(op)
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
          .Default([&](mlir::Operation *) { return varPtr; });

  return mlir::cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(baseRef);
}

static mlir::acc::VariableTypeCategory
categorizePointee(mlir::Type pointer,
                  mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
                  mlir::Type varType) {
  // FIR uses operations to compute interior pointers.
  // So for example, an array element or composite field access to a float
  // value would both be represented as !fir.ref<f32>. We do not want to treat
  // such a reference as a scalar. Thus unwrap interior pointer calculations.
  auto baseRef = getBaseRef(varPtr);
  mlir::Type eleTy = baseRef.getType().getElementType();

  if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(eleTy))
    return mappableTy.getTypeCategory(varPtr);

  if (isScalarLike(eleTy))
    return mlir::acc::VariableTypeCategory::scalar;
  if (isArrayLike(eleTy))
    return mlir::acc::VariableTypeCategory::array;
  if (isCompositeLike(eleTy))
    return mlir::acc::VariableTypeCategory::composite;
  if (mlir::isa<fir::CharacterType, mlir::FunctionType>(eleTy))
    return mlir::acc::VariableTypeCategory::nonscalar;
  // "pointers" - in the sense of raw address point-of-view, are considered
  // scalars. However
  if (mlir::isa<fir::LLVMPointerType>(eleTy))
    return mlir::acc::VariableTypeCategory::scalar;

  // Without further checking, this type cannot be categorized.
  return mlir::acc::VariableTypeCategory::uncategorized;
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

} // namespace fir::acc
