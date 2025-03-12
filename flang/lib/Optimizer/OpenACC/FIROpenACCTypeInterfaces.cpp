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
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

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

} // namespace fir::acc
