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
            upperbound = mlir::arith::AddIOp::create(builder, loc, lowerbound,
                                                     upperbound);
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

static fir::ShapeOp genShapeOp(mlir::OpBuilder &builder,
                               fir::SequenceType seqTy, mlir::Location loc) {
  llvm::SmallVector<mlir::Value> extents;
  mlir::Type idxTy = builder.getIndexType();
  for (auto extent : seqTy.getShape())
    extents.push_back(mlir::arith::ConstantOp::create(
        builder, loc, idxTy, builder.getIntegerAttr(idxTy, extent)));
  return fir::ShapeOp::create(builder, loc, extents);
}

template <typename Ty>
mlir::Value OpenACCMappableModel<Ty>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal) const {
  mlir::Value retVal;
  mlir::Type unwrappedTy = fir::unwrapRefType(type);
  mlir::ModuleOp mod = builder.getInsertionBlock()
                           ->getParent()
                           ->getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder firBuilder(builder, mod);

  auto getDeclareOpForType = [&](mlir::Type ty) -> hlfir::DeclareOp {
    auto alloca = fir::AllocaOp::create(firBuilder, loc, ty);
    return hlfir::DeclareOp::create(
        firBuilder, loc, alloca, varName, /*shape=*/nullptr,
        llvm::ArrayRef<mlir::Value>{},
        /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});
  };

  if (fir::isa_trivial(unwrappedTy)) {
    auto declareOp = getDeclareOpForType(unwrappedTy);
    if (initVal) {
      auto convert = firBuilder.createConvert(loc, unwrappedTy, initVal);
      fir::StoreOp::create(firBuilder, loc, convert, declareOp.getBase());
    }
    retVal = declareOp.getBase();
  } else if (auto seqTy =
                 mlir::dyn_cast_or_null<fir::SequenceType>(unwrappedTy)) {
    if (fir::isa_trivial(seqTy.getEleTy())) {
      mlir::Value shape;
      if (seqTy.hasDynamicExtents()) {
        shape = fir::ShapeOp::create(firBuilder, loc, llvm::to_vector(extents));
      } else {
        shape = genShapeOp(firBuilder, seqTy, loc);
      }
      auto alloca = fir::AllocaOp::create(
          firBuilder, loc, seqTy, /*typeparams=*/mlir::ValueRange{}, extents);
      auto declareOp = hlfir::DeclareOp::create(
          firBuilder, loc, alloca, varName, shape,
          llvm::ArrayRef<mlir::Value>{},
          /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});

      if (initVal) {
        mlir::Type idxTy = firBuilder.getIndexType();
        mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
        llvm::SmallVector<fir::DoLoopOp> loops;
        llvm::SmallVector<mlir::Value> ivs;

        if (seqTy.hasDynamicExtents()) {
          hlfir::AssignOp::create(firBuilder, loc, initVal,
                                  declareOp.getBase());
        } else {
          for (auto ext : seqTy.getShape()) {
            auto lb = firBuilder.createIntegerConstant(loc, idxTy, 0);
            auto ub = firBuilder.createIntegerConstant(loc, idxTy, ext - 1);
            auto step = firBuilder.createIntegerConstant(loc, idxTy, 1);
            auto loop = fir::DoLoopOp::create(firBuilder, loc, lb, ub, step,
                                              /*unordered=*/false);
            firBuilder.setInsertionPointToStart(loop.getBody());
            loops.push_back(loop);
            ivs.push_back(loop.getInductionVar());
          }
          auto coord = fir::CoordinateOp::create(firBuilder, loc, refTy,
                                                 declareOp.getBase(), ivs);
          fir::StoreOp::create(firBuilder, loc, initVal, coord);
          firBuilder.setInsertionPointAfter(loops[0]);
        }
      }
      retVal = declareOp.getBase();
    }
  } else if (auto boxTy =
                 mlir::dyn_cast_or_null<fir::BaseBoxType>(unwrappedTy)) {
    mlir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
    if (fir::isa_trivial(innerTy)) {
      retVal = getDeclareOpForType(unwrappedTy).getBase();
      mlir::Value allocatedScalar =
          fir::AllocMemOp::create(builder, loc, innerTy);
      mlir::Value firClass =
          fir::EmboxOp::create(builder, loc, boxTy, allocatedScalar);
      fir::StoreOp::create(builder, loc, firClass, retVal);
    } else if (mlir::isa<fir::SequenceType>(innerTy)) {
      hlfir::Entity source = hlfir::Entity{var};
      auto [temp, cleanup] = hlfir::createTempFromMold(loc, firBuilder, source);
      if (fir::isa_ref_type(type)) {
        // When the temp is created - it is not a reference - thus we can
        // end up with a type inconsistency. Therefore ensure storage is created
        // for it.
        retVal = getDeclareOpForType(unwrappedTy).getBase();
        mlir::Value storeDst = retVal;
        if (fir::unwrapRefType(retVal.getType()) != temp.getType()) {
          // `createTempFromMold` makes the unfortunate choice to lose the
          // `fir.heap` and `fir.ptr` types when wrapping with a box. Namely,
          // when wrapping a `fir.heap<fir.array>`, it will create instead a
          // `fir.box<fir.array>`. Cast here to deal with this inconsistency.
          storeDst = firBuilder.createConvert(
              loc, firBuilder.getRefType(temp.getType()), retVal);
        }
        fir::StoreOp::create(builder, loc, temp, storeDst);
      } else {
        retVal = temp;
      }
    } else {
      TODO(loc, "Unsupported boxed type for OpenACC private-like recipe");
    }
    if (initVal) {
      hlfir::AssignOp::create(builder, loc, initVal, retVal);
    }
  }
  return retVal;
}

template mlir::Value
OpenACCMappableModel<fir::BaseBoxType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal) const;

template mlir::Value
OpenACCMappableModel<fir::ReferenceType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal) const;

template mlir::Value OpenACCMappableModel<fir::HeapType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal) const;

template mlir::Value
OpenACCMappableModel<fir::PointerType>::generatePrivateInit(
    mlir::Type type, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::MappableType> var, llvm::StringRef varName,
    mlir::ValueRange extents, mlir::Value initVal) const;

} // namespace fir::acc
