//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of external OpenACC type interfaces for CIR.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/OpenACC/CIROpenACCTypeInterfaces.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/OpenACC/CIROpenACCUtils.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir::acc {

// The alloca reached from a pointer. `baseEntity` is true when the pointer
// still refers to that whole allocation.
struct AllocaInfo {
  cir::AllocaOp alloca;
  bool baseEntity = false;
  explicit operator bool() const { return static_cast<bool>(alloca); }
};

static bool isZero(mlir::Value value) {
  auto constant = value.getDefiningOp<cir::ConstantOp>();
  if (!constant)
    return false;
  auto attr = mlir::dyn_cast<cir::IntAttr>(constant.getValue());
  return attr && attr.getValue() == 0;
}

// The pointee type does not describe the allocated entity. Walk the
// operations that constructed `value` and recover that entity from its alloca.
static AllocaInfo findAlloca(mlir::Value value) {
  bool baseEntity = true;
  while (mlir::Operation *def = value.getDefiningOp()) {
    if (auto alloca = mlir::dyn_cast<cir::AllocaOp>(def))
      return {alloca, baseEntity};
    if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
      baseEntity &= (cast.getKind() == cir::CastKind::bitcast ||
                     cast.getKind() == cir::CastKind::address_space ||
                     cast.getKind() == cir::CastKind::array_to_ptrdecay);
      value = cast.getSrc();
      continue;
    }
    if (auto stride = mlir::dyn_cast<cir::PtrStrideOp>(def)) {
      baseEntity &= isZero(stride.getStride());
      value = stride.getBase();
      continue;
    }
    if (auto member = mlir::dyn_cast<cir::GetMemberOp>(def)) {
      baseEntity = false;
      value = member.getAddr();
      continue;
    }
    break;
  }
  return {};
}

static mlir::Type getBaseType(mlir::Value varPtr) {
  if (AllocaInfo found = findAlloca(varPtr))
    return found.alloca.getAllocaType();
  // A block argument or other value with no alloca has only its pointer type.
  return mlir::cast<cir::PointerType>(varPtr.getType()).getPointee();
}

template <>
mlir::acc::VariableTypeCategory
OpenACCPointerLikeModel<cir::PointerType>::getPointeeTypeCategory(
    mlir::Type pointer, mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
    mlir::Type varType) const {
  // A dynamic alloca is an array of its element. Classifying it from the
  // pointer type alone would report a scalar.
  if (AllocaInfo base = findAlloca(varPtr))
    if (base.alloca.isDynamic())
      return mlir::acc::VariableTypeCategory::array;

  mlir::Type eleTy = getBaseType(varPtr);

  if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(eleTy))
    return mappableTy.getTypeCategory(varPtr);

  if (isAnyIntegerOrFloatingPointType(eleTy) ||
      mlir::isa<cir::BoolType, cir::PointerType, cir::ComplexType>(eleTy))
    return mlir::acc::VariableTypeCategory::scalar;
  if (mlir::isa<cir::ArrayType>(eleTy))
    return mlir::acc::VariableTypeCategory::array;
  if (mlir::isa<cir::RecordType>(eleTy))
    return mlir::acc::VariableTypeCategory::composite;
  if (mlir::isa<cir::FuncType>(eleTy) || mlir::isa<cir::VectorType>(eleTy))
    return mlir::acc::VariableTypeCategory::nonscalar;

  // Without further checking, this type cannot be categorized.
  return mlir::acc::VariableTypeCategory::uncategorized;
}

// Incomplete records and types without a data layout have no size.
static bool hasKnownLayout(mlir::Type type) {
  if (auto record = mlir::dyn_cast<cir::RecordType>(type))
    return record.isComplete();
  if (auto array = mlir::dyn_cast<cir::ArrayType>(type))
    return hasKnownLayout(array.getElementType());
  return type.hasTrait<mlir::DataLayoutTypeInterface::Trait>();
}

// A scalar whose size comes from the data layout.
static bool isKnownSizeScalar(mlir::Type type) {
  return hasKnownLayout(type) &&
         (isAnyIntegerOrFloatingPointType(type) ||
          mlir::isa<cir::BoolType, cir::PointerType, cir::ComplexType>(type));
}

static bool isKnownSizeArray(mlir::Type type) {
  return mlir::isa<cir::ArrayType>(type) && hasKnownLayout(type);
}

// Alignment comes from the module that contains the insertion point.
static std::optional<mlir::IntegerAttr>
allocaAlignment(mlir::OpBuilder &builder, mlir::Type pointee) {
  mlir::Operation *scope = nullptr;
  if (mlir::Block *block = builder.getInsertionBlock())
    scope = block->getParentOp();
  std::optional<mlir::DataLayout> layout = mlir::acc::getDataLayout(scope);
  if (!layout)
    return std::nullopt;
  uint64_t align = layout->getTypeABIAlignment(pointee);
  if (align == 0)
    return std::nullopt;
  return builder.getI64IntegerAttr(align);
}

// A dynamic extent is a non-zero constant.
static std::optional<uint64_t> constantCount(mlir::Value size) {
  auto constant = size.getDefiningOp<cir::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto attr = mlir::dyn_cast<cir::IntAttr>(constant.getValue());
  if (!attr || attr.getUInt() == 0)
    return std::nullopt;
  return attr.getUInt();
}

static mlir::Value constantSizeHere(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value size) {
  if (!constantCount(size))
    return {};
  return cir::ConstantOp::create(
      builder, loc, size.getDefiningOp<cir::ConstantOp>().getValue());
}

static cir::PointerType pointerTo(mlir::Type pointee, cir::PointerType like) {
  if (like.getAddrSpace())
    return cir::PointerType::get(pointee, like.getAddrSpace());
  return cir::PointerType::get(pointee);
}

// `count` is set for a dynamic alloca. A constant count is folded into
// `!cir.array` when no size operand is requested.
struct Object {
  mlir::Type type;
  mlir::Value count;
};

struct Resolved {
  bool sawAlloca = false;
  std::optional<Object> object;
};

// The scalar or constant-size array described by `pointer`. `needCount` keeps
// a dynamic size as a value. Otherwise a constant count is part of the type.
static Resolved resolveObject(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value pointer, bool needCount) {
  AllocaInfo found = pointer ? findAlloca(pointer) : AllocaInfo{};
  if (!found)
    return {};

  if (!found.baseEntity)
    return {true, std::nullopt};

  mlir::Type type = found.alloca.getAllocaType();
  if (!isKnownSizeScalar(type) && !isKnownSizeArray(type))
    return {true, std::nullopt};
  if (!found.alloca.isDynamic())
    return {true, Object{type, {}}};

  std::optional<uint64_t> count = constantCount(found.alloca.getDynAllocSize());
  if (!count)
    return {true, std::nullopt};
  if (!needCount)
    return {true, Object{cir::ArrayType::get(type, *count), {}}};

  mlir::Value size =
      constantSizeHere(builder, loc, found.alloca.getDynAllocSize());
  if (!size)
    return {true, std::nullopt};
  return {true, Object{type, size}};
}

// The scalar or constant-size array described by `pointer`. Without an
// alloca, `fallback` is that object when its size is known.
static std::optional<Object>
knownSizeObject(mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value pointer, mlir::Type fallback, bool needCount) {
  Resolved resolved = resolveObject(builder, loc, pointer, needCount);
  if (resolved.sawAlloca && !resolved.object)
    return std::nullopt;
  if (resolved.object)
    return resolved.object;
  if (fallback && (isKnownSizeScalar(fallback) || isKnownSizeArray(fallback)))
    return Object{fallback, {}};
  return std::nullopt;
}

// Pointer-to-pointer conversions, and integer-to-pointer conversions. A
// pointer bitcast is used when the pointee changes, then an address-space
// cast when the address space changes. Value-to-value conversions are left
// to the caller.
static mlir::Value castPointer(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value value, mlir::Type resultType) {
  mlir::Type srcType = value.getType();
  if (srcType == resultType)
    return value;

  auto srcPtr = mlir::dyn_cast<cir::PointerType>(srcType);
  auto dstPtr = mlir::dyn_cast<cir::PointerType>(resultType);
  if (!dstPtr)
    return {};
  if (!srcPtr) {
    if (!mlir::isa<cir::IntType>(srcType))
      return {};
    return cir::CastOp::create(builder, loc, resultType,
                               cir::CastKind::int_to_ptr, value);
  }

  mlir::Value current = value;
  if (srcPtr.getPointee() != dstPtr.getPointee()) {
    auto midTy =
        cir::PointerType::get(dstPtr.getPointee(), srcPtr.getAddrSpace());
    if (current.getType() != midTy)
      current = cir::CastOp::create(builder, loc, midTy, cir::CastKind::bitcast,
                                    current);
  }
  if (current.getType() != resultType)
    current = cir::CastOp::create(builder, loc, resultType,
                                  cir::CastKind::address_space, current);
  return current;
}

// A memref element is a builtin scalar, complex, or fixed vector. CIR signs
// are dropped: memref storage is signless.
static mlir::Type convertMemRefElementType(mlir::Type type,
                                           const mlir::DataLayout &layout) {
  if (auto intTy = mlir::dyn_cast<cir::IntType>(type)) {
    if (intTy.isBitInt() || intTy.getWidth() == 0)
      return {};
    return mlir::IntegerType::get(type.getContext(), intTy.getWidth());
  }
  if (mlir::isa<cir::BoolType>(type)) {
    llvm::TypeSize bits = layout.getTypeSizeInBits(type);
    if (!bits.isFixed() || !bits.getFixedValue())
      return {};
    return mlir::IntegerType::get(type.getContext(),
                                  static_cast<unsigned>(bits.getFixedValue()));
  }
  if (mlir::isa<cir::SingleType>(type))
    return mlir::Float32Type::get(type.getContext());
  if (mlir::isa<cir::DoubleType>(type))
    return mlir::Float64Type::get(type.getContext());
  if (mlir::isa<cir::FP16Type>(type))
    return mlir::Float16Type::get(type.getContext());
  if (mlir::isa<cir::BF16Type>(type))
    return mlir::BFloat16Type::get(type.getContext());
  if (mlir::isa<cir::FP80Type>(type))
    return mlir::Float80Type::get(type.getContext());
  if (mlir::isa<cir::FP128Type>(type))
    return mlir::Float128Type::get(type.getContext());
  if (auto longDouble = mlir::dyn_cast<cir::LongDoubleType>(type))
    return convertMemRefElementType(longDouble.getUnderlying(), layout);
  if (auto complex = mlir::dyn_cast<cir::ComplexType>(type)) {
    mlir::Type element =
        convertMemRefElementType(complex.getElementType(), layout);
    if (!element)
      return {};
    return mlir::ComplexType::get(element);
  }
  if (auto vector = mlir::dyn_cast<cir::VectorType>(type)) {
    if (vector.isScalable())
      return {};
    mlir::Type element =
        convertMemRefElementType(vector.getElementType(), layout);
    if (!element || !element.isIntOrIndexOrFloat())
      return {};
    return mlir::VectorType::get({static_cast<int64_t>(vector.getFixedSize())},
                                 element);
  }
  return {};
}

// `shape` is outermost first, matching `!cir.array` nesting.
static bool collectMemRefShape(mlir::Type type,
                               llvm::SmallVectorImpl<int64_t> &shape,
                               mlir::Type &element,
                               const mlir::DataLayout &layout) {
  if (auto array = mlir::dyn_cast<cir::ArrayType>(type)) {
    shape.push_back(static_cast<int64_t>(array.getSize()));
    return collectMemRefShape(array.getElementType(), shape, element, layout);
  }
  if (mlir::isa<cir::PointerType>(type))
    return false;
  element = convertMemRefElementType(type, layout);
  return static_cast<bool>(element) &&
         mlir::BaseMemRefType::isValidElementType(element);
}

static mlir::MemRefType
convertPointerToMemRefType(mlir::Type pointer, const mlir::DataLayout &layout) {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(pointer);
  if (!ptrTy)
    return {};

  mlir::Type pointee = ptrTy.getPointee();
  if (!hasKnownLayout(pointee))
    return {};

  llvm::SmallVector<int64_t> shape;
  mlir::Type element;
  if (!collectMemRefShape(pointee, shape, element, layout))
    return {};
  return mlir::MemRefType::get(shape, element);
}

template <>
mlir::Value OpenACCPointerLikeModel<cir::PointerType>::genAllocate(
    mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
    llvm::StringRef varName, mlir::Type /*varType*/, mlir::Value originalVar,
    bool &needsFree) const {
  auto ptrTy = mlir::cast<cir::PointerType>(pointer);
  mlir::Type viewPointee = ptrTy.getPointee();
  std::optional<Object> object = knownSizeObject(
      builder, loc, originalVar, viewPointee, /*needCount=*/true);
  if (!object)
    return {};
  mlir::Type objectType = object->type;
  mlir::Value dynSize = object->count;

  mlir::Type objectPtrTy =
      ptrTy.getAddrSpace()
          ? cir::PointerType::get(objectType, ptrTy.getAddrSpace())
          : cir::PointerType::get(objectType);
  std::optional<mlir::IntegerAttr> alignment =
      allocaAlignment(builder, objectType);
  if (!alignment)
    return {};

  // cir.alloca needs no deallocation.
  needsFree = false;
  cir::AllocaOp alloca =
      dynSize ? cir::AllocaOp::create(builder, loc, objectPtrTy, varName,
                                      *alignment, dynSize)
              : cir::AllocaOp::create(builder, loc, objectPtrTy, varName,
                                      *alignment);
  if (!varName.empty())
    alloca->setDiscardableAttr(
        mlir::acc::getVarNameAttrName(),
        mlir::acc::VarNameAttr::get(builder.getContext(), varName));

  mlir::Value result = alloca.getResult();
  if (result.getType() == pointer)
    return result;
  if (!dynSize) {
    if (auto array = mlir::dyn_cast<cir::ArrayType>(objectType))
      if (array.getElementType() == viewPointee)
        return cir::CastOp::create(builder, loc, pointer,
                                   cir::CastKind::array_to_ptrdecay, result);
  }
  return castPointer(builder, loc, result, pointer);
}

template <>
bool OpenACCPointerLikeModel<cir::PointerType>::genFree(
    mlir::Type /*pointer*/, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
    mlir::Value allocRes, mlir::Type /*varType*/) const {
  mlir::Value inspected = allocRes ? allocRes : varToFree;
  return knownSizeObject(builder, loc, inspected, /*fallback=*/{},
                         /*needCount=*/false)
      .has_value();
}

template <>
bool OpenACCPointerLikeModel<cir::PointerType>::genCopy(
    mlir::Type /*pointer*/, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> destination,
    mlir::TypedValue<mlir::acc::PointerLikeType> source,
    mlir::Type /*varType*/) const {
  auto destTy = mlir::dyn_cast<cir::PointerType>(destination.getType());
  auto srcTy = mlir::dyn_cast<cir::PointerType>(source.getType());
  if (!destTy || !srcTy || destTy.getPointee() != srcTy.getPointee())
    return false;

  std::optional<Object> object = knownSizeObject(
      builder, loc, source, destTy.getPointee(), /*needCount=*/false);
  if (!object)
    return false;
  mlir::Type objectType = object->type;

  mlir::Value srcObj =
      castPointer(builder, loc, source, pointerTo(objectType, srcTy));
  mlir::Value dstObj =
      castPointer(builder, loc, destination, pointerTo(objectType, destTy));
  if (!srcObj || !dstObj)
    return false;
  cir::CopyOp::create(builder, loc, dstObj, srcObj, {}, {});
  return true;
}

template <>
mlir::Value OpenACCPointerLikeModel<cir::PointerType>::genLoad(
    mlir::Type /*pointer*/, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
    mlir::Type valueType) const {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(srcPtr.getType());
  if (!ptrTy || !hasKnownLayout(ptrTy.getPointee()))
    return {};
  mlir::Value loaded = cir::LoadOp::create(builder, loc, srcPtr);
  if (!valueType || loaded.getType() == valueType)
    return loaded;
  return castPointer(builder, loc, loaded, valueType);
}

template <>
bool OpenACCPointerLikeModel<cir::PointerType>::genStore(
    mlir::Type /*pointer*/, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value valueToStore,
    mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(destPtr.getType());
  if (!ptrTy || !hasKnownLayout(ptrTy.getPointee()))
    return false;
  mlir::Value stored = valueToStore;
  if (stored.getType() != ptrTy.getPointee()) {
    stored = castPointer(builder, loc, stored, ptrTy.getPointee());
    if (!stored)
      return false;
  }
  cir::StoreOp::create(builder, loc, stored, destPtr);
  return true;
}

template <>
mlir::Value OpenACCPointerLikeModel<cir::PointerType>::genCast(
    mlir::Type /*pointer*/, mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value value, mlir::Type resultType) const {
  return castPointer(builder, loc, value, resultType);
}

template <>
mlir::MemRefType OpenACCPointerLikeModel<cir::PointerType>::getAsMemRefType(
    mlir::Type pointer, mlir::ModuleOp module) const {
  std::optional<mlir::DataLayout> layout = mlir::acc::getDataLayout(module);
  if (!layout)
    return {};
  return convertPointerToMemRefType(pointer, *layout);
}

template <>
bool OpenACCPointerLikeModel<cir::PointerType>::isDeviceAccessible(
    mlir::Type /*pointer*/, mlir::Value var,
    mlir::SymbolTableCollection * /*symbolTables*/) const {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(var.getType());
  if (!ptrTy)
    return false;
  auto lang = mlir::dyn_cast_if_present<cir::LangAddressSpaceAttr>(
      ptrTy.getAddrSpace());
  return lang && isDeviceAccessibleAddressSpace(lang.getValue());
}

template <>
bool OpenACCPointerLikeModel<cir::PointerType>::isInDeviceMemory(
    mlir::Type /*pointer*/, mlir::Value var,
    mlir::SymbolTableCollection * /*symbolTables*/) const {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(var.getType());
  if (!ptrTy)
    return false;
  auto lang = mlir::dyn_cast_if_present<cir::LangAddressSpaceAttr>(
      ptrTy.getAddrSpace());
  return lang && isDeviceResidentAddressSpace(lang.getValue());
}

} // namespace cir::acc
