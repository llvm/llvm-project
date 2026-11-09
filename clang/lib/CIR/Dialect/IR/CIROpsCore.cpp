//===- CIROpsCore.cpp - CIR ops in partition `Core` -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `Core`: the hand-written
// parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsCore.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

void cir::AllocaOp::build(mlir::OpBuilder &odsBuilder,
                          mlir::OperationState &odsState, mlir::Type addr,
                          llvm::StringRef name, mlir::IntegerAttr alignment) {
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(addr);
}

cir::AllocaOp cir::getUnderlyingAlloca(mlir::Value addr) {
  mlir::Value ptr = addr;
  while (cir::CastOp castOp = ptr.getDefiningOp<cir::CastOp>()) {
    if (!castOp.isAllocaPreservingCast())
      break;
    ptr = castOp.getSrc();
  }
  return ptr.getDefiningOp<cir::AllocaOp>();
}

//===----------------------------------------------------------------------===//
// DeleteArrayOp
//===----------------------------------------------------------------------===//

LogicalResult cir::DeleteArrayOp::verify() {
  if (getDtorMayThrow() && !getElementDtorAttr())
    return emitOpError(
        "'dtor_may_throw' requires an 'element_dtor' to be present");
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult cir::checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                      mlir::Attribute attrType) {
  if (isa<cir::ConstPtrAttr>(attrType)) {
    if (!mlir::isa<cir::PointerType>(opType))
      return op->emitOpError(
          "pointer constant initializing a non-pointer type");
    return success();
  }

  if (isa<cir::DataMemberAttr, cir::DataMemberOffsetAttr, cir::MethodAttr>(
          attrType)) {
    // More detailed type verifications are already done in
    // DataMemberAttr::verify or MethodAttr::verify. Don't need to repeat here.
    return success();
  }

  if (isa<cir::ZeroAttr>(attrType)) {
    if (isa<cir::RecordType, cir::ArrayType, cir::MatrixType, cir::VectorType,
            cir::ComplexType>(opType))
      return success();
    return op->emitOpError(
        "zero expects struct, array, vector, or complex type");
  }

  if (mlir::isa<cir::UndefAttr>(attrType)) {
    if (!mlir::isa<cir::VoidType>(opType))
      return success();
    return op->emitOpError("undef expects non-void type");
  }

  if (mlir::isa<cir::BoolAttr>(attrType)) {
    if (!mlir::isa<cir::BoolType>(opType))
      return op->emitOpError("result type (")
             << opType << ") must be '!cir.bool' for '" << attrType << "'";
    return success();
  }

  if (mlir::isa<cir::IntAttr, cir::FPAttr>(attrType)) {
    auto at = cast<TypedAttr>(attrType);
    if (at.getType() != opType) {
      return op->emitOpError("result type (")
             << opType << ") does not match value type (" << at.getType()
             << ")";
    }
    return success();
  }

  if (mlir::isa<cir::BlockAddrDiffAttr, cir::BlockAddrInfoAttr,
                cir::ConstArrayAttr, cir::ConstVectorAttr,
                cir::ConstComplexAttr, cir::ConstRecordAttr,
                cir::GlobalOffsetAttr, cir::GlobalViewAttr, cir::PoisonAttr,
                cir::TypeInfoAttr, cir::VTableAttr>(attrType))
    return success();

  assert(isa<TypedAttr>(attrType) && "What else could we be looking at here?");
  return op->emitOpError("global with type ")
         << cast<TypedAttr>(attrType).getType() << " not yet supported";
}

LogicalResult cir::ConstantOp::verify() {
  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  return checkConstantTypes(getOperation(), getType(), getValue());
}

OpFoldResult cir::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

static bool isFloatingPointCastKind(cir::CastKind kind) {
  switch (kind) {
  case cir::CastKind::floating:
  case cir::CastKind::int_to_float:
  case cir::CastKind::float_to_int:
  case cir::CastKind::float_to_bool:
  case cir::CastKind::bool_to_float:
  case cir::CastKind::float_to_complex:
  case cir::CastKind::float_complex_to_real:
  case cir::CastKind::float_complex_to_bool:
  case cir::CastKind::float_complex:
  case cir::CastKind::float_complex_to_int_complex:
  case cir::CastKind::int_complex_to_float_complex:
    return true;
  default:
    return false;
  }
}

LogicalResult cir::CastOp::verify() {
  mlir::Type resType = getType();
  mlir::Type srcType = getSrc().getType();
  cir::CastKind kind = getKind();

  if (getFenvAttr() && !isFloatingPointCastKind(kind))
    return emitOpError()
           << "'fenv' is only valid for floating-point cast kinds";

  // Verify address space casts for pointer types. given that
  // casts for within a different address space are illegal.
  auto srcPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
  auto resPtrTy = mlir::dyn_cast<cir::PointerType>(resType);
  if (srcPtrTy && resPtrTy && (kind != cir::CastKind::address_space))
    if (srcPtrTy.getAddrSpace() != resPtrTy.getAddrSpace()) {
      return emitOpError() << "result type address space does not match the "
                              "address space of the operand";
    }

  auto srcVTy = mlir::dyn_cast<cir::VectorType>(srcType);
  auto resVTy = mlir::dyn_cast<cir::VectorType>(resType);
  if (srcVTy && resVTy) {
    if ((kind == cir::CastKind::int_to_float ||
         kind == cir::CastKind::float_to_int) &&
        srcVTy.getSize() != resVTy.getSize()) {
      return emitOpError()
             << "vector float-to-int and int-to-float casts require "
                "source and destination vectors to have the same number of "
                "elements";
    }
    // Use the element type of the vector to verify the cast kind. (Except for
    // bitcast, see below.)
    srcType = srcVTy.getElementType();
    resType = resVTy.getElementType();
  }

  switch (kind) {
  case cir::CastKind::int_to_bool: {
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    return success();
  }
  case cir::CastKind::ptr_to_bool: {
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    if (!mlir::isa<cir::PointerType>(srcType))
      return emitOpError() << "requires !cir.ptr type for source";
    return success();
  }
  case cir::CastKind::integral: {
    if (!mlir::isa<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    return success();
  }
  case cir::CastKind::array_to_ptrdecay: {
    const auto arrayPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    const auto flatPtrTy = mlir::dyn_cast<cir::PointerType>(resType);
    if (!arrayPtrTy || !flatPtrTy)
      return emitOpError() << "requires !cir.ptr type for source and result";

    // TODO(CIR): Make sure the AddrSpace of both types are equals
    return success();
  }
  case cir::CastKind::bitcast: {
    // Handle the pointer types first.
    auto srcPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    auto resPtrTy = mlir::dyn_cast<cir::PointerType>(resType);

    if (srcPtrTy && resPtrTy) {
      return success();
    }

    return success();
  }
  case cir::CastKind::floating: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType) ||
        !mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for source and result";
    return success();
  }
  case cir::CastKind::float_to_int: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType))
      return emitOpError() << "requires !cir.float type for source";
    if (!mlir::dyn_cast<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    return success();
  }
  case cir::CastKind::int_to_ptr: {
    if (!mlir::dyn_cast<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    if (!mlir::dyn_cast<cir::PointerType>(resType))
      return emitOpError() << "requires !cir.ptr type for result";
    return success();
  }
  case cir::CastKind::ptr_to_int: {
    if (!mlir::dyn_cast<cir::PointerType>(srcType))
      return emitOpError() << "requires !cir.ptr type for source";
    if (!mlir::dyn_cast<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    return success();
  }
  case cir::CastKind::float_to_bool: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType))
      return emitOpError() << "requires !cir.float type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::bool_to_int: {
    if (!mlir::isa<cir::BoolType>(srcType))
      return emitOpError() << "requires !cir.bool type for source";
    if (!mlir::isa<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    return success();
  }
  case cir::CastKind::int_to_float: {
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    if (!mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    return success();
  }
  case cir::CastKind::bool_to_float: {
    if (!mlir::isa<cir::BoolType>(srcType))
      return emitOpError() << "requires !cir.bool type for source";
    if (!mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    return success();
  }
  case cir::CastKind::address_space: {
    auto srcPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    auto resPtrTy = mlir::dyn_cast<cir::PointerType>(resType);
    if (!srcPtrTy || !resPtrTy)
      return emitOpError() << "requires !cir.ptr type for source and result";
    if (srcPtrTy.getPointee() != resPtrTy.getPointee())
      return emitOpError() << "requires two types differ in addrspace only";
    return success();
  }
  case cir::CastKind::float_to_complex: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType))
      return emitOpError() << "requires !cir.float type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy)
      return emitOpError() << "requires !cir.complex type for result";
    if (srcType != resComplexTy.getElementType())
      return emitOpError() << "requires source type match result element type";
    return success();
  }
  case cir::CastKind::int_to_complex: {
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy)
      return emitOpError() << "requires !cir.complex type for result";
    if (srcType != resComplexTy.getElementType())
      return emitOpError() << "requires source type match result element type";
    return success();
  }
  case cir::CastKind::float_complex_to_real: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy)
      return emitOpError() << "requires !cir.complex type for source";
    if (!mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    if (srcComplexTy.getElementType() != resType)
      return emitOpError() << "requires source element type match result type";
    return success();
  }
  case cir::CastKind::int_complex_to_real: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy)
      return emitOpError() << "requires !cir.complex type for source";
    if (!mlir::isa<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    if (srcComplexTy.getElementType() != resType)
      return emitOpError() << "requires source element type match result type";
    return success();
  }
  case cir::CastKind::float_complex_to_bool: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::int_complex_to_bool: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isIntegerComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::float_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for result";
    return success();
  }
  case cir::CastKind::float_complex_to_int_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for result";
    return success();
  }
  case cir::CastKind::int_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for result";
    return success();
  }
  case cir::CastKind::int_complex_to_float_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for result";
    return success();
  }
  case cir::CastKind::member_ptr_to_bool: {
    if (!mlir::isa<cir::DataMemberType, cir::MethodType>(srcType))
      return emitOpError()
             << "requires !cir.data_member or !cir.method type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  }
  llvm_unreachable("Unknown CastOp kind?");
}

static bool isIntOrBoolCast(cir::CastOp op) {
  auto kind = op.getKind();
  return kind == cir::CastKind::bool_to_int ||
         kind == cir::CastKind::int_to_bool || kind == cir::CastKind::integral;
}

static bool isCirFunctionPointerType(mlir::Type ty) {
  const auto ptrTy = mlir::dyn_cast<cir::PointerType>(ty);
  return ptrTy && mlir::isa<cir::FuncType>(ptrTy.getPointee());
}

static Value tryFoldCastChain(cir::CastOp op) {
  cir::CastOp head = op, tail = op;

  while (op) {
    if (!isIntOrBoolCast(op))
      break;
    head = op;
    op = head.getSrc().getDefiningOp<cir::CastOp>();
  }

  if (head != tail) {
    // if bool_to_int -> ...  -> int_to_bool: take the bool
    // as we had it was before all casts
    if (head.getKind() == cir::CastKind::bool_to_int &&
        tail.getKind() == cir::CastKind::int_to_bool)
      return head.getSrc();

    // if int_to_bool -> ...  -> int_to_bool: take the result
    // of the first one, as no other casts (and ext casts as well)
    // don't change the first result
    if (head.getKind() == cir::CastKind::int_to_bool &&
        tail.getKind() == cir::CastKind::int_to_bool)
      return head.getResult();

    return {};
  }

  // Bitcast round-trip on function pointers: T0 -> T1 -> T0 (e.g. no-proto
  // redeclaration vs. actual prototype). Restrict to function pointers so
  // other pointer bitcast chains are unchanged.
  if (tail.getKind() == cir::CastKind::bitcast) {
    auto *inner = tail.getSrc().getDefiningOp();
    if (inner && isCirFunctionPointerType(tail.getType())) {
      auto innerCast = mlir::dyn_cast<cir::CastOp>(inner);
      if (innerCast && innerCast.getKind() == cir::CastKind::bitcast &&
          innerCast.getSrc().getType() == tail.getType() &&
          innerCast.getType() == tail.getSrc().getType()) {
        return innerCast.getSrc();
      }
    }
  }

  return {};
}

OpFoldResult cir::CastOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getSrc())) {
    // Propagate poison value
    return cir::PoisonAttr::get(getContext(), getType());
  }

  // Propagate Undef value
  if (mlir::isa_and_present<cir::UndefAttr>(adaptor.getSrc()))
    return cir::UndefAttr::get(getType());

  if (getSrc().getType() == getType()) {
    switch (getKind()) {
    case cir::CastKind::integral:
    case cir::CastKind::bitcast:
    case cir::CastKind::address_space:
    case cir::CastKind::float_complex:
    case cir::CastKind::int_complex: {
      return getSrc();
    }
    default:
      return {};
    }
  }

  // Handle cases where a chain of casts cancel out.
  Value result = tryFoldCastChain(*this);
  if (result)
    return result;

  // Handle simple constant casts.
  if (auto srcConst = getSrc().getDefiningOp<cir::ConstantOp>()) {
    switch (getKind()) {
    case cir::CastKind::integral: {
      mlir::Type srcTy = getSrc().getType();
      // Don't try to fold vector casts for now.
      assert(mlir::isa<cir::VectorType>(srcTy) ==
             mlir::isa<cir::VectorType>(getType()));
      if (mlir::isa<cir::VectorType>(srcTy))
        break;

      auto srcIntTy = mlir::cast<cir::IntType>(srcTy);
      auto dstIntTy = mlir::cast<cir::IntType>(getType());
      auto constIntAttr = srcConst.getValueAttr<cir::IntAttr>();
      if (!constIntAttr)
        return {};

      APInt srcValue = constIntAttr.getValue();
      APInt newVal = srcIntTy.isSigned()
                         ? srcValue.sextOrTrunc(dstIntTy.getWidth())
                         : srcValue.zextOrTrunc(dstIntTy.getWidth());
      return cir::IntAttr::get(dstIntTy, newVal);
    }
    default:
      break;
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BuiltinIntCastOp
//===----------------------------------------------------------------------===//

LogicalResult cir::BuiltinIntCastOp::verify() {
  mlir::Type srcType = getSrc().getType();
  mlir::Type resType = getType();

  auto srcCirInt = mlir::dyn_cast<cir::IntType>(srcType);
  auto resCirInt = mlir::dyn_cast<cir::IntType>(resType);

  // One side must be a CIR integer the other must be a builtin
  // integer or index type.
  if (static_cast<bool>(srcCirInt) == static_cast<bool>(resCirInt))
    return emitOpError()
           << "requires exactly one '!cir.int' operand or result; the other "
              "must be a builtin integer or 'index' type";

  mlir::Type builtinType = srcCirInt ? resType : srcType;
  if (!mlir::isa<mlir::IntegerType, mlir::IndexType>(builtinType))
    return emitOpError() << "requires a builtin integer or 'index' type on the "
                            "non-CIR side";

  // The cast preserves bit width. 'index' has no fixed width, so only check
  // when the builtin side is a fixed-width integer.
  if (auto builtinInt = mlir::dyn_cast<mlir::IntegerType>(builtinType)) {
    cir::IntType cirInt = srcCirInt ? srcCirInt : resCirInt;
    if (cirInt.getWidth() != builtinInt.getWidth())
      return emitOpError()
             << "requires the CIR and builtin integer types to have the same "
                "width; use 'cir.cast' for width conversions";
  }

  return success();
}

OpFoldResult cir::BuiltinIntCastOp::fold(FoldAdaptor adaptor) {
  // Fold: builtin_int_cast(builtin_int_cast(x)) -> x
  // Inner source type must match the cast's result type.
  if (auto inner = getSrc().getDefiningOp<cir::BuiltinIntCastOp>())
    if (inner.getSrc().getType() == getType())
      return inner.getSrc();
  return {};
}

//===----------------------------------------------------------------------===//
// PtrMaskOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::PtrMaskOp::verify() {
  mlir::DataLayout layout = mlir::DataLayout::closest(*this);
  std::optional<uint64_t> indexWidth =
      layout.getTypeIndexBitwidth(getPtr().getType());
  if (!indexWidth)
    return emitOpError() << "pointer has no index width";

  uint64_t maskWidth = getMask().getType().getWidth();
  if (maskWidth != *indexWidth)
    return emitOpError() << "mask width " << maskWidth
                         << " must equal the pointer index width "
                         << *indexWidth;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GetMemberOp Definitions
//===----------------------------------------------------------------------===//

static mlir::Type memberPointeeType(cir::RecordType recordTy, unsigned idx) {
  return cir::memberStorageType(recordTy.getMembers()[idx]);
}

LogicalResult cir::GetMemberOp::verify() {
  const auto recordTy = dyn_cast<RecordType>(getAddrTy().getPointee());
  if (!recordTy)
    return emitError() << "expected pointer to a record type";

  if (recordTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of bounds";

  mlir::Type pointeeTy = memberPointeeType(recordTy, getIndex());
  if (!pointeeTy)
    return emitError() << "member owns no storage to point at";

  if (pointeeTy != getType().getPointee())
    return emitError() << "member type mismatch";

  if (getAddrTy().getAddrSpace() != getType().getAddrSpace())
    return emitError() << "address space mismatch";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ExtractMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::ExtractMemberOp::verify() {
  if (mlir::isa<cir::UnionType>(getRecord().getType()))
    return emitError()
           << "cir.extract_member currently does not support unions";
  auto structTy = mlir::cast<cir::StructType>(getRecord().getType());
  if (structTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of bounds";
  mlir::Type memberTy = structTy.getMembers()[getIndex()];
  if (mlir::isa<cir::BitFieldType>(memberTy))
    return emitError() << "cir.extract_member does not support bit-fields";
  if (memberTy != getType())
    return emitError() << "member type mismatch";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// InsertMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::InsertMemberOp::verify() {
  if (mlir::isa<cir::UnionType>(getRecord().getType()))
    return emitError() << "cir.insert_member currently does not support unions";
  auto structTy = mlir::cast<cir::StructType>(getRecord().getType());
  if (structTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of bounds";
  mlir::Type memberTy = structTy.getMembers()[getIndex()];
  if (mlir::isa<cir::BitFieldType>(memberTy))
    return emitError() << "cir.insert_member does not support bit-fields";
  if (memberTy != getValue().getType())
    return emitError() << "member type mismatch";
  // The op trait already checks that the types of $result and $record match.
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsCore.cpp.inc"
