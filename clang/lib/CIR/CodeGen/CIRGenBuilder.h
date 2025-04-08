//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H

#include "CIRGenTypeCache.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/STLExtras.h"

namespace clang::CIRGen {

class CIRGenBuilderTy : public cir::CIRBaseBuilderTy {
  const CIRGenTypeCache &typeCache;

public:
  CIRGenBuilderTy(mlir::MLIRContext &mlirContext, const CIRGenTypeCache &tc)
      : CIRBaseBuilderTy(mlirContext), typeCache(tc) {}

  cir::LongDoubleType getLongDoubleTy(const llvm::fltSemantics &format) const {
    if (&format == &llvm::APFloat::IEEEdouble())
      return cir::LongDoubleType::get(getContext(), typeCache.DoubleTy);
    if (&format == &llvm::APFloat::x87DoubleExtended())
      return cir::LongDoubleType::get(getContext(), typeCache.FP80Ty);
    if (&format == &llvm::APFloat::IEEEquad())
      return cir::LongDoubleType::get(getContext(), typeCache.FP128Ty);
    if (&format == &llvm::APFloat::PPCDoubleDouble())
      llvm_unreachable("NYI: PPC double-double format for long double");
    llvm_unreachable("Unsupported format for long double");
  }

  bool isSized(mlir::Type ty) {
    if (mlir::isa<cir::PointerType, cir::ArrayType, cir::BoolType,
                  cir::IntType>(ty))
      return true;

    assert(!cir::MissingFeatures::unsizedTypes());
    return false;
  }

  // Return true if the value is a null constant such as null pointer, (+0.0)
  // for floating-point or zero initializer
  bool isNullValue(mlir::Attribute attr) const {
    if (mlir::isa<cir::ZeroAttr>(attr))
      return true;

    if (const auto ptrVal = mlir::dyn_cast<cir::ConstPtrAttr>(attr))
      return ptrVal.isNullValue();

    if (const auto intVal = mlir::dyn_cast<cir::IntAttr>(attr))
      return intVal.isNullValue();

    if (const auto boolVal = mlir::dyn_cast<cir::BoolAttr>(attr))
      return !boolVal.getValue();

    if (auto fpAttr = mlir::dyn_cast<cir::FPAttr>(attr)) {
      auto fpVal = fpAttr.getValue();
      bool ignored;
      llvm::APFloat fv(+0.0);
      fv.convert(fpVal.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                 &ignored);
      return fv.bitwiseIsEqual(fpVal);
    }

    if (const auto arrayVal = mlir::dyn_cast<cir::ConstArrayAttr>(attr)) {
      if (mlir::isa<mlir::StringAttr>(arrayVal.getElts()))
        return false;

      return llvm::all_of(
          mlir::cast<mlir::ArrayAttr>(arrayVal.getElts()),
          [&](const mlir::Attribute &elt) { return isNullValue(elt); });
    }
    return false;
  }

  //
  // Type helpers
  // ------------
  //
  cir::IntType getUIntNTy(int n) {
    switch (n) {
    case 8:
      return getUInt8Ty();
    case 16:
      return getUInt16Ty();
    case 32:
      return getUInt32Ty();
    case 64:
      return getUInt64Ty();
    default:
      return cir::IntType::get(getContext(), n, false);
    }
  }

  cir::IntType getSIntNTy(int n) {
    switch (n) {
    case 8:
      return getSInt8Ty();
    case 16:
      return getSInt16Ty();
    case 32:
      return getSInt32Ty();
    case 64:
      return getSInt64Ty();
    default:
      return cir::IntType::get(getContext(), n, true);
    }
  }

  cir::VoidType getVoidTy() { return typeCache.VoidTy; }

  cir::IntType getSInt8Ty() { return typeCache.SInt8Ty; }
  cir::IntType getSInt16Ty() { return typeCache.SInt16Ty; }
  cir::IntType getSInt32Ty() { return typeCache.SInt32Ty; }
  cir::IntType getSInt64Ty() { return typeCache.SInt64Ty; }

  cir::IntType getUInt8Ty() { return typeCache.UInt8Ty; }
  cir::IntType getUInt16Ty() { return typeCache.UInt16Ty; }
  cir::IntType getUInt32Ty() { return typeCache.UInt32Ty; }
  cir::IntType getUInt64Ty() { return typeCache.UInt64Ty; }

  bool isInt8Ty(mlir::Type i) {
    return i == typeCache.UInt8Ty || i == typeCache.SInt8Ty;
  }
  bool isInt16Ty(mlir::Type i) {
    return i == typeCache.UInt16Ty || i == typeCache.SInt16Ty;
  }
  bool isInt32Ty(mlir::Type i) {
    return i == typeCache.UInt32Ty || i == typeCache.SInt32Ty;
  }
  bool isInt64Ty(mlir::Type i) {
    return i == typeCache.UInt64Ty || i == typeCache.SInt64Ty;
  }
  bool isInt(mlir::Type i) { return mlir::isa<cir::IntType>(i); }

  // Creates constant nullptr for pointer type ty.
  cir::ConstantOp getNullPtr(mlir::Type ty, mlir::Location loc) {
    assert(!cir::MissingFeatures::targetCodeGenInfoGetNullPointer());
    return create<cir::ConstantOp>(loc, ty, getConstPtrAttr(ty, 0));
  }

  mlir::Value createNeg(mlir::Value value) {

    if (auto intTy = mlir::dyn_cast<cir::IntType>(value.getType())) {
      // Source is a unsigned integer: first cast it to signed.
      if (intTy.isUnsigned())
        value = createIntCast(value, getSIntNTy(intTy.getWidth()));
      return create<cir::UnaryOp>(value.getLoc(), value.getType(),
                                  cir::UnaryOpKind::Minus, value);
    }

    llvm_unreachable("negation for the given type is NYI");
  }

  // TODO: split this to createFPExt/createFPTrunc when we have dedicated cast
  // operations.
  mlir::Value createFloatingCast(mlir::Value v, mlir::Type destType) {
    assert(!cir::MissingFeatures::fpConstraints());

    return create<cir::CastOp>(v.getLoc(), destType, cir::CastKind::floating,
                               v);
  }

  mlir::Value createFSub(mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    assert(!cir::MissingFeatures::fpConstraints());
    assert(!cir::MissingFeatures::fastMathFlags());

    return create<cir::BinOp>(loc, cir::BinOpKind::Sub, lhs, rhs);
  }

  mlir::Value createFAdd(mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    assert(!cir::MissingFeatures::fpConstraints());
    assert(!cir::MissingFeatures::fastMathFlags());

    return create<cir::BinOp>(loc, cir::BinOpKind::Add, lhs, rhs);
  }
  mlir::Value createFMul(mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    assert(!cir::MissingFeatures::fpConstraints());
    assert(!cir::MissingFeatures::fastMathFlags());

    return create<cir::BinOp>(loc, cir::BinOpKind::Mul, lhs, rhs);
  }
  mlir::Value createFDiv(mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    assert(!cir::MissingFeatures::fpConstraints());
    assert(!cir::MissingFeatures::fastMathFlags());

    return create<cir::BinOp>(loc, cir::BinOpKind::Div, lhs, rhs);
  }

  /// Create a cir.ptr_stride operation to get access to an array element.
  /// \p idx is the index of the element to access, \p shouldDecay is true if
  /// the result should decay to a pointer to the element type.
  mlir::Value getArrayElement(mlir::Location arrayLocBegin,
                              mlir::Location arrayLocEnd, mlir::Value arrayPtr,
                              mlir::Type eltTy, mlir::Value idx,
                              bool shouldDecay);

  /// Returns a decayed pointer to the first element of the array
  /// pointed to by \p arrayPtr.
  mlir::Value maybeBuildArrayDecay(mlir::Location loc, mlir::Value arrayPtr,
                                   mlir::Type eltTy);
};

} // namespace clang::CIRGen

#endif
