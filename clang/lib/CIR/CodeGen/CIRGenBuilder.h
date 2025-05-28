//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENBUILDER_H

#include "Address.h"
#include "CIRGenTypeCache.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/STLExtras.h"

namespace clang::CIRGen {

class CIRGenBuilderTy : public cir::CIRBaseBuilderTy {
  const CIRGenTypeCache &typeCache;
  llvm::StringMap<unsigned> recordNames;

public:
  CIRGenBuilderTy(mlir::MLIRContext &mlirContext, const CIRGenTypeCache &tc)
      : CIRBaseBuilderTy(mlirContext), typeCache(tc) {}

  /// Get a cir::ConstArrayAttr for a string literal.
  /// Note: This is different from what is returned by
  /// mlir::Builder::getStringAttr() which is an mlir::StringAttr.
  mlir::Attribute getString(llvm::StringRef str, mlir::Type eltTy,
                            std::optional<size_t> size) {
    size_t finalSize = size.value_or(str.size());

    size_t lastNonZeroPos = str.find_last_not_of('\0');
    // If the string is full of null bytes, emit a #cir.zero rather than
    // a #cir.const_array.
    if (lastNonZeroPos == llvm::StringRef::npos) {
      auto arrayTy = cir::ArrayType::get(eltTy, finalSize);
      return cir::ZeroAttr::get(arrayTy);
    }
    // We emit trailing zeros only if there are multiple trailing zeros.
    size_t trailingZerosNum = 0;
    if (finalSize > lastNonZeroPos + 2)
      trailingZerosNum = finalSize - lastNonZeroPos - 1;
    auto truncatedArrayTy =
        cir::ArrayType::get(eltTy, finalSize - trailingZerosNum);
    auto fullArrayTy = cir::ArrayType::get(eltTy, finalSize);
    return cir::ConstArrayAttr::get(
        fullArrayTy,
        mlir::StringAttr::get(str.drop_back(trailingZerosNum),
                              truncatedArrayTy),
        trailingZerosNum);
  }

  std::string getUniqueAnonRecordName() { return getUniqueRecordName("anon"); }

  std::string getUniqueRecordName(const std::string &baseName) {
    auto it = recordNames.find(baseName);
    if (it == recordNames.end()) {
      recordNames[baseName] = 0;
      return baseName;
    }

    return baseName + "." + std::to_string(recordNames[baseName]++);
  }

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

  /// Get a CIR record kind from a AST declaration tag.
  cir::RecordType::RecordKind getRecordKind(const clang::TagTypeKind kind) {
    switch (kind) {
    case clang::TagTypeKind::Class:
    case clang::TagTypeKind::Struct:
      return cir::RecordType::Struct;
    case clang::TagTypeKind::Union:
      return cir::RecordType::Union;
    case clang::TagTypeKind::Interface:
      llvm_unreachable("interface records are NYI");
    case clang::TagTypeKind::Enum:
      llvm_unreachable("enums are not records");
    }
    llvm_unreachable("Unsupported record kind");
  }

  /// Get an incomplete CIR struct type. If we have a complete record
  /// declaration, we may create an incomplete type and then add the
  /// members, so \p rd here may be complete.
  cir::RecordType getIncompleteRecordTy(llvm::StringRef name,
                                        const clang::RecordDecl *rd) {
    const mlir::StringAttr nameAttr = getStringAttr(name);
    cir::RecordType::RecordKind kind = cir::RecordType::RecordKind::Struct;
    if (rd)
      kind = getRecordKind(rd->getTagKind());
    return getType<cir::RecordType>(nameAttr, kind);
  }

  bool isSized(mlir::Type ty) {
    if (mlir::isa<cir::PointerType, cir::ArrayType, cir::BoolType,
                  cir::IntType>(ty))
      return true;

    if (const auto vt = mlir::dyn_cast<cir::VectorType>(ty))
      return isSized(vt.getElementType());

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

  //
  // Constant creation helpers
  // -------------------------
  //
  cir::ConstantOp getSInt32(int32_t c, mlir::Location loc) {
    return getConstantInt(loc, getSInt32Ty(), c);
  }

  // Creates constant nullptr for pointer type ty.
  cir::ConstantOp getNullPtr(mlir::Type ty, mlir::Location loc) {
    assert(!cir::MissingFeatures::targetCodeGenInfoGetNullPointer());
    return create<cir::ConstantOp>(loc, getConstPtrAttr(ty, 0));
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

  cir::LoadOp createLoad(mlir::Location loc, Address addr,
                         bool isVolatile = false) {
    mlir::IntegerAttr align;
    uint64_t alignment = addr.getAlignment().getQuantity();
    if (alignment)
      align = getI64IntegerAttr(alignment);
    return create<cir::LoadOp>(loc, addr.getPointer(), /*isDeref=*/false,
                               align);
  }

  cir::StoreOp createStore(mlir::Location loc, mlir::Value val, Address dst,
                           ::mlir::IntegerAttr align = {}) {
    if (!align) {
      uint64_t alignment = dst.getAlignment().getQuantity();
      if (alignment)
        align = mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 64),
                                       alignment);
    }
    return CIRBaseBuilderTy::createStore(loc, val, dst.getPointer(), align);
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
