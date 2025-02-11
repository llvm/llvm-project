//===-- CIRGenBuilder.h - CIRBuilder implementation  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H

#include "Address.h"
#include "CIRGenRecordLayout.h"
#include "CIRGenTypeCache.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <optional>
#include <string>
#include <utility>

namespace clang::CIRGen {

class CIRGenFunction;

class CIRGenBuilderTy : public cir::CIRBaseBuilderTy {
  const CIRGenTypeCache &typeCache;
  bool IsFPConstrained = false;
  cir::fp::ExceptionBehavior DefaultConstrainedExcept = cir::fp::ebStrict;
  llvm::RoundingMode DefaultConstrainedRounding = llvm::RoundingMode::Dynamic;

  llvm::StringMap<unsigned> GlobalsVersioning;
  llvm::StringMap<unsigned> RecordNames;

public:
  CIRGenBuilderTy(mlir::MLIRContext &C, const CIRGenTypeCache &tc)
      : CIRBaseBuilderTy(C), typeCache(tc) {
    RecordNames["anon"] = 0; // in order to start from the name "anon.0"
  }

  std::string getUniqueAnonRecordName() { return getUniqueRecordName("anon"); }

  std::string getUniqueRecordName(const std::string &baseName) {
    auto it = RecordNames.find(baseName);
    if (it == RecordNames.end()) {
      RecordNames[baseName] = 0;
      return baseName;
    }

    return baseName + "." + std::to_string(RecordNames[baseName]++);
  }

  //
  // Floating point specific helpers
  // -------------------------------
  //

  /// Enable/Disable use of constrained floating point math. When enabled the
  /// CreateF<op>() calls instead create constrained floating point intrinsic
  /// calls. Fast math flags are unaffected by this setting.
  void setIsFPConstrained(bool IsCon) {
    if (IsCon)
      llvm_unreachable("Constrained FP NYI");
    IsFPConstrained = IsCon;
  }

  /// Query for the use of constrained floating point math
  bool getIsFPConstrained() {
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");
    return IsFPConstrained;
  }

  /// Set the exception handling to be used with constrained floating point
  void setDefaultConstrainedExcept(cir::fp::ExceptionBehavior NewExcept) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> ExceptStr =
        cir::convertExceptionBehaviorToStr(NewExcept);
    assert(ExceptStr && "Garbage strict exception behavior!");
#endif
    DefaultConstrainedExcept = NewExcept;
  }

  /// Set the rounding mode handling to be used with constrained floating point
  void setDefaultConstrainedRounding(llvm::RoundingMode NewRounding) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> RoundingStr =
        cir::convertRoundingModeToStr(NewRounding);
    assert(RoundingStr && "Garbage strict rounding mode!");
#endif
    DefaultConstrainedRounding = NewRounding;
  }

  /// Get the exception handling used with constrained floating point
  cir::fp::ExceptionBehavior getDefaultConstrainedExcept() {
    return DefaultConstrainedExcept;
  }

  /// Get the rounding mode handling used with constrained floating point
  llvm::RoundingMode getDefaultConstrainedRounding() {
    return DefaultConstrainedRounding;
  }

  //
  // Attribute helpers
  // -----------------
  //

  /// Get constant address of a global variable as an MLIR attribute.
  /// This wrapper infers the attribute type through the global op.
  cir::GlobalViewAttr getGlobalViewAttr(cir::GlobalOp globalOp,
                                        mlir::ArrayAttr indices = {}) {
    auto type = getPointerTo(globalOp.getSymType());
    return getGlobalViewAttr(type, globalOp, indices);
  }

  /// Get constant address of a global variable as an MLIR attribute.
  cir::GlobalViewAttr getGlobalViewAttr(cir::PointerType type,
                                        cir::GlobalOp globalOp,
                                        mlir::ArrayAttr indices = {}) {
    auto symbol = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
    return cir::GlobalViewAttr::get(type, symbol, indices);
  }

  cir::GlobalViewAttr getGlobalViewAttr(cir::PointerType type,
                                        cir::GlobalOp globalOp,
                                        llvm::ArrayRef<int64_t> indices) {
    llvm::SmallVector<mlir::Attribute> attrs;
    for (auto ind : indices) {
      auto a =
          mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 64), ind);
      attrs.push_back(a);
    }

    mlir::ArrayAttr arAttr = mlir::ArrayAttr::get(getContext(), attrs);
    return getGlobalViewAttr(type, globalOp, arAttr);
  }

  mlir::Attribute getString(llvm::StringRef str, mlir::Type eltTy,
                            unsigned size = 0) {
    unsigned finalSize = size ? size : str.size();

    size_t lastNonZeroPos = str.find_last_not_of('\0');
    // If the string is full of null bytes, emit a #cir.zero rather than
    // a #cir.const_array.
    if (lastNonZeroPos == llvm::StringRef::npos) {
      auto arrayTy = cir::ArrayType::get(getContext(), eltTy, finalSize);
      return getZeroAttr(arrayTy);
    }
    // We will use trailing zeros only if there are more than one zero
    // at the end
    int trailingZerosNum =
        finalSize > lastNonZeroPos + 2 ? finalSize - lastNonZeroPos - 1 : 0;
    auto truncatedArrayTy =
        cir::ArrayType::get(getContext(), eltTy, finalSize - trailingZerosNum);
    auto fullArrayTy = cir::ArrayType::get(getContext(), eltTy, finalSize);
    return cir::ConstArrayAttr::get(
        getContext(), fullArrayTy,
        mlir::StringAttr::get(str.drop_back(trailingZerosNum),
                              truncatedArrayTy),
        trailingZerosNum);
  }

  cir::ConstArrayAttr getConstArray(mlir::Attribute attrs,
                                    cir::ArrayType arrayTy) {
    return cir::ConstArrayAttr::get(arrayTy, attrs);
  }

  mlir::Attribute getConstStructOrZeroAttr(mlir::ArrayAttr arrayAttr,
                                           bool packed = false,
                                           bool padded = false,
                                           mlir::Type type = {}) {
    llvm::SmallVector<mlir::Type, 8> members;
    auto structTy = mlir::dyn_cast<cir::StructType>(type);
    assert(structTy && "expected cir.struct");

    // Collect members and check if they are all zero.
    bool isZero = true;
    for (auto &attr : arrayAttr) {
      const auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attr);
      members.push_back(typedAttr.getType());
      isZero &= isNullValue(typedAttr);
    }

    // Struct type not specified: create anon struct type from members.
    if (!structTy)
      structTy = getType<cir::StructType>(members, packed, padded,
                                          cir::StructType::Struct,
                                          /*ast=*/nullptr);

    // Return zero or anonymous constant struct.
    if (isZero)
      return cir::ZeroAttr::get(getContext(), structTy);
    return cir::ConstStructAttr::get(structTy, arrayAttr);
  }

  cir::ConstStructAttr getAnonConstStruct(mlir::ArrayAttr arrayAttr,
                                          bool packed = false,
                                          bool padded = false,
                                          mlir::Type ty = {}) {
    llvm::SmallVector<mlir::Type, 4> members;
    for (auto &f : arrayAttr) {
      auto ta = mlir::dyn_cast<mlir::TypedAttr>(f);
      assert(ta && "expected typed attribute member");
      members.push_back(ta.getType());
    }

    if (!ty)
      ty = getAnonStructTy(members, packed, padded);

    auto sTy = mlir::dyn_cast<cir::StructType>(ty);
    assert(sTy && "expected struct type");
    return cir::ConstStructAttr::get(sTy, arrayAttr);
  }

  cir::TypeInfoAttr getTypeInfo(mlir::ArrayAttr fieldsAttr) {
    auto anonStruct = getAnonConstStruct(fieldsAttr);
    return cir::TypeInfoAttr::get(anonStruct.getType(), fieldsAttr);
  }

  cir::CmpThreeWayInfoAttr getCmpThreeWayInfoStrongOrdering(
      const llvm::APSInt &lt, const llvm::APSInt &eq, const llvm::APSInt &gt) {
    return cir::CmpThreeWayInfoAttr::get(getContext(), lt.getSExtValue(),
                                         eq.getSExtValue(), gt.getSExtValue());
  }

  cir::CmpThreeWayInfoAttr getCmpThreeWayInfoPartialOrdering(
      const llvm::APSInt &lt, const llvm::APSInt &eq, const llvm::APSInt &gt,
      const llvm::APSInt &unordered) {
    return cir::CmpThreeWayInfoAttr::get(getContext(), lt.getSExtValue(),
                                         eq.getSExtValue(), gt.getSExtValue(),
                                         unordered.getSExtValue());
  }

  cir::DataMemberAttr getDataMemberAttr(cir::DataMemberType ty,
                                        unsigned memberIndex) {
    return cir::DataMemberAttr::get(getContext(), ty, memberIndex);
  }

  cir::DataMemberAttr getNullDataMemberAttr(cir::DataMemberType ty) {
    return cir::DataMemberAttr::get(getContext(), ty, std::nullopt);
  }

  // TODO(cir): Once we have CIR float types, replace this by something like a
  // NullableValueInterface to allow for type-independent queries.
  bool isNullValue(mlir::Attribute attr) const {
    if (mlir::isa<cir::ZeroAttr>(attr))
      return true;
    if (const auto ptrVal = mlir::dyn_cast<cir::ConstPtrAttr>(attr))
      return ptrVal.isNullValue();

    if (mlir::isa<cir::GlobalViewAttr>(attr))
      return false;

    // TODO(cir): introduce char type in CIR and check for that instead.
    if (const auto intVal = mlir::dyn_cast<cir::IntAttr>(attr))
      return intVal.isNullValue();

    if (const auto boolVal = mlir::dyn_cast<cir::BoolAttr>(attr))
      return !boolVal.getValue();

    if (auto fpAttr = mlir::dyn_cast<cir::FPAttr>(attr)) {
      auto fpVal = fpAttr.getValue();
      bool ignored;
      llvm::APFloat FV(+0.0);
      FV.convert(fpVal.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                 &ignored);
      return FV.bitwiseIsEqual(fpVal);
    }

    if (const auto structVal = mlir::dyn_cast<cir::ConstStructAttr>(attr)) {
      for (const auto elt : structVal.getMembers()) {
        // FIXME(cir): the struct's ID should not be considered a member.
        if (mlir::isa<mlir::StringAttr>(elt))
          continue;
        if (!isNullValue(elt))
          return false;
      }
      return true;
    }

    if (const auto arrayVal = mlir::dyn_cast<cir::ConstArrayAttr>(attr)) {
      if (mlir::isa<mlir::StringAttr>(arrayVal.getElts()))
        return false;
      for (const auto elt : mlir::cast<mlir::ArrayAttr>(arrayVal.getElts())) {
        if (!isNullValue(elt))
          return false;
      }
      return true;
    }

    llvm_unreachable("NYI");
  }

  //
  // Type helpers
  // ------------
  //
  cir::IntType getUIntNTy(int N) {
    switch (N) {
    case 8:
      return getUInt8Ty();
    case 16:
      return getUInt16Ty();
    case 32:
      return getUInt32Ty();
    case 64:
      return getUInt64Ty();
    default:
      return cir::IntType::get(getContext(), N, false);
    }
  }

  cir::IntType getSIntNTy(int N) {
    switch (N) {
    case 8:
      return getSInt8Ty();
    case 16:
      return getSInt16Ty();
    case 32:
      return getSInt32Ty();
    case 64:
      return getSInt64Ty();
    default:
      return cir::IntType::get(getContext(), N, true);
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

  cir::IntType getExtendedIntTy(cir::IntType ty, bool isSigned) {
    switch (ty.getWidth()) {
    case 8:
      return isSigned ? typeCache.SInt16Ty : typeCache.UInt16Ty;
    case 16:
      return isSigned ? typeCache.SInt32Ty : typeCache.UInt32Ty;
    case 32:
      return isSigned ? typeCache.SInt64Ty : typeCache.UInt64Ty;
    default:
      llvm_unreachable("NYI");
    }
  }

  cir::IntType getTruncatedIntTy(cir::IntType ty, bool isSigned) {
    switch (ty.getWidth()) {
    case 16:
      return isSigned ? typeCache.SInt8Ty : typeCache.UInt8Ty;
    case 32:
      return isSigned ? typeCache.SInt16Ty : typeCache.UInt16Ty;
    case 64:
      return isSigned ? typeCache.SInt32Ty : typeCache.UInt32Ty;
    default:
      llvm_unreachable("NYI");
    }
  }

  cir::VectorType
  getExtendedOrTruncatedElementVectorType(cir::VectorType vt, bool isExtended,
                                          bool isSigned = false) {
    auto elementTy = mlir::dyn_cast_or_null<cir::IntType>(vt.getEltType());
    assert(elementTy && "expected int vector");
    return cir::VectorType::get(getContext(),
                                isExtended
                                    ? getExtendedIntTy(elementTy, isSigned)
                                    : getTruncatedIntTy(elementTy, isSigned),
                                vt.getSize());
  }

  cir::LongDoubleType getLongDoubleTy(const llvm::fltSemantics &format) const {
    if (&format == &llvm::APFloat::IEEEdouble())
      return cir::LongDoubleType::get(getContext(), typeCache.DoubleTy);
    if (&format == &llvm::APFloat::x87DoubleExtended())
      return cir::LongDoubleType::get(getContext(), typeCache.FP80Ty);
    if (&format == &llvm::APFloat::IEEEquad())
      return cir::LongDoubleType::get(getContext(), typeCache.FP128Ty);
    if (&format == &llvm::APFloat::PPCDoubleDouble())
      llvm_unreachable("NYI");

    llvm_unreachable("unsupported long double format");
  }

  mlir::Type getVirtualFnPtrType(bool isVarArg = false) {
    // FIXME: replay LLVM codegen for now, perhaps add a vtable ptr special
    // type so it's a bit more clear and C++ idiomatic.
    auto fnTy = cir::FuncType::get({}, getUInt32Ty(), isVarArg);
    assert(!cir::MissingFeatures::isVarArg());
    return getPointerTo(getPointerTo(fnTy));
  }

  cir::FuncType getFuncType(llvm::ArrayRef<mlir::Type> params, mlir::Type retTy,
                            bool isVarArg = false) {
    return cir::FuncType::get(params, retTy, isVarArg);
  }

  // Fetch the type representing a pointer to unsigned int values.
  cir::PointerType getUInt8PtrTy(unsigned AddrSpace = 0) {
    return typeCache.UInt8PtrTy;
  }
  cir::PointerType getUInt32PtrTy(unsigned AddrSpace = 0) {
    return cir::PointerType::get(getContext(), typeCache.UInt32Ty);
  }

  /// Get a CIR anonymous struct type.
  cir::StructType getAnonStructTy(llvm::ArrayRef<mlir::Type> members,
                                  bool packed = false, bool padded = false,
                                  const clang::RecordDecl *ast = nullptr) {
    cir::ASTRecordDeclAttr astAttr = nullptr;
    auto kind = cir::StructType::RecordKind::Struct;
    if (ast) {
      astAttr = getAttr<cir::ASTRecordDeclAttr>(ast);
      kind = getRecordKind(ast->getTagKind());
    }
    return getType<cir::StructType>(members, packed, padded, kind, astAttr);
  }

  /// Get a CIR record kind from a AST declaration tag.
  cir::StructType::RecordKind getRecordKind(const clang::TagTypeKind kind) {
    switch (kind) {
    case clang::TagTypeKind::Struct:
      return cir::StructType::Struct;
    case clang::TagTypeKind::Union:
      return cir::StructType::Union;
    case clang::TagTypeKind::Class:
      return cir::StructType::Class;
    case clang::TagTypeKind::Interface:
      llvm_unreachable("interface records are NYI");
    case clang::TagTypeKind::Enum:
      llvm_unreachable("enum records are NYI");
    }
  }

  /// Get a incomplete CIR struct type.
  cir::StructType getIncompleteStructTy(llvm::StringRef name,
                                        const clang::RecordDecl *ast) {
    const auto nameAttr = getStringAttr(name);
    auto kind = cir::StructType::RecordKind::Struct;
    if (ast)
      kind = getRecordKind(ast->getTagKind());
    return getType<cir::StructType>(nameAttr, kind);
  }

  /// Get a CIR named struct type.
  ///
  /// If a struct already exists and is complete, but the client tries to fetch
  /// it with a different set of attributes, this method will crash.
  cir::StructType getCompleteStructTy(llvm::ArrayRef<mlir::Type> members,
                                      llvm::StringRef name, bool packed,
                                      bool padded,
                                      const clang::RecordDecl *ast) {
    const auto nameAttr = getStringAttr(name);
    cir::ASTRecordDeclAttr astAttr = nullptr;
    auto kind = cir::StructType::RecordKind::Struct;
    if (ast) {
      astAttr = getAttr<cir::ASTRecordDeclAttr>(ast);
      kind = getRecordKind(ast->getTagKind());
    }

    // Create or get the struct.
    auto type = getType<cir::StructType>(members, nameAttr, packed, padded,
                                         kind, astAttr);

    // Complete an incomplete struct or ensure the existing complete struct
    // matches the requested attributes.
    type.complete(members, packed, padded, astAttr);

    return type;
  }

  cir::StructType
  getCompleteStructType(mlir::ArrayAttr fields, bool packed = false,
                        bool padded = false, llvm::StringRef name = "",
                        const clang::RecordDecl *ast = nullptr) {
    llvm::SmallVector<mlir::Type, 8> members;
    for (auto &attr : fields) {
      const auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attr);
      members.push_back(typedAttr.getType());
    }

    if (name.empty())
      return getAnonStructTy(members, packed, padded, ast);
    else
      return getCompleteStructTy(members, name, packed, padded, ast);
  }

  cir::ArrayType getArrayType(mlir::Type eltType, unsigned size) {
    return cir::ArrayType::get(getContext(), eltType, size);
  }

  bool isSized(mlir::Type ty) {
    if (mlir::isa<cir::PointerType, cir::StructType, cir::ArrayType,
                  cir::BoolType, cir::IntType, cir::CIRFPTypeInterface>(ty))
      return true;
    if (mlir::isa<cir::VectorType>(ty)) {
      return isSized(mlir::cast<cir::VectorType>(ty).getEltType());
    }
    assert(0 && "Unimplemented size for type");
    return false;
  }

  //
  // Constant creation helpers
  // -------------------------
  //
  cir::ConstantOp getUInt8(uint8_t c, mlir::Location loc) {
    auto uInt8Ty = getUInt8Ty();
    return create<cir::ConstantOp>(loc, uInt8Ty, cir::IntAttr::get(uInt8Ty, c));
  }
  cir::ConstantOp getSInt32(int32_t c, mlir::Location loc) {
    auto sInt32Ty = getSInt32Ty();
    return create<cir::ConstantOp>(loc, sInt32Ty,
                                   cir::IntAttr::get(sInt32Ty, c));
  }
  cir::ConstantOp getUInt32(uint32_t C, mlir::Location loc) {
    auto uInt32Ty = getUInt32Ty();
    return create<cir::ConstantOp>(loc, uInt32Ty,
                                   cir::IntAttr::get(uInt32Ty, C));
  }
  cir::ConstantOp getSInt64(uint64_t C, mlir::Location loc) {
    auto sInt64Ty = getSInt64Ty();
    return create<cir::ConstantOp>(loc, sInt64Ty,
                                   cir::IntAttr::get(sInt64Ty, C));
  }
  cir::ConstantOp getUInt64(uint64_t C, mlir::Location loc) {
    auto uInt64Ty = getUInt64Ty();
    return create<cir::ConstantOp>(loc, uInt64Ty,
                                   cir::IntAttr::get(uInt64Ty, C));
  }

  cir::ConstantOp getConstInt(mlir::Location loc, llvm::APSInt intVal);

  cir::ConstantOp getConstInt(mlir::Location loc, llvm::APInt intVal);

  cir::ConstantOp getConstInt(mlir::Location loc, mlir::Type t, uint64_t C);

  cir::ConstantOp getConstFP(mlir::Location loc, mlir::Type t,
                             llvm::APFloat fpVal) {
    assert((mlir::isa<cir::SingleType, cir::DoubleType>(t)) &&
           "expected cir::SingleType or cir::DoubleType");
    return create<cir::ConstantOp>(loc, t, getAttr<cir::FPAttr>(t, fpVal));
  }

  cir::IsFPClassOp createIsFPClass(mlir::Location loc, mlir::Value src,
                                   unsigned flags) {
    return create<cir::IsFPClassOp>(loc, src, flags);
  }

  /// Create constant nullptr for pointer-to-data-member type ty.
  cir::ConstantOp getNullDataMemberPtr(cir::DataMemberType ty,
                                       mlir::Location loc) {
    return create<cir::ConstantOp>(loc, ty, getNullDataMemberAttr(ty));
  }

  cir::ConstantOp getNullMethodPtr(cir::MethodType ty, mlir::Location loc) {
    return create<cir::ConstantOp>(loc, ty, getNullMethodAttr(ty));
  }

  cir::ConstantOp getZero(mlir::Location loc, mlir::Type ty) {
    // TODO: dispatch creation for primitive types.
    assert((mlir::isa<cir::StructType>(ty) || mlir::isa<cir::ArrayType>(ty) ||
            mlir::isa<cir::VectorType>(ty)) &&
           "NYI for other types");
    return create<cir::ConstantOp>(loc, ty, getZeroAttr(ty));
  }

  //
  // Operation creation helpers
  // --------------------------
  //

  /// Create a break operation.
  cir::BreakOp createBreak(mlir::Location loc) {
    return create<cir::BreakOp>(loc);
  }

  /// Create a continue operation.
  cir::ContinueOp createContinue(mlir::Location loc) {
    return create<cir::ContinueOp>(loc);
  }

  cir::MemCpyOp createMemCpy(mlir::Location loc, mlir::Value dst,
                             mlir::Value src, mlir::Value len) {
    return create<cir::MemCpyOp>(loc, dst, src, len);
  }

  cir::MemMoveOp createMemMove(mlir::Location loc, mlir::Value dst,
                               mlir::Value src, mlir::Value len) {
    return create<cir::MemMoveOp>(loc, dst, src, len);
  }

  cir::MemSetOp createMemSet(mlir::Location loc, mlir::Value dst,
                             mlir::Value val, mlir::Value len) {
    val = createIntCast(val, cir::IntType::get(getContext(), 32, true));
    return create<cir::MemSetOp>(loc, dst, val, len);
  }

  cir::MemSetInlineOp createMemSetInline(mlir::Location loc, mlir::Value dst,
                                         mlir::Value val,
                                         mlir::IntegerAttr len) {
    val = createIntCast(val, cir::IntType::get(getContext(), 32, true));
    return create<cir::MemSetInlineOp>(loc, dst, val, len);
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
    if (getIsFPConstrained())
      llvm_unreachable("constrainedfp NYI");

    return create<cir::CastOp>(v.getLoc(), destType, cir::CastKind::floating,
                               v);
  }

  mlir::Value createFSub(mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!cir::MissingFeatures::foldBinOpFMF());
    return create<cir::BinOp>(lhs.getLoc(), cir::BinOpKind::Sub, lhs, rhs);
  }

  mlir::Value createFAdd(mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!cir::MissingFeatures::foldBinOpFMF());
    return create<cir::BinOp>(lhs.getLoc(), cir::BinOpKind::Add, lhs, rhs);
  }
  mlir::Value createFMul(mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!cir::MissingFeatures::foldBinOpFMF());
    return create<cir::BinOp>(lhs.getLoc(), cir::BinOpKind::Mul, lhs, rhs);
  }
  mlir::Value createFDiv(mlir::Value lhs, mlir::Value rhs) {
    assert(!cir::MissingFeatures::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!cir::MissingFeatures::foldBinOpFMF());
    return create<cir::BinOp>(lhs.getLoc(), cir::BinOpKind::Div, lhs, rhs);
  }

  mlir::Value createDynCast(mlir::Location loc, mlir::Value src,
                            cir::PointerType destType, bool isRefCast,
                            cir::DynamicCastInfoAttr info) {
    auto castKind =
        isRefCast ? cir::DynamicCastKind::ref : cir::DynamicCastKind::ptr;
    return create<cir::DynamicCastOp>(loc, destType, castKind, src, info,
                                      /*relative_layout=*/false);
  }

  mlir::Value createDynCastToVoid(mlir::Location loc, mlir::Value src,
                                  bool vtableUseRelativeLayout) {
    // TODO(cir): consider address space here.
    assert(!cir::MissingFeatures::addressSpace());
    auto destTy = getVoidPtrTy();
    return create<cir::DynamicCastOp>(loc, destTy, cir::DynamicCastKind::ptr,
                                      src, cir::DynamicCastInfoAttr{},
                                      vtableUseRelativeLayout);
  }

  Address createBaseClassAddr(mlir::Location loc, Address addr,
                              mlir::Type destType, unsigned offset,
                              bool assumeNotNull) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    auto baseAddr = create<cir::BaseClassAddrOp>(
        loc, ptrTy, addr.getPointer(), mlir::APInt(64, offset), assumeNotNull);
    return Address(baseAddr, ptrTy, addr.getAlignment());
  }

  Address createDerivedClassAddr(mlir::Location loc, Address addr,
                                 mlir::Type destType, unsigned offset,
                                 bool assumeNotNull) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    auto derivedAddr = create<cir::DerivedClassAddrOp>(
        loc, ptrTy, addr.getPointer(), mlir::APInt(64, offset), assumeNotNull);
    return Address(derivedAddr, ptrTy, addr.getAlignment());
  }

  mlir::Value createVTTAddrPoint(mlir::Location loc, mlir::Type retTy,
                                 mlir::Value addr, uint64_t offset) {
    return create<cir::VTTAddrPointOp>(loc, retTy, mlir::FlatSymbolRefAttr{},
                                       addr, offset);
  }

  mlir::Value createVTTAddrPoint(mlir::Location loc, mlir::Type retTy,
                                 mlir::FlatSymbolRefAttr sym, uint64_t offset) {
    return create<cir::VTTAddrPointOp>(loc, retTy, sym, mlir::Value{}, offset);
  }

  // FIXME(cir): CIRGenBuilder class should have an attribute with a reference
  // to the module so that we don't have search for it or pass it around.
  // FIXME(cir): Track a list of globals, or at least the last one inserted, so
  // that we can insert globals in the same order they are defined by CIRGen.

  [[nodiscard]] cir::GlobalOp
  createGlobal(mlir::ModuleOp module, mlir::Location loc, mlir::StringRef name,
               mlir::Type type, bool isConst, cir::GlobalLinkageKind linkage,
               cir::AddressSpaceAttr addrSpace = {}) {
    mlir::OpBuilder::InsertionGuard guard(*this);
    setInsertionPointToStart(module.getBody());
    return create<cir::GlobalOp>(loc, name, type, isConst, linkage, addrSpace);
  }

  /// Creates a versioned global variable. If the symbol is already taken, an ID
  /// will be appended to the symbol. The returned global must always be queried
  /// for its name so it can be referenced correctly.
  [[nodiscard]] cir::GlobalOp
  createVersionedGlobal(mlir::ModuleOp module, mlir::Location loc,
                        mlir::StringRef name, mlir::Type type, bool isConst,
                        cir::GlobalLinkageKind linkage,
                        cir::AddressSpaceAttr addrSpace = {}) {
    // Create a unique name if the given name is already taken.
    std::string uniqueName;
    if (unsigned version = GlobalsVersioning[name.str()]++)
      uniqueName = name.str() + "." + std::to_string(version);
    else
      uniqueName = name.str();

    return createGlobal(module, loc, uniqueName, type, isConst, linkage,
                        addrSpace);
  }

  mlir::Value createGetBitfield(mlir::Location loc, mlir::Type resultType,
                                mlir::Value addr, mlir::Type storageType,
                                const CIRGenBitFieldInfo &info,
                                bool isLvalueVolatile, bool useVolatile) {
    auto offset = useVolatile ? info.VolatileOffset : info.Offset;
    return create<cir::GetBitfieldOp>(loc, resultType, addr, storageType,
                                      info.Name, info.Size, offset,
                                      info.IsSigned, isLvalueVolatile);
  }

  mlir::Value createSetBitfield(mlir::Location loc, mlir::Type resultType,
                                mlir::Value dstAddr, mlir::Type storageType,
                                mlir::Value src, const CIRGenBitFieldInfo &info,
                                bool isLvalueVolatile, bool useVolatile) {
    auto offset = useVolatile ? info.VolatileOffset : info.Offset;
    return create<cir::SetBitfieldOp>(loc, resultType, dstAddr, storageType,
                                      src, info.Name, info.Size, offset,
                                      info.IsSigned, isLvalueVolatile);
  }

  /// Create a pointer to a record member.
  mlir::Value createGetMember(mlir::Location loc, mlir::Type result,
                              mlir::Value base, llvm::StringRef name,
                              unsigned index) {
    return create<cir::GetMemberOp>(loc, result, base, name, index);
  }

  /// Create a cir.complex.real_ptr operation that derives a pointer to the real
  /// part of the complex value pointed to by the specified pointer value.
  mlir::Value createRealPtr(mlir::Location loc, mlir::Value value) {
    auto srcPtrTy = mlir::cast<cir::PointerType>(value.getType());
    auto srcComplexTy = mlir::cast<cir::ComplexType>(srcPtrTy.getPointee());
    return create<cir::ComplexRealPtrOp>(
        loc, getPointerTo(srcComplexTy.getElementTy()), value);
  }

  Address createRealPtr(mlir::Location loc, Address addr) {
    return Address{createRealPtr(loc, addr.getPointer()), addr.getAlignment()};
  }

  /// Create a cir.complex.imag_ptr operation that derives a pointer to the
  /// imaginary part of the complex value pointed to by the specified pointer
  /// value.
  mlir::Value createImagPtr(mlir::Location loc, mlir::Value value) {
    auto srcPtrTy = mlir::cast<cir::PointerType>(value.getType());
    auto srcComplexTy = mlir::cast<cir::ComplexType>(srcPtrTy.getPointee());
    return create<cir::ComplexImagPtrOp>(
        loc, getPointerTo(srcComplexTy.getElementTy()), value);
  }

  Address createImagPtr(mlir::Location loc, Address addr) {
    return Address{createImagPtr(loc, addr.getPointer()), addr.getAlignment()};
  }

  /// Cast the element type of the given address to a different type,
  /// preserving information like the alignment.
  Address createElementBitCast(mlir::Location loc, Address addr,
                               mlir::Type destType) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    return Address(createBitcast(loc, addr.getPointer(), ptrTy), destType,
                   addr.getAlignment());
  }

  cir::LoadOp createLoad(mlir::Location loc, Address addr,
                         bool isVolatile = false) {
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(addr.getPointer().getType());
    if (addr.getElementType() != ptrTy.getPointee())
      addr = addr.withPointer(
          createPtrBitcast(addr.getPointer(), addr.getElementType()));

    return create<cir::LoadOp>(
        loc, addr.getElementType(), addr.getPointer(), /*isDeref=*/false,
        /*is_volatile=*/isVolatile, /*alignment=*/mlir::IntegerAttr{},
        /*mem_order=*/cir::MemOrderAttr{}, /*tbaa=*/cir::TBAAAttr{});
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Type ty,
                                mlir::Value ptr, llvm::MaybeAlign align,
                                bool isVolatile) {
    if (ty != mlir::cast<cir::PointerType>(ptr.getType()).getPointee())
      ptr = createPtrBitcast(ptr, ty);
    uint64_t alignment = align ? align->value() : 0;
    return CIRBaseBuilderTy::createLoad(loc, ptr, isVolatile, alignment);
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Type ty,
                                mlir::Value ptr, llvm::MaybeAlign align) {
    // TODO: make sure callsites shouldn't be really passing volatile.
    assert(!cir::MissingFeatures::volatileLoadOrStore());
    return createAlignedLoad(loc, ty, ptr, align, /*isVolatile=*/false);
  }

  mlir::Value
  createAlignedLoad(mlir::Location loc, mlir::Type ty, mlir::Value addr,
                    clang::CharUnits align = clang::CharUnits::One()) {
    return createAlignedLoad(loc, ty, addr, align.getAsAlign());
  }

  cir::StoreOp createStore(mlir::Location loc, mlir::Value val, Address dst,
                           bool _volatile = false,
                           ::mlir::IntegerAttr align = {},
                           cir::MemOrderAttr order = {}) {
    return CIRBaseBuilderTy::createStore(loc, val, dst.getPointer(), _volatile,
                                         align, order);
  }

  cir::StoreOp createFlagStore(mlir::Location loc, bool val, mlir::Value dst) {
    auto flag = getBool(val, loc);
    return CIRBaseBuilderTy::createStore(loc, flag, dst);
  }

  cir::VecShuffleOp
  createVecShuffle(mlir::Location loc, mlir::Value vec1, mlir::Value vec2,
                   llvm::ArrayRef<mlir::Attribute> maskAttrs) {
    auto vecType = mlir::cast<cir::VectorType>(vec1.getType());
    auto resultTy = cir::VectorType::get(getContext(), vecType.getEltType(),
                                         maskAttrs.size());
    return CIRBaseBuilderTy::create<cir::VecShuffleOp>(
        loc, resultTy, vec1, vec2, getArrayAttr(maskAttrs));
  }

  cir::VecShuffleOp createVecShuffle(mlir::Location loc, mlir::Value vec1,
                                     mlir::Value vec2,
                                     llvm::ArrayRef<int64_t> mask) {
    llvm::SmallVector<mlir::Attribute, 4> maskAttrs;
    for (int32_t idx : mask) {
      maskAttrs.push_back(cir::IntAttr::get(getSInt32Ty(), idx));
    }

    return createVecShuffle(loc, vec1, vec2, maskAttrs);
  }

  cir::VecShuffleOp createVecShuffle(mlir::Location loc, mlir::Value vec1,
                                     llvm::ArrayRef<int64_t> mask) {
    // FIXME(cir): Support use cir.vec.shuffle with single vec
    // Workaround: pass Vec as both vec1 and vec2
    return createVecShuffle(loc, vec1, vec1, mask);
  }

  cir::StoreOp
  createAlignedStore(mlir::Location loc, mlir::Value val, mlir::Value dst,
                     clang::CharUnits align = clang::CharUnits::One(),
                     bool _volatile = false, cir::MemOrderAttr order = {}) {
    llvm::MaybeAlign mayAlign = align.getAsAlign();
    mlir::IntegerAttr alignAttr;
    if (mayAlign) {
      uint64_t alignment = mayAlign ? mayAlign->value() : 0;
      alignAttr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(dst.getContext(), 64), alignment);
    }
    return CIRBaseBuilderTy::createStore(loc, val, dst, _volatile, alignAttr,
                                         order);
  }

  // Convert byte offset to sequence of high-level indices suitable for
  // GlobalViewAttr. Ideally we shouldn't deal with low-level offsets at all
  // but currently some parts of Clang AST, which we don't want to touch just
  // yet, return them.
  void computeGlobalViewIndicesFromFlatOffset(
      int64_t Offset, mlir::Type Ty, cir::CIRDataLayout Layout,
      llvm::SmallVectorImpl<int64_t> &Indices);

  // Convert high-level indices (e.g. from GlobalViewAttr) to byte offset
  uint64_t computeOffsetFromGlobalViewIndices(const cir::CIRDataLayout &layout,
                                              mlir::Type t,
                                              llvm::ArrayRef<int64_t> indexes);

  cir::StackSaveOp createStackSave(mlir::Location loc, mlir::Type ty) {
    return create<cir::StackSaveOp>(loc, ty);
  }

  cir::StackRestoreOp createStackRestore(mlir::Location loc, mlir::Value v) {
    return create<cir::StackRestoreOp>(loc, v);
  }

  // TODO(cir): Change this to hoist alloca to the parent *scope* instead.
  /// Move alloca operation to the parent region.
  void hoistAllocaToParentRegion(cir::AllocaOp alloca) {
    auto &block = alloca->getParentOp()->getParentRegion()->front();
    const auto allocas = block.getOps<cir::AllocaOp>();
    if (allocas.empty()) {
      alloca->moveBefore(&block, block.begin());
    } else {
      alloca->moveAfter(*std::prev(allocas.end()));
    }
  }

  cir::CmpThreeWayOp createThreeWayCmpStrong(mlir::Location loc,
                                             mlir::Value lhs, mlir::Value rhs,
                                             const llvm::APSInt &ltRes,
                                             const llvm::APSInt &eqRes,
                                             const llvm::APSInt &gtRes) {
    assert(ltRes.getBitWidth() == eqRes.getBitWidth() &&
           ltRes.getBitWidth() == gtRes.getBitWidth() &&
           "the three comparison results must have the same bit width");
    auto cmpResultTy = getSIntNTy(ltRes.getBitWidth());
    auto infoAttr = getCmpThreeWayInfoStrongOrdering(ltRes, eqRes, gtRes);
    return create<cir::CmpThreeWayOp>(loc, cmpResultTy, lhs, rhs, infoAttr);
  }

  cir::CmpThreeWayOp
  createThreeWayCmpPartial(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                           const llvm::APSInt &ltRes, const llvm::APSInt &eqRes,
                           const llvm::APSInt &gtRes,
                           const llvm::APSInt &unorderedRes) {
    assert(ltRes.getBitWidth() == eqRes.getBitWidth() &&
           ltRes.getBitWidth() == gtRes.getBitWidth() &&
           ltRes.getBitWidth() == unorderedRes.getBitWidth() &&
           "the four comparison results must have the same bit width");
    auto cmpResultTy = getSIntNTy(ltRes.getBitWidth());
    auto infoAttr =
        getCmpThreeWayInfoPartialOrdering(ltRes, eqRes, gtRes, unorderedRes);
    return create<cir::CmpThreeWayOp>(loc, cmpResultTy, lhs, rhs, infoAttr);
  }

  cir::GetRuntimeMemberOp createGetIndirectMember(mlir::Location loc,
                                                  mlir::Value objectPtr,
                                                  mlir::Value memberPtr) {
    auto memberPtrTy = mlir::cast<cir::DataMemberType>(memberPtr.getType());

    // TODO(cir): consider address space.
    assert(!cir::MissingFeatures::addressSpace());
    auto resultTy = getPointerTo(memberPtrTy.getMemberTy());

    return create<cir::GetRuntimeMemberOp>(loc, resultTy, objectPtr, memberPtr);
  }

  /// Create a cir.ptr_stride operation to get access to an array element.
  /// idx is the index of the element to access, shouldDecay is true if the
  /// result should decay to a pointer to the element type.
  mlir::Value getArrayElement(mlir::Location arrayLocBegin,
                              mlir::Location arrayLocEnd, mlir::Value arrayPtr,
                              mlir::Type eltTy, mlir::Value idx,
                              bool shouldDecay);

  /// Returns a decayed pointer to the first element of the array
  /// pointed to by arrayPtr.
  mlir::Value maybeBuildArrayDecay(mlir::Location loc, mlir::Value arrayPtr,
                                   mlir::Type eltTy);
};

} // namespace clang::CIRGen
#endif
