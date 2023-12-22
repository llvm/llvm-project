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
#include "CIRDataLayout.h"
#include "CIRGenRecordLayout.h"
#include "CIRGenTypeCache.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
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
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <optional>
#include <string>

namespace cir {

class CIRGenFunction;

class CIRGenBuilderTy : public CIRBaseBuilderTy {
  const CIRGenTypeCache &typeCache;
  bool IsFPConstrained = false;
  fp::ExceptionBehavior DefaultConstrainedExcept = fp::ebStrict;
  llvm::RoundingMode DefaultConstrainedRounding = llvm::RoundingMode::Dynamic;

  llvm::StringMap<unsigned> GlobalsVersioning;
  llvm::StringSet<> anonRecordNames;

public:
  CIRGenBuilderTy(mlir::MLIRContext &C, const CIRGenTypeCache &tc)
      : CIRBaseBuilderTy(C), typeCache(tc) {}

  std::string getUniqueAnonRecordName() {
    std::string name = "anon." + std::to_string(anonRecordNames.size());
    anonRecordNames.insert(name);
    return name;
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
  void setDefaultConstrainedExcept(fp::ExceptionBehavior NewExcept) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> ExceptStr =
        convertExceptionBehaviorToStr(NewExcept);
    assert(ExceptStr && "Garbage strict exception behavior!");
#endif
    DefaultConstrainedExcept = NewExcept;
  }

  /// Set the rounding mode handling to be used with constrained floating point
  void setDefaultConstrainedRounding(llvm::RoundingMode NewRounding) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> RoundingStr =
        convertRoundingModeToStr(NewRounding);
    assert(RoundingStr && "Garbage strict rounding mode!");
#endif
    DefaultConstrainedRounding = NewRounding;
  }

  /// Get the exception handling used with constrained floating point
  fp::ExceptionBehavior getDefaultConstrainedExcept() {
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
  mlir::cir::GlobalViewAttr getGlobalViewAttr(mlir::cir::GlobalOp globalOp,
                                              mlir::ArrayAttr indices = {}) {
    auto type = getPointerTo(globalOp.getSymType());
    return getGlobalViewAttr(type, globalOp, indices);
  }

  /// Get constant address of a global variable as an MLIR attribute.
  mlir::cir::GlobalViewAttr getGlobalViewAttr(mlir::cir::PointerType type,
                                              mlir::cir::GlobalOp globalOp,
                                              mlir::ArrayAttr indices = {}) {
    auto symbol = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
    return mlir::cir::GlobalViewAttr::get(type, symbol, indices);
  }

  mlir::TypedAttr getZeroAttr(mlir::Type t) {
    return mlir::cir::ZeroAttr::get(getContext(), t);
  }

  mlir::cir::BoolAttr getCIRBoolAttr(bool state) {
    return mlir::cir::BoolAttr::get(getContext(), getBoolTy(), state);
  }

  mlir::TypedAttr getConstPtrAttr(mlir::Type t, uint64_t v) {
    assert(t.isa<mlir::cir::PointerType>() && "expected cir.ptr");
    return mlir::cir::ConstPtrAttr::get(getContext(), t, v);
  }

  mlir::cir::ConstArrayAttr getString(llvm::StringRef str, mlir::Type eltTy,
                                      unsigned size = 0) {
    unsigned finalSize = size ? size : str.size();
    auto arrayTy = mlir::cir::ArrayType::get(getContext(), eltTy, finalSize);
    return getConstArray(mlir::StringAttr::get(str, arrayTy), arrayTy);
  }

  mlir::cir::ConstArrayAttr getConstArray(mlir::Attribute attrs,
                                          mlir::cir::ArrayType arrayTy) {
    return mlir::cir::ConstArrayAttr::get(arrayTy, attrs);
  }

  mlir::Attribute getConstStructOrZeroAttr(mlir::ArrayAttr arrayAttr,
                                           bool packed = false,
                                           mlir::Type type = {}) {
    llvm::SmallVector<mlir::Type, 8> members;
    auto structTy = type.dyn_cast<mlir::cir::StructType>();
    assert(structTy && "expected cir.struct");
    assert(!packed && "unpacked struct is NYI");

    // Collect members and check if they are all zero.
    bool isZero = true;
    for (auto &attr : arrayAttr) {
      const auto typedAttr = attr.dyn_cast<mlir::TypedAttr>();
      members.push_back(typedAttr.getType());
      isZero &= isNullValue(typedAttr);
    }

    // Struct type not specified: create anon struct type from members.
    if (!structTy)
      structTy = getType<mlir::cir::StructType>(members, packed,
                                                mlir::cir::StructType::Struct,
                                                /*ast=*/nullptr);

    // Return zero or anonymous constant struct.
    if (isZero)
      return mlir::cir::ZeroAttr::get(getContext(), structTy);
    return mlir::cir::ConstStructAttr::get(structTy, arrayAttr);
  }

  mlir::cir::ConstStructAttr getAnonConstStruct(mlir::ArrayAttr arrayAttr,
                                                bool packed = false,
                                                mlir::Type ty = {}) {
    assert(!packed && "NYI");
    llvm::SmallVector<mlir::Type, 4> members;
    for (auto &f : arrayAttr) {
      auto ta = f.dyn_cast<mlir::TypedAttr>();
      assert(ta && "expected typed attribute member");
      members.push_back(ta.getType());
    }

    if (!ty)
      ty = getAnonStructTy(members, packed);

    auto sTy = ty.dyn_cast<mlir::cir::StructType>();
    assert(sTy && "expected struct type");
    return mlir::cir::ConstStructAttr::get(sTy, arrayAttr);
  }

  mlir::cir::TypeInfoAttr getTypeInfo(mlir::ArrayAttr fieldsAttr) {
    auto anonStruct = getAnonConstStruct(fieldsAttr);
    return mlir::cir::TypeInfoAttr::get(anonStruct.getType(), fieldsAttr);
  }

  mlir::TypedAttr getZeroInitAttr(mlir::Type ty) {
    if (ty.isa<mlir::cir::IntType>())
      return mlir::cir::IntAttr::get(ty, 0);
    if (ty.isa<mlir::FloatType>())
      return mlir::FloatAttr::get(ty, 0.0);
    if (auto arrTy = ty.dyn_cast<mlir::cir::ArrayType>())
      return getZeroAttr(arrTy);
    if (auto ptrTy = ty.dyn_cast<mlir::cir::PointerType>())
      return getConstPtrAttr(ptrTy, 0);
    if (auto structTy = ty.dyn_cast<mlir::cir::StructType>())
      return getZeroAttr(structTy);
    llvm_unreachable("Zero initializer for given type is NYI");
  }

  // TODO(cir): Once we have CIR float types, replace this by something like a
  // NullableValueInterface to allow for type-independent queries.
  bool isNullValue(mlir::Attribute attr) const {
    if (attr.isa<mlir::cir::ZeroAttr>())
      return true;
    if (const auto ptrVal = attr.dyn_cast<mlir::cir::ConstPtrAttr>())
      return ptrVal.isNullValue();

    if (attr.isa<mlir::cir::GlobalViewAttr>())
      return false;

    // TODO(cir): introduce char type in CIR and check for that instead.
    if (const auto intVal = attr.dyn_cast<mlir::cir::IntAttr>())
      return intVal.isNullValue();

    if (const auto fpVal = attr.dyn_cast<mlir::FloatAttr>()) {
      bool ignored;
      llvm::APFloat FV(+0.0);
      FV.convert(fpVal.getValue().getSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &ignored);
      return FV.bitwiseIsEqual(fpVal.getValue());
    }

    if (const auto structVal = attr.dyn_cast<mlir::cir::ConstStructAttr>()) {
      for (const auto elt : structVal.getMembers()) {
        // FIXME(cir): the struct's ID should not be considered a member.
        if (elt.isa<mlir::StringAttr>())
          continue;
        if (!isNullValue(elt))
          return false;
      }
      return true;
    }

    if (const auto arrayVal = attr.dyn_cast<mlir::cir::ConstArrayAttr>()) {
      if (arrayVal.getElts().isa<mlir::StringAttr>())
        return false;
      for (const auto elt : arrayVal.getElts().cast<mlir::ArrayAttr>()) {
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
  mlir::cir::IntType getUIntNTy(int N) {
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
      llvm_unreachable("Unknown bit-width");
    }
  }

  mlir::cir::IntType getSIntNTy(int N) {
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
      llvm_unreachable("Unknown bit-width");
    }
  }

  mlir::cir::VoidType getVoidTy() { return typeCache.VoidTy; }

  mlir::cir::IntType getSInt8Ty() { return typeCache.SInt8Ty; }
  mlir::cir::IntType getSInt16Ty() { return typeCache.SInt16Ty; }
  mlir::cir::IntType getSInt32Ty() { return typeCache.SInt32Ty; }
  mlir::cir::IntType getSInt64Ty() { return typeCache.SInt64Ty; }

  mlir::cir::IntType getUInt8Ty() { return typeCache.UInt8Ty; }
  mlir::cir::IntType getUInt16Ty() { return typeCache.UInt16Ty; }
  mlir::cir::IntType getUInt32Ty() { return typeCache.UInt32Ty; }
  mlir::cir::IntType getUInt64Ty() { return typeCache.UInt64Ty; }

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
  bool isInt(mlir::Type i) { return i.isa<mlir::cir::IntType>(); }

  mlir::FloatType getLongDouble80BitsTy() const {
    return typeCache.LongDouble80BitsTy;
  }

  /// Get the proper floating point type for the given semantics.
  mlir::FloatType getFloatTyForFormat(const llvm::fltSemantics &format,
                                      bool useNativeHalf) const {
    if (&format == &llvm::APFloat::IEEEhalf()) {
      llvm_unreachable("IEEEhalf float format is NYI");
    }

    if (&format == &llvm::APFloat::BFloat())
      llvm_unreachable("BFloat float format is NYI");
    if (&format == &llvm::APFloat::IEEEsingle())
      llvm_unreachable("IEEEsingle float format is NYI");
    if (&format == &llvm::APFloat::IEEEdouble())
      llvm_unreachable("IEEEdouble float format is NYI");
    if (&format == &llvm::APFloat::IEEEquad())
      llvm_unreachable("IEEEquad float format is NYI");
    if (&format == &llvm::APFloat::PPCDoubleDouble())
      llvm_unreachable("PPCDoubleDouble float format is NYI");
    if (&format == &llvm::APFloat::x87DoubleExtended())
      return getLongDouble80BitsTy();

    llvm_unreachable("Unknown float format!");
  }

  mlir::cir::BoolType getBoolTy() {
    return ::mlir::cir::BoolType::get(getContext());
  }
  mlir::Type getVirtualFnPtrType(bool isVarArg = false) {
    // FIXME: replay LLVM codegen for now, perhaps add a vtable ptr special
    // type so it's a bit more clear and C++ idiomatic.
    auto fnTy = mlir::cir::FuncType::get({}, getUInt32Ty(), isVarArg);
    assert(!UnimplementedFeature::isVarArg());
    return getPointerTo(getPointerTo(fnTy));
  }

  mlir::cir::FuncType getFuncType(llvm::ArrayRef<mlir::Type> params,
                                  mlir::Type retTy, bool isVarArg = false) {
    return mlir::cir::FuncType::get(params, retTy, isVarArg);
  }

  // Fetch the type representing a pointer to unsigned int values.
  mlir::cir::PointerType getUInt8PtrTy(unsigned AddrSpace = 0) {
    return typeCache.UInt8PtrTy;
  }
  mlir::cir::PointerType getUInt32PtrTy(unsigned AddrSpace = 0) {
    return mlir::cir::PointerType::get(getContext(), typeCache.UInt32Ty);
  }
  mlir::cir::PointerType getPointerTo(mlir::Type ty,
                                      unsigned addressSpace = 0) {
    assert(!UnimplementedFeature::addressSpace() && "NYI");
    return mlir::cir::PointerType::get(getContext(), ty);
  }

  /// Get a CIR anonymous struct type.
  mlir::cir::StructType
  getAnonStructTy(llvm::ArrayRef<mlir::Type> members, bool packed = false,
                  const clang::RecordDecl *ast = nullptr) {
    mlir::cir::ASTRecordDeclAttr astAttr = nullptr;
    auto kind = mlir::cir::StructType::RecordKind::Struct;
    if (ast) {
      astAttr = getAttr<mlir::cir::ASTRecordDeclAttr>(ast);
      kind = getRecordKind(ast->getTagKind());
    }
    return getType<mlir::cir::StructType>(members, packed, kind, astAttr);
  }

  /// Get a CIR record kind from a AST declaration tag.
  mlir::cir::StructType::RecordKind
  getRecordKind(const clang::TagTypeKind kind) {
    switch (kind) {
    case clang::TagTypeKind::Struct:
      return mlir::cir::StructType::Struct;
    case clang::TagTypeKind::Union:
      return mlir::cir::StructType::Union;
    case clang::TagTypeKind::Class:
      return mlir::cir::StructType::Class;
    case clang::TagTypeKind::Interface:
      llvm_unreachable("interface records are NYI");
    case clang::TagTypeKind::Enum:
      llvm_unreachable("enum records are NYI");
    }
  }

  /// Get a incomplete CIR struct type.
  mlir::cir::StructType getIncompleteStructTy(llvm::StringRef name,
                                              const clang::RecordDecl *ast) {
    const auto nameAttr = getStringAttr(name);
    auto kind = mlir::cir::StructType::RecordKind::Struct;
    if (ast)
      kind = getRecordKind(ast->getTagKind());
    return getType<mlir::cir::StructType>(nameAttr, kind);
  }

  /// Get a CIR named struct type.
  ///
  /// If a struct already exists and is complete, but the client tries to fetch
  /// it with a different set of attributes, this method will crash.
  mlir::cir::StructType getCompleteStructTy(llvm::ArrayRef<mlir::Type> members,
                                            llvm::StringRef name, bool packed,
                                            const clang::RecordDecl *ast) {
    const auto nameAttr = getStringAttr(name);
    mlir::cir::ASTRecordDeclAttr astAttr = nullptr;
    auto kind = mlir::cir::StructType::RecordKind::Struct;
    if (ast) {
      astAttr = getAttr<mlir::cir::ASTRecordDeclAttr>(ast);
      kind = getRecordKind(ast->getTagKind());
    }

    // Create or get the struct.
    auto type = getType<mlir::cir::StructType>(members, nameAttr, packed, kind,
                                               astAttr);

    // Complete an incomplete struct or ensure the existing complete struct
    // matches the requested attributes.
    type.complete(members, packed, astAttr);

    return type;
  }

  mlir::cir::ArrayType getArrayType(mlir::Type eltType, unsigned size) {
    return mlir::cir::ArrayType::get(getContext(), eltType, size);
  }

  bool isSized(mlir::Type ty) {
    if (ty.isIntOrFloat() ||
        ty.isa<mlir::cir::PointerType, mlir::cir::StructType,
               mlir::cir::ArrayType, mlir::cir::BoolType, mlir::cir::IntType>())
      return true;
    assert(0 && "Unimplemented size for type");
    return false;
  }

  //
  // Constant creation helpers
  // -------------------------
  //
  mlir::cir::ConstantOp getSInt32(uint32_t c, mlir::Location loc) {
    auto sInt32Ty = getSInt32Ty();
    return create<mlir::cir::ConstantOp>(loc, sInt32Ty,
                                         mlir::cir::IntAttr::get(sInt32Ty, c));
  }
  mlir::cir::ConstantOp getUInt32(uint32_t C, mlir::Location loc) {
    auto uInt32Ty = getUInt32Ty();
    return create<mlir::cir::ConstantOp>(loc, uInt32Ty,
                                         mlir::cir::IntAttr::get(uInt32Ty, C));
  }
  mlir::cir::ConstantOp getSInt64(uint64_t C, mlir::Location loc) {
    auto sInt64Ty = getSInt64Ty();
    return create<mlir::cir::ConstantOp>(loc, sInt64Ty,
                                         mlir::cir::IntAttr::get(sInt64Ty, C));
  }
  mlir::cir::ConstantOp getUInt64(uint64_t C, mlir::Location loc) {
    auto uInt64Ty = getUInt64Ty();
    return create<mlir::cir::ConstantOp>(loc, uInt64Ty,
                                         mlir::cir::IntAttr::get(uInt64Ty, C));
  }
  mlir::cir::ConstantOp getConstInt(mlir::Location loc, mlir::cir::IntType t,
                                    uint64_t C) {
    return create<mlir::cir::ConstantOp>(loc, t, mlir::cir::IntAttr::get(t, C));
  }
  mlir::cir::ConstantOp getConstInt(mlir::Location loc, llvm::APSInt intVal) {
    bool isSigned = intVal.isSigned();
    auto width = intVal.getBitWidth();
    mlir::cir::IntType t = isSigned ? getSIntNTy(width) : getUIntNTy(width);
    return getConstInt(
        loc, t, isSigned ? intVal.getSExtValue() : intVal.getZExtValue());
  }
  mlir::cir::ConstantOp getBool(bool state, mlir::Location loc) {
    return create<mlir::cir::ConstantOp>(loc, getBoolTy(),
                                         getCIRBoolAttr(state));
  }
  mlir::cir::ConstantOp getFalse(mlir::Location loc) {
    return getBool(false, loc);
  }
  mlir::cir::ConstantOp getTrue(mlir::Location loc) {
    return getBool(true, loc);
  }

  // Creates constant nullptr for pointer type ty.
  mlir::cir::ConstantOp getNullPtr(mlir::Type ty, mlir::Location loc) {
    return create<mlir::cir::ConstantOp>(loc, ty, getConstPtrAttr(ty, 0));
  }

  // Creates constant null value for integral type ty.
  mlir::cir::ConstantOp getNullValue(mlir::Type ty, mlir::Location loc) {
    if (ty.isa<mlir::cir::PointerType>())
      return getNullPtr(ty, loc);

    mlir::TypedAttr attr;
    if (ty.isa<mlir::cir::IntType>())
      attr = mlir::cir::IntAttr::get(ty, 0);
    else
      llvm_unreachable("NYI");

    return create<mlir::cir::ConstantOp>(loc, ty, attr);
  }

  mlir::cir::ConstantOp getZero(mlir::Location loc, mlir::Type ty) {
    // TODO: dispatch creation for primitive types.
    assert(ty.isa<mlir::cir::StructType>() && "NYI for other types");
    return create<mlir::cir::ConstantOp>(loc, ty, getZeroAttr(ty));
  }

  mlir::cir::ConstantOp getConstant(mlir::Location loc, mlir::TypedAttr attr) {
    return create<mlir::cir::ConstantOp>(loc, attr.getType(), attr);
  }

  //
  // Block handling helpers
  // ----------------------
  //
  OpBuilder::InsertPoint getBestAllocaInsertPoint(mlir::Block *block) {
    auto lastAlloca =
        std::find_if(block->rbegin(), block->rend(), [](mlir::Operation &op) {
          return mlir::isa<mlir::cir::AllocaOp>(&op);
        });

    if (lastAlloca != block->rend())
      return OpBuilder::InsertPoint(block,
                                    ++mlir::Block::iterator(&*lastAlloca));
    return OpBuilder::InsertPoint(block, block->begin());
  };

  //
  // Operation creation helpers
  // --------------------------
  //

  /// Create a copy with inferred length.
  mlir::cir::CopyOp createCopy(mlir::Value dst, mlir::Value src) {
    return create<mlir::cir::CopyOp>(dst.getLoc(), dst, src);
  }

  mlir::cir::MemCpyOp createMemCpy(mlir::Location loc, mlir::Value dst,
                                   mlir::Value src, mlir::Value len) {
    return create<mlir::cir::MemCpyOp>(loc, dst, src, len);
  }

  mlir::Value createNeg(mlir::Value value) {

    if (auto intTy = value.getType().dyn_cast<mlir::cir::IntType>()) {
      // Source is a unsigned integer: first cast it to signed.
      if (intTy.isUnsigned())
        value = createIntCast(value, getSIntNTy(intTy.getWidth()));
      return create<mlir::cir::UnaryOp>(value.getLoc(), value.getType(),
                                        mlir::cir::UnaryOpKind::Minus, value);
    }

    llvm_unreachable("negation for the given type is NYI");
  }

  // TODO: split this to createFPExt/createFPTrunc when we have dedicated cast
  // operations.
  mlir::Value createFloatingCast(mlir::Value v, mlir::Type destType) {
    if (getIsFPConstrained())
      llvm_unreachable("constrainedfp NYI");

    return create<mlir::cir::CastOp>(v.getLoc(), destType,
                                     mlir::cir::CastKind::floating, v);
  }

  mlir::Value createFSub(mlir::Value lhs, mlir::Value rhs) {
    assert(!UnimplementedFeature::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!UnimplementedFeature::foldBinOpFMF());
    return create<mlir::cir::BinOp>(lhs.getLoc(), mlir::cir::BinOpKind::Sub,
                                    lhs, rhs);
  }

  mlir::Value createPtrToBoolCast(mlir::Value v) {
    return create<mlir::cir::CastOp>(v.getLoc(), getBoolTy(),
                                     mlir::cir::CastKind::ptr_to_bool, v);
  }

  cir::Address createBaseClassAddr(mlir::Location loc, cir::Address addr,
                                   mlir::Type destType) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    auto baseAddr =
        create<mlir::cir::BaseClassAddrOp>(loc, ptrTy, addr.getPointer());

    return Address(baseAddr, ptrTy, addr.getAlignment());
  }

  // FIXME(cir): CIRGenBuilder class should have an attribute with a reference
  // to the module so that we don't have search for it or pass it around.
  // FIXME(cir): Track a list of globals, or at least the last one inserted, so
  // that we can insert globals in the same order they are defined by CIRGen.

  /// Creates a versioned global variable. If the symbol is already taken, an ID
  /// will be appended to the symbol. The returned global must always be queried
  /// for its name so it can be referenced correctly.
  [[nodiscard]] mlir::cir::GlobalOp
  createVersionedGlobal(mlir::ModuleOp module, mlir::Location loc,
                        mlir::StringRef name, mlir::Type type, bool isConst,
                        mlir::cir::GlobalLinkageKind linkage) {
    mlir::OpBuilder::InsertionGuard guard(*this);
    setInsertionPointToStart(module.getBody());

    // Create a unique name if the given name is already taken.
    std::string uniqueName;
    if (unsigned version = GlobalsVersioning[name.str()]++)
      uniqueName = name.str() + "." + std::to_string(version);
    else
      uniqueName = name.str();

    return create<mlir::cir::GlobalOp>(loc, uniqueName, type, isConst, linkage);
  }

  mlir::Value createGetGlobal(mlir::cir::GlobalOp global) {
    return create<mlir::cir::GetGlobalOp>(
        global.getLoc(), getPointerTo(global.getSymType()), global.getName());
  }

  mlir::Value createGetBitfield(mlir::Location loc, mlir::Type resultType,
                                mlir::Value addr, mlir::Type storageType,
                                const CIRGenBitFieldInfo &info,
                                bool useVolatile) {
    auto offset = useVolatile ? info.VolatileOffset : info.Offset;
    return create<mlir::cir::GetBitfieldOp>(loc, resultType, addr, storageType,
                                            info.Name, info.Size, offset,
                                            info.IsSigned);
  }

  mlir::Value createSetBitfield(mlir::Location loc, mlir::Type resultType,
                                mlir::Value dstAddr, mlir::Type storageType,
                                mlir::Value src, const CIRGenBitFieldInfo &info,
                                bool useVolatile) {
    auto offset = useVolatile ? info.VolatileOffset : info.Offset;
    return create<mlir::cir::SetBitfieldOp>(loc, resultType, dstAddr,
                                            storageType, src, info.Name,
                                            info.Size, offset, info.IsSigned);
  }

  /// Create a pointer to a record member.
  mlir::Value createGetMember(mlir::Location loc, mlir::Type result,
                              mlir::Value base, llvm::StringRef name,
                              unsigned index) {
    return create<mlir::cir::GetMemberOp>(loc, result, base, name, index);
  }

  /// Cast the element type of the given address to a different type,
  /// preserving information like the alignment.
  cir::Address createElementBitCast(mlir::Location loc, cir::Address addr,
                                    mlir::Type destType) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    return Address(createBitcast(loc, addr.getPointer(), ptrTy), destType,
                   addr.getAlignment());
  }

  mlir::Value createLoad(mlir::Location loc, Address addr) {
    return create<mlir::cir::LoadOp>(loc, addr.getElementType(),
                                     addr.getPointer());
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Type ty,
                                mlir::Value ptr,
                                [[maybe_unused]] llvm::MaybeAlign align,
                                [[maybe_unused]] bool isVolatile) {
    assert(!UnimplementedFeature::volatileLoadOrStore());
    assert(!UnimplementedFeature::alignedLoad());
    return create<mlir::cir::LoadOp>(loc, ty, ptr);
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Type ty,
                                mlir::Value ptr, llvm::MaybeAlign align) {
    return createAlignedLoad(loc, ty, ptr, align, /*isVolatile=*/false);
  }

  mlir::Value
  createAlignedLoad(mlir::Location loc, mlir::Type ty, mlir::Value addr,
                    clang::CharUnits align = clang::CharUnits::One()) {
    return createAlignedLoad(loc, ty, addr, align.getAsAlign());
  }

  mlir::cir::StoreOp createStore(mlir::Location loc, mlir::Value val,
                                 Address dst) {
    return create<mlir::cir::StoreOp>(loc, val, dst.getPointer());
  }

  mlir::cir::StoreOp createFlagStore(mlir::Location loc, bool val,
                                     mlir::Value dst) {
    auto flag = getBool(val, loc);
    return create<mlir::cir::StoreOp>(loc, flag, dst);
  }

  // Convert byte offset to sequence of high-level indices suitable for
  // GlobalViewAttr. Ideally we shouldn't deal with low-level offsets at all
  // but currently some parts of Clang AST, which we don't want to touch just
  // yet, return them.
  void computeGlobalViewIndicesFromFlatOffset(
      int64_t Offset, mlir::Type Ty, CIRDataLayout Layout,
      llvm::SmallVectorImpl<int64_t> &Indices) {
    if (!Offset)
      return;

    mlir::Type SubType;

    if (auto ArrayTy = Ty.dyn_cast<mlir::cir::ArrayType>()) {
      auto EltSize = Layout.getTypeAllocSize(ArrayTy.getEltType());
      Indices.push_back(Offset / EltSize);
      SubType = ArrayTy.getEltType();
      Offset %= EltSize;
    } else if (auto PtrTy = Ty.dyn_cast<mlir::cir::PointerType>()) {
      auto EltSize = Layout.getTypeAllocSize(PtrTy.getPointee());
      Indices.push_back(Offset / EltSize);
      SubType = PtrTy.getPointee();
      Offset %= EltSize;
    } else if (auto StructTy = Ty.dyn_cast<mlir::cir::StructType>()) {
      auto Elts = StructTy.getMembers();
      for (size_t I = 0; I < Elts.size(); ++I) {
        auto EltSize = Layout.getTypeAllocSize(Elts[I]);
        if (Offset < EltSize) {
          Indices.push_back(I);
          SubType = Elts[I];
          break;
        }
        Offset -= EltSize;
      }
    } else {
      llvm_unreachable("unexpected type");
    }

    assert(SubType);
    computeGlobalViewIndicesFromFlatOffset(Offset, SubType, Layout, Indices);
  }

  mlir::cir::StackSaveOp createStackSave(mlir::Location loc, mlir::Type ty) {
    return create<mlir::cir::StackSaveOp>(loc, ty);
  }

  mlir::cir::StackRestoreOp createStackRestore(mlir::Location loc, mlir::Value v) {
    return create<mlir::cir::StackRestoreOp>(loc, v);
  }

};

} // namespace cir
#endif
