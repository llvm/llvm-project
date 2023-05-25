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
#include "CIRGenTypeCache.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/FloatingPointMode.h"

namespace cir {

class CIRGenFunction;

class CIRGenBuilderTy : public mlir::OpBuilder {
  const CIRGenTypeCache &typeCache;
  bool IsFPConstrained = false;
  fp::ExceptionBehavior DefaultConstrainedExcept = fp::ebStrict;
  llvm::RoundingMode DefaultConstrainedRounding = llvm::RoundingMode::Dynamic;

public:
  CIRGenBuilderTy(mlir::MLIRContext &C, const CIRGenTypeCache &tc)
      : mlir::OpBuilder(&C), typeCache(tc) {}

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
  mlir::TypedAttr getZeroAttr(mlir::Type t) {
    return mlir::cir::ZeroAttr::get(getContext(), t);
  }

  mlir::cir::BoolAttr getCIRBoolAttr(bool state) {
    return mlir::cir::BoolAttr::get(getContext(), getBoolTy(), state);
  }

  mlir::TypedAttr getNullPtrAttr(mlir::Type t) {
    assert(t.isa<mlir::cir::PointerType>() && "expected cir.ptr");
    return mlir::cir::NullAttr::get(getContext(), t);
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
    auto *ctx = arrayAttr.getContext();
    if (!ty)
      ty = mlir::cir::StructType::get(ctx, members, mlir::StringAttr::get(ctx),
                                      /*body=*/true, packed,
                                      /*ast=*/std::nullopt);
    auto sTy = ty.dyn_cast<mlir::cir::StructType>();
    assert(sTy && "expected struct type");
    return mlir::cir::ConstStructAttr::get(sTy, arrayAttr);
  }

  mlir::cir::TypeInfoAttr getTypeInfo(mlir::ArrayAttr fieldsAttr) {
    auto anonStruct = getAnonConstStruct(fieldsAttr);
    return mlir::cir::TypeInfoAttr::get(anonStruct.getType(), anonStruct);
  }

  //
  // Type helpers
  // ------------
  //
  mlir::Type getSInt8Ty() { return typeCache.SInt8Ty; }
  mlir::Type getSInt16Ty() { return typeCache.SInt16Ty; }
  mlir::Type getSInt32Ty() { return typeCache.SInt32Ty; }
  mlir::Type getSInt64Ty() { return typeCache.SInt64Ty; }

  mlir::Type getUInt8Ty() { return typeCache.UInt8Ty; }
  mlir::Type getUInt16Ty() { return typeCache.UInt16Ty; }
  mlir::Type getUInt32Ty() { return typeCache.UInt32Ty; }
  mlir::Type getUInt64Ty() { return typeCache.UInt64Ty; }

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

  mlir::cir::BoolType getBoolTy() {
    return ::mlir::cir::BoolType::get(getContext());
  }
  mlir::Type getVirtualFnPtrType(bool isVarArg = false) {
    // FIXME: replay LLVM codegen for now, perhaps add a vtable ptr special
    // type so it's a bit more clear and C++ idiomatic.
    auto fnTy =
        mlir::cir::FuncType::get(getContext(), {}, {getUInt32Ty()}, isVarArg);
    assert(!UnimplementedFeature::isVarArg());
    return getPointerTo(getPointerTo(fnTy));
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
  mlir::cir::ConstantOp getSInt64(uint32_t C, mlir::Location loc) {
    auto sInt64Ty = getSInt64Ty();
    return create<mlir::cir::ConstantOp>(loc, sInt64Ty,
                                         mlir::cir::IntAttr::get(sInt64Ty, C));
  }
  mlir::cir::ConstantOp getUInt64(uint32_t C, mlir::Location loc) {
    auto uInt64Ty = getUInt64Ty();
    return create<mlir::cir::ConstantOp>(loc, uInt64Ty,
                                         mlir::cir::IntAttr::get(uInt64Ty, C));
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
    return create<mlir::cir::ConstantOp>(loc, ty, getNullPtrAttr(ty));
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
  mlir::Value createFPExt(mlir::Value v, mlir::Type destType) {
    if (getIsFPConstrained())
      llvm_unreachable("constrainedfp NYI");

    return create<mlir::cir::CastOp>(v.getLoc(), destType,
                                     mlir::cir::CastKind::floating, v);
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

  mlir::Value createBitcast(mlir::Location loc, mlir::Value src,
                            mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return create<mlir::cir::CastOp>(loc, newTy, mlir::cir::CastKind::bitcast,
                                     src);
  }

  mlir::Value createLoad(mlir::Location loc, Address addr) {
    return create<mlir::cir::LoadOp>(loc, addr.getElementType(),
                                     addr.getPointer());
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

  mlir::Value createNot(mlir::Value value) {
    return create<mlir::cir::UnaryOp>(value.getLoc(), value.getType(),
                                      mlir::cir::UnaryOpKind::Not, value);
  }

  mlir::Value createZExtOrBitCast(mlir::Location loc, mlir::Value src,
                                  mlir::Type newTy) {
    if (src.getType() == newTy)
      return src;
    llvm_unreachable("NYI");
  }
};

} // namespace cir

#endif
