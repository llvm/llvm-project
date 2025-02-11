//===-- CIRBaseBuilder.h - CIRBuilder implementation  -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIRBASEBUILDER_H
#define LLVM_CLANG_LIB_CIRBASEBUILDER_H

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"
#include "clang/CIR/MissingFeatures.h"

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

class CIRBaseBuilderTy : public mlir::OpBuilder {

public:
  CIRBaseBuilderTy(mlir::MLIRContext &C) : mlir::OpBuilder(&C) {}

  mlir::Value getConstAPSInt(mlir::Location loc, const llvm::APSInt &val) {
    auto ty =
        cir::IntType::get(getContext(), val.getBitWidth(), val.isSigned());
    return create<cir::ConstantOp>(loc, ty, getAttr<cir::IntAttr>(ty, val));
  }

  mlir::Value getSignedInt(mlir::Location loc, int64_t val, unsigned numBits) {
    return getConstAPSInt(
        loc, llvm::APSInt(llvm::APInt(numBits, val, /*isSigned=*/true),
                          /*isUnsigned=*/false));
  }

  mlir::Value getUnsignedInt(mlir::Location loc, uint64_t val,
                             unsigned numBits) {
    return getConstAPSInt(
        loc, llvm::APSInt(llvm::APInt(numBits, val), /*isUnsigned=*/true));
  }

  mlir::Value getConstAPInt(mlir::Location loc, mlir::Type typ,
                            const llvm::APInt &val) {
    return create<cir::ConstantOp>(loc, typ, getAttr<cir::IntAttr>(typ, val));
  }

  cir::ConstantOp getConstant(mlir::Location loc, mlir::TypedAttr attr) {
    return create<cir::ConstantOp>(loc, attr.getType(), attr);
  }

  // Creates constant null value for integral type ty.
  cir::ConstantOp getNullValue(mlir::Type ty, mlir::Location loc) {
    return create<cir::ConstantOp>(loc, ty, getZeroInitAttr(ty));
  }

  cir::ConstantOp getBool(bool state, mlir::Location loc) {
    return create<cir::ConstantOp>(loc, getBoolTy(), getCIRBoolAttr(state));
  }
  cir::ConstantOp getFalse(mlir::Location loc) { return getBool(false, loc); }
  cir::ConstantOp getTrue(mlir::Location loc) { return getBool(true, loc); }

  cir::BoolType getBoolTy() { return cir::BoolType::get(getContext()); }

  cir::VoidType getVoidTy() { return cir::VoidType::get(getContext()); }

  cir::IntType getUIntNTy(int N) {
    return cir::IntType::get(getContext(), N, false);
  }

  cir::IntType getSIntNTy(int N) {
    return cir::IntType::get(getContext(), N, true);
  }

  cir::AddressSpaceAttr getAddrSpaceAttr(clang::LangAS langAS) {
    if (langAS == clang::LangAS::Default)
      return {};
    return cir::AddressSpaceAttr::get(getContext(), langAS);
  }

  cir::PointerType getPointerTo(mlir::Type ty,
                                cir::AddressSpaceAttr cirAS = {}) {
    return cir::PointerType::get(getContext(), ty, cirAS);
  }

  cir::PointerType getPointerTo(mlir::Type ty, clang::LangAS langAS) {
    return getPointerTo(ty, getAddrSpaceAttr(langAS));
  }

  cir::PointerType getVoidPtrTy(clang::LangAS langAS = clang::LangAS::Default) {
    return getPointerTo(cir::VoidType::get(getContext()), langAS);
  }

  cir::PointerType getVoidPtrTy(cir::AddressSpaceAttr cirAS) {
    return getPointerTo(cir::VoidType::get(getContext()), cirAS);
  }

  cir::MethodAttr getMethodAttr(cir::MethodType ty, cir::FuncOp methodFuncOp) {
    auto methodFuncSymbolRef = mlir::FlatSymbolRefAttr::get(methodFuncOp);
    return cir::MethodAttr::get(ty, methodFuncSymbolRef);
  }

  cir::MethodAttr getNullMethodAttr(cir::MethodType ty) {
    return cir::MethodAttr::get(ty);
  }

  cir::BoolAttr getCIRBoolAttr(bool state) {
    return cir::BoolAttr::get(getContext(), getBoolTy(), state);
  }

  mlir::TypedAttr getZeroAttr(mlir::Type t) {
    return cir::ZeroAttr::get(getContext(), t);
  }

  mlir::TypedAttr getZeroInitAttr(mlir::Type ty) {
    if (mlir::isa<cir::IntType>(ty))
      return cir::IntAttr::get(ty, 0);
    if (auto fltType = mlir::dyn_cast<cir::SingleType>(ty))
      return cir::FPAttr::getZero(fltType);
    if (auto fltType = mlir::dyn_cast<cir::DoubleType>(ty))
      return cir::FPAttr::getZero(fltType);
    if (auto fltType = mlir::dyn_cast<cir::FP16Type>(ty))
      return cir::FPAttr::getZero(fltType);
    if (auto fltType = mlir::dyn_cast<cir::BF16Type>(ty))
      return cir::FPAttr::getZero(fltType);
    if (auto complexType = mlir::dyn_cast<cir::ComplexType>(ty))
      return getZeroAttr(complexType);
    if (auto arrTy = mlir::dyn_cast<cir::ArrayType>(ty))
      return getZeroAttr(arrTy);
    if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(ty))
      return getConstNullPtrAttr(ptrTy);
    if (auto structTy = mlir::dyn_cast<cir::StructType>(ty))
      return getZeroAttr(structTy);
    if (auto methodTy = mlir::dyn_cast<cir::MethodType>(ty))
      return getNullMethodAttr(methodTy);
    if (mlir::isa<cir::BoolType>(ty)) {
      return getCIRBoolAttr(false);
    }
    llvm_unreachable("Zero initializer for given type is NYI");
  }

  cir::LoadOp createLoad(mlir::Location loc, mlir::Value ptr,
                         bool isVolatile = false, uint64_t alignment = 0) {
    mlir::IntegerAttr intAttr;
    if (alignment)
      intAttr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(ptr.getContext(), 64), alignment);

    return create<cir::LoadOp>(loc, ptr, /*isDeref=*/false, isVolatile,
                               /*alignment=*/intAttr,
                               /*mem_order=*/
                               cir::MemOrderAttr{},
                               /*tbaa=*/cir::TBAAAttr{});
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Value ptr,
                                uint64_t alignment) {
    return createLoad(loc, ptr, /*isVolatile=*/false, alignment);
  }

  mlir::Value createNot(mlir::Value value) {
    return create<cir::UnaryOp>(value.getLoc(), value.getType(),
                                cir::UnaryOpKind::Not, value);
  }

  cir::CmpOp createCompare(mlir::Location loc, cir::CmpOpKind kind,
                           mlir::Value lhs, mlir::Value rhs) {
    return create<cir::CmpOp>(loc, getBoolTy(), kind, lhs, rhs);
  }

  mlir::Value createIsNaN(mlir::Location loc, mlir::Value operand) {
    return createCompare(loc, cir::CmpOpKind::ne, operand, operand);
  }

  mlir::Value createUnaryOp(mlir::Location loc, cir::UnaryOpKind kind,
                            mlir::Value operand) {
    return create<cir::UnaryOp>(loc, kind, operand);
  }

  mlir::Value createBinop(mlir::Value lhs, cir::BinOpKind kind,
                          const llvm::APInt &rhs) {
    return create<cir::BinOp>(lhs.getLoc(), lhs.getType(), kind, lhs,
                              getConstAPInt(lhs.getLoc(), lhs.getType(), rhs));
  }

  mlir::Value createBinop(mlir::Value lhs, cir::BinOpKind kind,
                          mlir::Value rhs) {
    return create<cir::BinOp>(lhs.getLoc(), lhs.getType(), kind, lhs, rhs);
  }

  mlir::Value createBinop(mlir::Location loc, mlir::Value lhs,
                          cir::BinOpKind kind, mlir::Value rhs) {
    return create<cir::BinOp>(loc, lhs.getType(), kind, lhs, rhs);
  }

  mlir::Value createShift(mlir::Value lhs, const llvm::APInt &rhs,
                          bool isShiftLeft) {
    return create<cir::ShiftOp>(lhs.getLoc(), lhs.getType(), lhs,
                                getConstAPInt(lhs.getLoc(), lhs.getType(), rhs),
                                isShiftLeft);
  }

  mlir::Value createShift(mlir::Value lhs, unsigned bits, bool isShiftLeft) {
    auto width = mlir::dyn_cast<cir::IntType>(lhs.getType()).getWidth();
    auto shift = llvm::APInt(width, bits);
    return createShift(lhs, shift, isShiftLeft);
  }

  mlir::Value createShiftLeft(mlir::Value lhs, unsigned bits) {
    return createShift(lhs, bits, true);
  }

  mlir::Value createShiftRight(mlir::Value lhs, unsigned bits) {
    return createShift(lhs, bits, false);
  }

  mlir::Value createLowBitsSet(mlir::Location loc, unsigned size,
                               unsigned bits) {
    auto val = llvm::APInt::getLowBitsSet(size, bits);
    auto typ = cir::IntType::get(getContext(), size, false);
    return getConstAPInt(loc, typ, val);
  }

  mlir::Value createAnd(mlir::Value lhs, llvm::APInt rhs) {
    auto val = getConstAPInt(lhs.getLoc(), lhs.getType(), rhs);
    return createBinop(lhs, cir::BinOpKind::And, val);
  }

  mlir::Value createAnd(mlir::Value lhs, mlir::Value rhs) {
    return createBinop(lhs, cir::BinOpKind::And, rhs);
  }

  mlir::Value createAnd(mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    return createBinop(loc, lhs, cir::BinOpKind::And, rhs);
  }

  mlir::Value createOr(mlir::Value lhs, llvm::APInt rhs) {
    auto val = getConstAPInt(lhs.getLoc(), lhs.getType(), rhs);
    return createBinop(lhs, cir::BinOpKind::Or, val);
  }

  mlir::Value createOr(mlir::Value lhs, mlir::Value rhs) {
    return createBinop(lhs, cir::BinOpKind::Or, rhs);
  }

  mlir::Value createMul(mlir::Value lhs, mlir::Value rhs, bool hasNUW = false,
                        bool hasNSW = false) {
    auto op = create<cir::BinOp>(lhs.getLoc(), lhs.getType(),
                                 cir::BinOpKind::Mul, lhs, rhs);
    if (hasNUW)
      op.setNoUnsignedWrap(true);
    if (hasNSW)
      op.setNoSignedWrap(true);
    return op;
  }
  mlir::Value createNSWMul(mlir::Value lhs, mlir::Value rhs) {
    return createMul(lhs, rhs, false, true);
  }
  mlir::Value createNUWAMul(mlir::Value lhs, mlir::Value rhs) {
    return createMul(lhs, rhs, true, false);
  }

  mlir::Value createMul(mlir::Value lhs, llvm::APInt rhs) {
    auto val = getConstAPInt(lhs.getLoc(), lhs.getType(), rhs);
    return createBinop(lhs, cir::BinOpKind::Mul, val);
  }

  mlir::Value createSelect(mlir::Location loc, mlir::Value condition,
                           mlir::Value trueValue, mlir::Value falseValue) {
    assert(trueValue.getType() == falseValue.getType() &&
           "trueValue and falseValue should have the same type");
    return create<cir::SelectOp>(loc, trueValue.getType(), condition, trueValue,
                                 falseValue);
  }

  mlir::Value createLogicalAnd(mlir::Location loc, mlir::Value lhs,
                               mlir::Value rhs) {
    return createSelect(loc, lhs, rhs, getBool(false, loc));
  }

  mlir::Value createLogicalOr(mlir::Location loc, mlir::Value lhs,
                              mlir::Value rhs) {
    return createSelect(loc, lhs, getBool(true, loc), rhs);
  }

  mlir::Value createComplexCreate(mlir::Location loc, mlir::Value real,
                                  mlir::Value imag) {
    auto resultComplexTy = cir::ComplexType::get(getContext(), real.getType());
    return create<cir::ComplexCreateOp>(loc, resultComplexTy, real, imag);
  }

  mlir::Value createComplexReal(mlir::Location loc, mlir::Value operand) {
    auto operandTy = mlir::cast<cir::ComplexType>(operand.getType());
    return create<cir::ComplexRealOp>(loc, operandTy.getElementTy(), operand);
  }

  mlir::Value createComplexImag(mlir::Location loc, mlir::Value operand) {
    auto operandTy = mlir::cast<cir::ComplexType>(operand.getType());
    return create<cir::ComplexImagOp>(loc, operandTy.getElementTy(), operand);
  }

  mlir::Value createComplexBinOp(mlir::Location loc, mlir::Value lhs,
                                 cir::ComplexBinOpKind kind, mlir::Value rhs,
                                 cir::ComplexRangeKind range, bool promoted) {
    return create<cir::ComplexBinOp>(loc, kind, lhs, rhs, range, promoted);
  }

  mlir::Value createComplexAdd(mlir::Location loc, mlir::Value lhs,
                               mlir::Value rhs) {
    return createBinop(loc, lhs, cir::BinOpKind::Add, rhs);
  }

  mlir::Value createComplexSub(mlir::Location loc, mlir::Value lhs,
                               mlir::Value rhs) {
    return createBinop(loc, lhs, cir::BinOpKind::Sub, rhs);
  }

  mlir::Value createComplexMul(mlir::Location loc, mlir::Value lhs,
                               mlir::Value rhs, cir::ComplexRangeKind range,
                               bool promoted) {
    return createComplexBinOp(loc, lhs, cir::ComplexBinOpKind::Mul, rhs, range,
                              promoted);
  }

  mlir::Value createComplexDiv(mlir::Location loc, mlir::Value lhs,
                               mlir::Value rhs, cir::ComplexRangeKind range,
                               bool promoted) {
    return createComplexBinOp(loc, lhs, cir::ComplexBinOpKind::Div, rhs, range,
                              promoted);
  }

  cir::StoreOp createStore(mlir::Location loc, mlir::Value val, mlir::Value dst,
                           bool _volatile = false,
                           ::mlir::IntegerAttr align = {},
                           cir::MemOrderAttr order = {}) {
    if (mlir::cast<cir::PointerType>(dst.getType()).getPointee() !=
        val.getType())
      dst = createPtrBitcast(dst, val.getType());
    return create<cir::StoreOp>(loc, val, dst, _volatile, align, order,
                                /*tbaa=*/cir::TBAAAttr{});
  }

  mlir::Value createAlloca(mlir::Location loc, cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           mlir::IntegerAttr alignment,
                           mlir::Value dynAllocSize) {
    return create<cir::AllocaOp>(loc, addrType, type, name, alignment,
                                 dynAllocSize);
  }

  mlir::Value createAlloca(mlir::Location loc, cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           clang::CharUnits alignment,
                           mlir::Value dynAllocSize) {
    auto alignmentIntAttr = getSizeFromCharUnits(getContext(), alignment);
    return createAlloca(loc, addrType, type, name, alignmentIntAttr,
                        dynAllocSize);
  }

  mlir::Value createAlloca(mlir::Location loc, cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           mlir::IntegerAttr alignment) {
    return create<cir::AllocaOp>(loc, addrType, type, name, alignment);
  }

  mlir::Value createAlloca(mlir::Location loc, cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           clang::CharUnits alignment) {
    auto alignmentIntAttr = getSizeFromCharUnits(getContext(), alignment);
    return createAlloca(loc, addrType, type, name, alignmentIntAttr);
  }

  mlir::Value createGetGlobal(mlir::Location loc, cir::GlobalOp global,
                              bool threadLocal = false) {
    return create<cir::GetGlobalOp>(
        loc, getPointerTo(global.getSymType(), global.getAddrSpaceAttr()),
        global.getName(), threadLocal);
  }

  mlir::Value createGetGlobal(cir::GlobalOp global, bool threadLocal = false) {
    return createGetGlobal(global.getLoc(), global, threadLocal);
  }

  /// Create a copy with inferred length.
  cir::CopyOp createCopy(mlir::Value dst, mlir::Value src,
                         bool isVolatile = false) {
    return create<cir::CopyOp>(dst.getLoc(), dst, src, isVolatile,
                               /*tbaa=*/cir::TBAAAttr{});
  }

  cir::MemCpyOp createMemCpy(mlir::Location loc, mlir::Value dst,
                             mlir::Value src, mlir::Value len) {
    return create<cir::MemCpyOp>(loc, dst, src, len);
  }

  cir::SignBitOp createSignBit(mlir::Location loc, mlir::Value val) {
    auto resTy = cir::BoolType::get(getContext());
    return create<cir::SignBitOp>(loc, resTy, val);
  }

  mlir::Value createSub(mlir::Value lhs, mlir::Value rhs, bool hasNUW = false,
                        bool hasNSW = false, bool saturated = false) {
    auto op = create<cir::BinOp>(lhs.getLoc(), lhs.getType(),
                                 cir::BinOpKind::Sub, lhs, rhs);
    if (hasNUW)
      op.setNoUnsignedWrap(true);
    if (hasNSW)
      op.setNoSignedWrap(true);
    if (saturated)
      op.setSaturated(true);
    return op;
  }

  mlir::Value createNSWSub(mlir::Value lhs, mlir::Value rhs) {
    return createSub(lhs, rhs, false, true);
  }

  mlir::Value createNUWSub(mlir::Value lhs, mlir::Value rhs) {
    return createSub(lhs, rhs, true, false);
  }

  mlir::Value createAdd(mlir::Value lhs, mlir::Value rhs, bool hasNUW = false,
                        bool hasNSW = false, bool saturated = false) {
    auto op = create<cir::BinOp>(lhs.getLoc(), lhs.getType(),
                                 cir::BinOpKind::Add, lhs, rhs);
    if (hasNUW)
      op.setNoUnsignedWrap(true);
    if (hasNSW)
      op.setNoSignedWrap(true);
    if (saturated)
      op.setSaturated(true);
    return op;
  }

  mlir::Value createNSWAdd(mlir::Value lhs, mlir::Value rhs) {
    return createAdd(lhs, rhs, false, true);
  }
  mlir::Value createNUWAdd(mlir::Value lhs, mlir::Value rhs) {
    return createAdd(lhs, rhs, true, false);
  }

  struct BinOpOverflowResults {
    mlir::Value result;
    mlir::Value overflow;
  };

  BinOpOverflowResults createBinOpOverflowOp(mlir::Location loc,
                                             cir::IntType resultTy,
                                             cir::BinOpOverflowKind kind,
                                             mlir::Value lhs, mlir::Value rhs) {
    auto op = create<cir::BinOpOverflowOp>(loc, resultTy, kind, lhs, rhs);
    return {op.getResult(), op.getOverflow()};
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  mlir::Value createCast(mlir::Location loc, cir::CastKind kind,
                         mlir::Value src, mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return create<cir::CastOp>(loc, newTy, kind, src);
  }

  mlir::Value createCast(cir::CastKind kind, mlir::Value src,
                         mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return createCast(src.getLoc(), kind, src, newTy);
  }

  mlir::Value createIntCast(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::integral, src, newTy);
  }

  mlir::Value createIntToPtr(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::int_to_ptr, src, newTy);
  }

  mlir::Value createGetMemberOp(mlir::Location &loc, mlir::Value structPtr,
                                const char *fldName, unsigned idx) {

    assert(mlir::isa<cir::PointerType>(structPtr.getType()));
    auto structBaseTy =
        mlir::cast<cir::PointerType>(structPtr.getType()).getPointee();
    assert(mlir::isa<cir::StructType>(structBaseTy));
    auto fldTy = mlir::cast<cir::StructType>(structBaseTy).getMembers()[idx];
    auto fldPtrTy = cir::PointerType::get(getContext(), fldTy);
    return create<cir::GetMemberOp>(loc, fldPtrTy, structPtr, fldName, idx);
  }

  mlir::Value createPtrToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::ptr_to_int, src, newTy);
  }

  mlir::Value createPtrToBoolCast(mlir::Value v) {
    return createCast(cir::CastKind::ptr_to_bool, v, getBoolTy());
  }

  // TODO(cir): the following function was introduced to keep in sync with LLVM
  // codegen. CIR does not have "zext" operations. It should eventually be
  // renamed or removed. For now, we just add whatever cast is required here.
  mlir::Value createZExtOrBitCast(mlir::Location loc, mlir::Value src,
                                  mlir::Type newTy) {
    auto srcTy = src.getType();

    if (srcTy == newTy)
      return src;

    if (mlir::isa<cir::BoolType>(srcTy) && mlir::isa<cir::IntType>(newTy))
      return createBoolToInt(src, newTy);

    llvm_unreachable("unhandled extension cast");
  }

  mlir::Value createBoolToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::bool_to_int, src, newTy);
  }

  mlir::Value createBitcast(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::bitcast, src, newTy);
  }

  mlir::Value createBitcast(mlir::Location loc, mlir::Value src,
                            mlir::Type newTy) {
    return createCast(loc, cir::CastKind::bitcast, src, newTy);
  }

  mlir::Value createPtrBitcast(mlir::Value src, mlir::Type newPointeeTy) {
    assert(mlir::isa<cir::PointerType>(src.getType()) && "expected ptr src");
    return createBitcast(src, getPointerTo(newPointeeTy));
  }

  mlir::Value createAddrSpaceCast(mlir::Location loc, mlir::Value src,
                                  mlir::Type newTy) {
    return createCast(loc, cir::CastKind::address_space, src, newTy);
  }

  mlir::Value createAddrSpaceCast(mlir::Value src, mlir::Type newTy) {
    return createAddrSpaceCast(src.getLoc(), src, newTy);
  }

  mlir::Value createPtrIsNull(mlir::Value ptr) {
    return createNot(createPtrToBoolCast(ptr));
  }

  //
  // Block handling helpers
  // ----------------------
  //
  static OpBuilder::InsertPoint getBestAllocaInsertPoint(mlir::Block *block) {
    auto last =
        std::find_if(block->rbegin(), block->rend(), [](mlir::Operation &op) {
          return mlir::isa<cir::AllocaOp, cir::LabelOp>(&op);
        });

    if (last != block->rend())
      return OpBuilder::InsertPoint(block, ++mlir::Block::iterator(&*last));
    return OpBuilder::InsertPoint(block, block->begin());
  };

  mlir::IntegerAttr getSizeFromCharUnits(mlir::MLIRContext *ctx,
                                         clang::CharUnits size) {
    // Note that mlir::IntegerType is used instead of cir::IntType here
    // because we don't need sign information for this to be useful, so keep
    // it simple.
    return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                  size.getQuantity());
  }

  /// Create a do-while operation.
  cir::DoWhileOp createDoWhile(
      mlir::Location loc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) {
    return create<cir::DoWhileOp>(loc, condBuilder, bodyBuilder);
  }

  /// Create a while operation.
  cir::WhileOp createWhile(
      mlir::Location loc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) {
    return create<cir::WhileOp>(loc, condBuilder, bodyBuilder);
  }

  /// Create a for operation.
  cir::ForOp createFor(
      mlir::Location loc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> stepBuilder) {
    return create<cir::ForOp>(loc, condBuilder, bodyBuilder, stepBuilder);
  }

  mlir::TypedAttr getConstPtrAttr(mlir::Type t, int64_t v) {
    auto val =
        mlir::IntegerAttr::get(mlir::IntegerType::get(t.getContext(), 64), v);
    return cir::ConstPtrAttr::get(getContext(), mlir::cast<cir::PointerType>(t),
                                  val);
  }

  mlir::TypedAttr getConstNullPtrAttr(mlir::Type t) {
    assert(mlir::isa<cir::PointerType>(t) && "expected cir.ptr");
    return getConstPtrAttr(t, 0);
  }

  // Creates constant nullptr for pointer type ty.
  cir::ConstantOp getNullPtr(mlir::Type ty, mlir::Location loc) {
    assert(!MissingFeatures::targetCodeGenInfoGetNullPointer());
    return create<cir::ConstantOp>(loc, ty, getConstPtrAttr(ty, 0));
  }

  /// Create a loop condition.
  cir::ConditionOp createCondition(mlir::Value condition) {
    return create<cir::ConditionOp>(condition.getLoc(), condition);
  }

  /// Create a yield operation.
  cir::YieldOp createYield(mlir::Location loc, mlir::ValueRange value = {}) {
    return create<cir::YieldOp>(loc, value);
  }

  cir::PtrStrideOp createPtrStride(mlir::Location loc, mlir::Value base,
                                   mlir::Value stride) {
    return create<cir::PtrStrideOp>(loc, base.getType(), base, stride);
  }

  cir::CallOp createCallOp(mlir::Location loc,
                           mlir::SymbolRefAttr callee = mlir::SymbolRefAttr(),
                           mlir::Type returnType = cir::VoidType(),
                           mlir::ValueRange operands = mlir::ValueRange(),
                           cir::CallingConv callingConv = cir::CallingConv::C,
                           cir::SideEffect sideEffect = cir::SideEffect::All,
                           cir::ExtraFuncAttributesAttr extraFnAttr = {}) {

    cir::CallOp callOp = create<cir::CallOp>(loc, callee, returnType, operands,
                                             callingConv, sideEffect);

    if (extraFnAttr) {
      callOp->setAttr("extra_attrs", extraFnAttr);
    } else {
      mlir::NamedAttrList empty;
      callOp->setAttr("extra_attrs",
                      cir::ExtraFuncAttributesAttr::get(
                          getContext(), empty.getDictionary(getContext())));
    }
    return callOp;
  }

  cir::CallOp createCallOp(mlir::Location loc, cir::FuncOp callee,
                           mlir::ValueRange operands = mlir::ValueRange(),
                           cir::CallingConv callingConv = cir::CallingConv::C,
                           cir::SideEffect sideEffect = cir::SideEffect::All,
                           cir::ExtraFuncAttributesAttr extraFnAttr = {}) {
    return createCallOp(loc, mlir::SymbolRefAttr::get(callee),
                        callee.getFunctionType().getReturnType(), operands,
                        callingConv, sideEffect, extraFnAttr);
  }

  cir::CallOp
  createIndirectCallOp(mlir::Location loc, mlir::Value ind_target,
                       cir::FuncType fn_type,
                       mlir::ValueRange operands = mlir::ValueRange(),
                       cir::CallingConv callingConv = cir::CallingConv::C,
                       cir::SideEffect sideEffect = cir::SideEffect::All,
                       cir::ExtraFuncAttributesAttr extraFnAttr = {}) {

    llvm::SmallVector<mlir::Value, 4> resOperands({ind_target});
    resOperands.append(operands.begin(), operands.end());

    return createCallOp(loc, mlir::SymbolRefAttr(), fn_type.getReturnType(),
                        resOperands, callingConv, sideEffect, extraFnAttr);
  }

  cir::CallOp createCallOp(mlir::Location loc, mlir::SymbolRefAttr callee,
                           mlir::ValueRange operands = mlir::ValueRange(),
                           cir::CallingConv callingConv = cir::CallingConv::C,
                           cir::SideEffect sideEffect = cir::SideEffect::All,
                           cir::ExtraFuncAttributesAttr extraFnAttr = {}) {
    return createCallOp(loc, callee, cir::VoidType(), operands, callingConv,
                        sideEffect, extraFnAttr);
  }

  cir::CallOp
  createTryCallOp(mlir::Location loc,
                  mlir::SymbolRefAttr callee = mlir::SymbolRefAttr(),
                  mlir::Type returnType = cir::VoidType(),
                  mlir::ValueRange operands = mlir::ValueRange(),
                  cir::CallingConv callingConv = cir::CallingConv::C,
                  cir::SideEffect sideEffect = cir::SideEffect::All,
                  cir::ExtraFuncAttributesAttr extraFnAttr = {}) {
    cir::CallOp tryCallOp =
        create<cir::CallOp>(loc, callee, returnType, operands, callingConv,
                            sideEffect, /*exception=*/getUnitAttr());
    if (extraFnAttr) {
      tryCallOp->setAttr("extra_attrs", extraFnAttr);
    } else {
      mlir::NamedAttrList empty;
      tryCallOp->setAttr("extra_attrs",
                         cir::ExtraFuncAttributesAttr::get(
                             getContext(), empty.getDictionary(getContext())));
    }
    return tryCallOp;
  }

  cir::CallOp
  createTryCallOp(mlir::Location loc, cir::FuncOp callee,
                  mlir::ValueRange operands,
                  cir::CallingConv callingConv = cir::CallingConv::C,
                  cir::SideEffect sideEffect = cir::SideEffect::All,
                  cir::ExtraFuncAttributesAttr extraFnAttr = {}) {
    return createTryCallOp(loc, mlir::SymbolRefAttr::get(callee),
                           callee.getFunctionType().getReturnType(), operands,
                           callingConv, sideEffect, extraFnAttr);
  }

  cir::CallOp
  createIndirectTryCallOp(mlir::Location loc, mlir::Value ind_target,
                          cir::FuncType fn_type, mlir::ValueRange operands,
                          cir::CallingConv callingConv = cir::CallingConv::C,
                          cir::SideEffect sideEffect = cir::SideEffect::All) {
    llvm::SmallVector<mlir::Value, 4> resOperands({ind_target});
    resOperands.append(operands.begin(), operands.end());
    return createTryCallOp(loc, mlir::SymbolRefAttr(), fn_type.getReturnType(),
                           resOperands, callingConv, sideEffect);
  }

  struct GetMethodResults {
    mlir::Value callee;
    mlir::Value adjustedThis;
  };

  GetMethodResults createGetMethod(mlir::Location loc, mlir::Value method,
                                   mlir::Value objectPtr) {
    // Build the callee function type.
    auto methodFuncTy =
        mlir::cast<cir::MethodType>(method.getType()).getMemberFuncTy();
    auto methodFuncInputTypes = methodFuncTy.getInputs();

    auto objectPtrTy = mlir::cast<cir::PointerType>(objectPtr.getType());
    auto objectPtrAddrSpace = mlir::cast_if_present<cir::AddressSpaceAttr>(
        objectPtrTy.getAddrSpace());
    auto adjustedThisTy = getVoidPtrTy(objectPtrAddrSpace);

    llvm::SmallVector<mlir::Type, 8> calleeFuncInputTypes{adjustedThisTy};
    calleeFuncInputTypes.insert(calleeFuncInputTypes.end(),
                                methodFuncInputTypes.begin(),
                                methodFuncInputTypes.end());
    auto calleeFuncTy =
        methodFuncTy.clone(calleeFuncInputTypes, methodFuncTy.getReturnType());
    // TODO(cir): consider the address space of the callee.
    assert(!MissingFeatures::addressSpace());
    auto calleeTy = getPointerTo(calleeFuncTy);

    auto op = create<cir::GetMethodOp>(loc, calleeTy, adjustedThisTy, method,
                                       objectPtr);
    return {op.getCallee(), op.getAdjustedThis()};
  }
};

} // namespace cir
#endif
