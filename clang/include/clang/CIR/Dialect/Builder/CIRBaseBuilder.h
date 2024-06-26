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
    auto ty = mlir::cir::IntType::get(getContext(), val.getBitWidth(),
                                      val.isSigned());
    return create<mlir::cir::ConstantOp>(loc, ty,
                                         getAttr<mlir::cir::IntAttr>(ty, val));
  }

  mlir::Value getConstAPInt(mlir::Location loc, mlir::Type typ,
                            const llvm::APInt &val) {
    return create<mlir::cir::ConstantOp>(loc, typ,
                                         getAttr<mlir::cir::IntAttr>(typ, val));
  }

  mlir::cir::ConstantOp getConstant(mlir::Location loc, mlir::TypedAttr attr) {
    return create<mlir::cir::ConstantOp>(loc, attr.getType(), attr);
  }

  mlir::cir::BoolType getBoolTy() {
    return ::mlir::cir::BoolType::get(getContext());
  }

  mlir::cir::VoidType getVoidTy() {
    return ::mlir::cir::VoidType::get(getContext());
  }

  mlir::cir::IntType getUIntNTy(int N) {
    return mlir::cir::IntType::get(getContext(), N, false);
  }

  mlir::cir::IntType getSIntNTy(int N) {
    return mlir::cir::IntType::get(getContext(), N, true);
  }

  mlir::cir::PointerType getPointerTo(mlir::Type ty,
                                      unsigned addressSpace = 0) {
    assert(!addressSpace && "address space is NYI");
    return mlir::cir::PointerType::get(getContext(), ty);
  }

  mlir::cir::PointerType getVoidPtrTy(unsigned addressSpace = 0) {
    return getPointerTo(::mlir::cir::VoidType::get(getContext()), addressSpace);
  }

  mlir::Value createLoad(mlir::Location loc, mlir::Value ptr,
                         bool isVolatile = false, uint64_t alignment = 0) {
    mlir::IntegerAttr intAttr;
    if (alignment)
      intAttr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(ptr.getContext(), 64), alignment);

    return create<mlir::cir::LoadOp>(loc, ptr, /*isDeref=*/false, isVolatile,
                                     /*alignment=*/intAttr,
                                     /*mem_order=*/mlir::cir::MemOrderAttr{});
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Value ptr,
                                uint64_t alignment) {
    return createLoad(loc, ptr, /*isVolatile=*/false, alignment);
  }

  mlir::Value createNot(mlir::Value value) {
    return create<mlir::cir::UnaryOp>(value.getLoc(), value.getType(),
                                      mlir::cir::UnaryOpKind::Not, value);
  }

  mlir::cir::CmpOp createCompare(mlir::Location loc, mlir::cir::CmpOpKind kind,
                                 mlir::Value lhs, mlir::Value rhs) {
    return create<mlir::cir::CmpOp>(loc, getBoolTy(), kind, lhs, rhs);
  }

  mlir::Value createBinop(mlir::Value lhs, mlir::cir::BinOpKind kind,
                          const llvm::APInt &rhs) {
    return create<mlir::cir::BinOp>(
        lhs.getLoc(), lhs.getType(), kind, lhs,
        getConstAPInt(lhs.getLoc(), lhs.getType(), rhs));
  }

  mlir::Value createBinop(mlir::Value lhs, mlir::cir::BinOpKind kind,
                          mlir::Value rhs) {
    return create<mlir::cir::BinOp>(lhs.getLoc(), lhs.getType(), kind, lhs,
                                    rhs);
  }

  mlir::Value createShift(mlir::Value lhs, const llvm::APInt &rhs,
                          bool isShiftLeft) {
    return create<mlir::cir::ShiftOp>(
        lhs.getLoc(), lhs.getType(), lhs,
        getConstAPInt(lhs.getLoc(), lhs.getType(), rhs), isShiftLeft);
  }

  mlir::Value createShift(mlir::Value lhs, unsigned bits, bool isShiftLeft) {
    auto width = mlir::dyn_cast<mlir::cir::IntType>(lhs.getType()).getWidth();
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
    auto typ = mlir::cir::IntType::get(getContext(), size, false);
    return getConstAPInt(loc, typ, val);
  }

  mlir::Value createAnd(mlir::Value lhs, llvm::APInt rhs) {
    auto val = getConstAPInt(lhs.getLoc(), lhs.getType(), rhs);
    return createBinop(lhs, mlir::cir::BinOpKind::And, val);
  }

  mlir::Value createAnd(mlir::Value lhs, mlir::Value rhs) {
    return createBinop(lhs, mlir::cir::BinOpKind::And, rhs);
  }

  mlir::Value createOr(mlir::Value lhs, llvm::APInt rhs) {
    auto val = getConstAPInt(lhs.getLoc(), lhs.getType(), rhs);
    return createBinop(lhs, mlir::cir::BinOpKind::Or, val);
  }

  mlir::Value createOr(mlir::Value lhs, mlir::Value rhs) {
    return createBinop(lhs, mlir::cir::BinOpKind::Or, rhs);
  }

  mlir::Value createMul(mlir::Value lhs, mlir::Value rhs, bool hasNUW = false,
                        bool hasNSW = false) {
    auto op = create<mlir::cir::BinOp>(lhs.getLoc(), lhs.getType(),
                                       mlir::cir::BinOpKind::Mul, lhs, rhs);
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
    return createBinop(lhs, mlir::cir::BinOpKind::Mul, val);
  }

  mlir::cir::StoreOp createStore(mlir::Location loc, mlir::Value val,
                                 mlir::Value dst, bool _volatile = false,
                                 ::mlir::IntegerAttr align = {},
                                 ::mlir::cir::MemOrderAttr order = {}) {
    if (mlir::cast<mlir::cir::PointerType>(dst.getType()).getPointee() !=
        val.getType())
      dst = createPtrBitcast(dst, val.getType());
    return create<mlir::cir::StoreOp>(loc, val, dst, _volatile, align, order);
  }

  mlir::Value createAlloca(mlir::Location loc, mlir::cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           mlir::IntegerAttr alignment,
                           mlir::Value dynAllocSize) {
    return create<mlir::cir::AllocaOp>(loc, addrType, type, name, alignment,
                                       dynAllocSize);
  }

  mlir::Value createAlloca(mlir::Location loc, mlir::cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           clang::CharUnits alignment,
                           mlir::Value dynAllocSize) {
    auto alignmentIntAttr = getSizeFromCharUnits(getContext(), alignment);
    return createAlloca(loc, addrType, type, name, alignmentIntAttr,
                        dynAllocSize);
  }

  mlir::Value createAlloca(mlir::Location loc, mlir::cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           mlir::IntegerAttr alignment) {
    return create<mlir::cir::AllocaOp>(loc, addrType, type, name, alignment);
  }

  mlir::Value createAlloca(mlir::Location loc, mlir::cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           clang::CharUnits alignment) {
    auto alignmentIntAttr = getSizeFromCharUnits(getContext(), alignment);
    return createAlloca(loc, addrType, type, name, alignmentIntAttr);
  }

  mlir::Value createSub(mlir::Value lhs, mlir::Value rhs, bool hasNUW = false,
                        bool hasNSW = false) {
    auto op = create<mlir::cir::BinOp>(lhs.getLoc(), lhs.getType(),
                                       mlir::cir::BinOpKind::Sub, lhs, rhs);
    if (hasNUW)
      op.setNoUnsignedWrap(true);
    if (hasNSW)
      op.setNoSignedWrap(true);
    return op;
  }

  mlir::Value createNSWSub(mlir::Value lhs, mlir::Value rhs) {
    return createSub(lhs, rhs, false, true);
  }

  mlir::Value createNUWSub(mlir::Value lhs, mlir::Value rhs) {
    return createSub(lhs, rhs, true, false);
  }

  mlir::Value createAdd(mlir::Value lhs, mlir::Value rhs, bool hasNUW = false,
                        bool hasNSW = false) {
    auto op = create<mlir::cir::BinOp>(lhs.getLoc(), lhs.getType(),
                                       mlir::cir::BinOpKind::Add, lhs, rhs);
    if (hasNUW)
      op.setNoUnsignedWrap(true);
    if (hasNSW)
      op.setNoSignedWrap(true);
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
                                             mlir::cir::IntType resultTy,
                                             mlir::cir::BinOpOverflowKind kind,
                                             mlir::Value lhs, mlir::Value rhs) {
    auto op = create<mlir::cir::BinOpOverflowOp>(loc, resultTy, kind, lhs, rhs);
    return {op.getResult(), op.getOverflow()};
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  mlir::Value createCast(mlir::Location loc, mlir::cir::CastKind kind,
                         mlir::Value src, mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return create<mlir::cir::CastOp>(loc, newTy, kind, src);
  }

  mlir::Value createCast(mlir::cir::CastKind kind, mlir::Value src,
                         mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return createCast(src.getLoc(), kind, src, newTy);
  }

  mlir::Value createIntCast(mlir::Value src, mlir::Type newTy) {
    return createCast(mlir::cir::CastKind::integral, src, newTy);
  }

  mlir::Value createIntToPtr(mlir::Value src, mlir::Type newTy) {
    return createCast(mlir::cir::CastKind::int_to_ptr, src, newTy);
  }

  mlir::Value createGetMemberOp(mlir::Location &loc, mlir::Value structPtr,
                                const char *fldName, unsigned idx) {

    assert(mlir::isa<mlir::cir::PointerType>(structPtr.getType()));
    auto structBaseTy =
        mlir::cast<mlir::cir::PointerType>(structPtr.getType()).getPointee();
    assert(mlir::isa<mlir::cir::StructType>(structBaseTy));
    auto fldTy =
        mlir::cast<mlir::cir::StructType>(structBaseTy).getMembers()[idx];
    auto fldPtrTy = ::mlir::cir::PointerType::get(getContext(), fldTy);
    return create<mlir::cir::GetMemberOp>(loc, fldPtrTy, structPtr, fldName,
                                          idx);
  }

  mlir::Value createPtrToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(mlir::cir::CastKind::ptr_to_int, src, newTy);
  }

  mlir::Value createPtrToBoolCast(mlir::Value v) {
    return createCast(mlir::cir::CastKind::ptr_to_bool, v, getBoolTy());
  }

  // TODO(cir): the following function was introduced to keep in sync with LLVM
  // codegen. CIR does not have "zext" operations. It should eventually be
  // renamed or removed. For now, we just add whatever cast is required here.
  mlir::Value createZExtOrBitCast(mlir::Location loc, mlir::Value src,
                                  mlir::Type newTy) {
    auto srcTy = src.getType();

    if (srcTy == newTy)
      return src;

    if (mlir::isa<mlir::cir::BoolType>(srcTy) &&
        mlir::isa<mlir::cir::IntType>(newTy))
      return createBoolToInt(src, newTy);

    llvm_unreachable("unhandled extension cast");
  }

  mlir::Value createBoolToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(mlir::cir::CastKind::bool_to_int, src, newTy);
  }

  mlir::Value createBitcast(mlir::Value src, mlir::Type newTy) {
    return createCast(mlir::cir::CastKind::bitcast, src, newTy);
  }

  mlir::Value createBitcast(mlir::Location loc, mlir::Value src,
                            mlir::Type newTy) {
    return createCast(loc, mlir::cir::CastKind::bitcast, src, newTy);
  }

  mlir::Value createPtrBitcast(mlir::Value src, mlir::Type newPointeeTy) {
    assert(mlir::isa<mlir::cir::PointerType>(src.getType()) &&
           "expected ptr src");
    return createBitcast(src, getPointerTo(newPointeeTy));
  }

  mlir::Value createAddrSpaceCast(mlir::Location loc, mlir::Value src,
                                  mlir::Type newTy) {
    return createCast(loc, mlir::cir::CastKind::address_space, src, newTy);
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

  mlir::IntegerAttr getSizeFromCharUnits(mlir::MLIRContext *ctx,
                                         clang::CharUnits size) {
    // Note that mlir::IntegerType is used instead of mlir::cir::IntType here
    // because we don't need sign information for this to be useful, so keep
    // it simple.
    return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                  size.getQuantity());
  }

  /// Create a do-while operation.
  mlir::cir::DoWhileOp createDoWhile(
      mlir::Location loc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) {
    return create<mlir::cir::DoWhileOp>(loc, condBuilder, bodyBuilder);
  }

  /// Create a while operation.
  mlir::cir::WhileOp createWhile(
      mlir::Location loc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder) {
    return create<mlir::cir::WhileOp>(loc, condBuilder, bodyBuilder);
  }

  /// Create a for operation.
  mlir::cir::ForOp createFor(
      mlir::Location loc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> stepBuilder) {
    return create<mlir::cir::ForOp>(loc, condBuilder, bodyBuilder, stepBuilder);
  }

  mlir::TypedAttr getConstPtrAttr(mlir::Type t, int64_t v) {
    auto val =
        mlir::IntegerAttr::get(mlir::IntegerType::get(t.getContext(), 64), v);
    return mlir::cir::ConstPtrAttr::get(
        getContext(), mlir::cast<mlir::cir::PointerType>(t), val);
  }

  // Creates constant nullptr for pointer type ty.
  mlir::cir::ConstantOp getNullPtr(mlir::Type ty, mlir::Location loc) {
    assert(!MissingFeatures::targetCodeGenInfoGetNullPointer());
    return create<mlir::cir::ConstantOp>(loc, ty, getConstPtrAttr(ty, 0));
  }

  /// Create a loop condition.
  mlir::cir::ConditionOp createCondition(mlir::Value condition) {
    return create<mlir::cir::ConditionOp>(condition.getLoc(), condition);
  }

  /// Create a yield operation.
  mlir::cir::YieldOp createYield(mlir::Location loc,
                                 mlir::ValueRange value = {}) {
    return create<mlir::cir::YieldOp>(loc, value);
  }

  mlir::cir::CallOp
  createCallOp(mlir::Location loc,
               mlir::SymbolRefAttr callee = mlir::SymbolRefAttr(),
               mlir::Type returnType = mlir::cir::VoidType(),
               mlir::ValueRange operands = mlir::ValueRange(),
               mlir::cir::ExtraFuncAttributesAttr extraFnAttr = {}) {

    mlir::cir::CallOp callOp =
        create<mlir::cir::CallOp>(loc, callee, returnType, operands);

    if (extraFnAttr) {
      callOp->setAttr("extra_attrs", extraFnAttr);
    } else {
      mlir::NamedAttrList empty;
      callOp->setAttr("extra_attrs",
                      mlir::cir::ExtraFuncAttributesAttr::get(
                          getContext(), empty.getDictionary(getContext())));
    }
    return callOp;
  }

  mlir::cir::CallOp
  createCallOp(mlir::Location loc, mlir::cir::FuncOp callee,
               mlir::ValueRange operands = mlir::ValueRange(),
               mlir::cir::ExtraFuncAttributesAttr extraFnAttr = {}) {
    return createCallOp(loc, mlir::SymbolRefAttr::get(callee),
                        callee.getFunctionType().getReturnType(), operands,
                        extraFnAttr);
  }

  mlir::cir::CallOp
  createIndirectCallOp(mlir::Location loc, mlir::Value ind_target,
                       mlir::cir::FuncType fn_type,
                       mlir::ValueRange operands = mlir::ValueRange(),
                       mlir::cir::ExtraFuncAttributesAttr extraFnAttr = {}) {

    llvm::SmallVector<mlir::Value, 4> resOperands({ind_target});
    resOperands.append(operands.begin(), operands.end());

    return createCallOp(loc, mlir::SymbolRefAttr(), fn_type.getReturnType(),
                        resOperands, extraFnAttr);
  }

  mlir::cir::CallOp
  createCallOp(mlir::Location loc, mlir::SymbolRefAttr callee,
               mlir::ValueRange operands = mlir::ValueRange(),
               mlir::cir::ExtraFuncAttributesAttr extraFnAttr = {}) {
    return createCallOp(loc, callee, mlir::cir::VoidType(), operands,
                        extraFnAttr);
  }
};

} // namespace cir
#endif
