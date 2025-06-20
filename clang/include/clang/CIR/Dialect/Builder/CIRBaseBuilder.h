//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H
#define LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H

#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/ErrorHandling.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"

namespace cir {

enum class OverflowBehavior {
  None = 0,
  NoSignedWrap = 1 << 0,
  NoUnsignedWrap = 1 << 1,
  Saturated = 1 << 2,
};

constexpr OverflowBehavior operator|(OverflowBehavior a, OverflowBehavior b) {
  return static_cast<OverflowBehavior>(llvm::to_underlying(a) |
                                       llvm::to_underlying(b));
}

constexpr OverflowBehavior operator&(OverflowBehavior a, OverflowBehavior b) {
  return static_cast<OverflowBehavior>(llvm::to_underlying(a) &
                                       llvm::to_underlying(b));
}

constexpr OverflowBehavior &operator|=(OverflowBehavior &a,
                                       OverflowBehavior b) {
  a = a | b;
  return a;
}

constexpr OverflowBehavior &operator&=(OverflowBehavior &a,
                                       OverflowBehavior b) {
  a = a & b;
  return a;
}

class CIRBaseBuilderTy : public mlir::OpBuilder {

public:
  CIRBaseBuilderTy(mlir::MLIRContext &mlirContext)
      : mlir::OpBuilder(&mlirContext) {}
  CIRBaseBuilderTy(mlir::OpBuilder &builder) : mlir::OpBuilder(builder) {}

  mlir::Value getConstAPInt(mlir::Location loc, mlir::Type typ,
                            const llvm::APInt &val) {
    return create<cir::ConstantOp>(loc, getAttr<cir::IntAttr>(typ, val));
  }

  cir::ConstantOp getConstant(mlir::Location loc, mlir::TypedAttr attr) {
    return create<cir::ConstantOp>(loc, attr);
  }

  cir::ConstantOp getConstantInt(mlir::Location loc, mlir::Type ty,
                                 int64_t value) {
    return getConstant(loc, cir::IntAttr::get(ty, value));
  }

  // Creates constant null value for integral type ty.
  cir::ConstantOp getNullValue(mlir::Type ty, mlir::Location loc) {
    return getConstant(loc, getZeroInitAttr(ty));
  }

  mlir::TypedAttr getConstNullPtrAttr(mlir::Type t) {
    assert(mlir::isa<cir::PointerType>(t) && "expected cir.ptr");
    return getConstPtrAttr(t, 0);
  }

  mlir::TypedAttr getZeroInitAttr(mlir::Type ty) {
    if (mlir::isa<cir::IntType>(ty))
      return cir::IntAttr::get(ty, 0);
    if (cir::isAnyFloatingPointType(ty))
      return cir::FPAttr::getZero(ty);
    if (auto complexType = mlir::dyn_cast<cir::ComplexType>(ty))
      return cir::ZeroAttr::get(complexType);
    if (auto arrTy = mlir::dyn_cast<cir::ArrayType>(ty))
      return cir::ZeroAttr::get(arrTy);
    if (auto vecTy = mlir::dyn_cast<cir::VectorType>(ty))
      return cir::ZeroAttr::get(vecTy);
    if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(ty))
      return getConstNullPtrAttr(ptrTy);
    if (auto recordTy = mlir::dyn_cast<cir::RecordType>(ty))
      return cir::ZeroAttr::get(recordTy);
    if (mlir::isa<cir::BoolType>(ty)) {
      return getFalseAttr();
    }
    llvm_unreachable("Zero initializer for given type is NYI");
  }

  cir::ConstantOp getBool(bool state, mlir::Location loc) {
    return create<cir::ConstantOp>(loc, getCIRBoolAttr(state));
  }
  cir::ConstantOp getFalse(mlir::Location loc) { return getBool(false, loc); }
  cir::ConstantOp getTrue(mlir::Location loc) { return getBool(true, loc); }

  cir::BoolType getBoolTy() { return cir::BoolType::get(getContext()); }

  cir::PointerType getPointerTo(mlir::Type ty) {
    return cir::PointerType::get(ty);
  }

  cir::PointerType getVoidPtrTy() {
    return getPointerTo(cir::VoidType::get(getContext()));
  }

  cir::BoolAttr getCIRBoolAttr(bool state) {
    return cir::BoolAttr::get(getContext(), state);
  }

  cir::BoolAttr getTrueAttr() { return getCIRBoolAttr(true); }
  cir::BoolAttr getFalseAttr() { return getCIRBoolAttr(false); }

  mlir::Value createNot(mlir::Value value) {
    return create<cir::UnaryOp>(value.getLoc(), value.getType(),
                                cir::UnaryOpKind::Not, value);
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

  /// Create a break operation.
  cir::BreakOp createBreak(mlir::Location loc) {
    return create<cir::BreakOp>(loc);
  }

  /// Create a continue operation.
  cir::ContinueOp createContinue(mlir::Location loc) {
    return create<cir::ContinueOp>(loc);
  }

  mlir::TypedAttr getConstPtrAttr(mlir::Type type, int64_t value) {
    return cir::ConstPtrAttr::get(type, getI64IntegerAttr(value));
  }

  mlir::Value createAlloca(mlir::Location loc, cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           mlir::IntegerAttr alignment) {
    return create<cir::AllocaOp>(loc, addrType, type, name, alignment);
  }

  mlir::Value createGetGlobal(mlir::Location loc, cir::GlobalOp global) {
    assert(!cir::MissingFeatures::addressSpace());
    return create<cir::GetGlobalOp>(loc, getPointerTo(global.getSymType()),
                                    global.getSymName());
  }

  mlir::Value createGetGlobal(cir::GlobalOp global) {
    return createGetGlobal(global.getLoc(), global);
  }

  cir::StoreOp createStore(mlir::Location loc, mlir::Value val, mlir::Value dst,
                           mlir::IntegerAttr align = {}) {
    return create<cir::StoreOp>(loc, val, dst, align);
  }

  [[nodiscard]] cir::GlobalOp createGlobal(mlir::ModuleOp mlirModule,
                                           mlir::Location loc,
                                           mlir::StringRef name,
                                           mlir::Type type,
                                           cir::GlobalLinkageKind linkage) {
    mlir::OpBuilder::InsertionGuard guard(*this);
    setInsertionPointToStart(mlirModule.getBody());
    return create<cir::GlobalOp>(loc, name, type, linkage);
  }

  cir::GetMemberOp createGetMember(mlir::Location loc, mlir::Type resultTy,
                                   mlir::Value base, llvm::StringRef name,
                                   unsigned index) {
    return create<cir::GetMemberOp>(loc, resultTy, base, name, index);
  }

  mlir::Value createDummyValue(mlir::Location loc, mlir::Type type,
                               clang::CharUnits alignment) {
    mlir::IntegerAttr alignmentAttr = getAlignmentAttr(alignment);
    auto addr = createAlloca(loc, getPointerTo(type), type, {}, alignmentAttr);
    return create<cir::LoadOp>(loc, addr, /*isDeref=*/false, alignmentAttr);
  }

  cir::PtrStrideOp createPtrStride(mlir::Location loc, mlir::Value base,
                                   mlir::Value stride) {
    return create<cir::PtrStrideOp>(loc, base.getType(), base, stride);
  }

  //===--------------------------------------------------------------------===//
  // Call operators
  //===--------------------------------------------------------------------===//

  cir::CallOp createCallOp(mlir::Location loc, mlir::SymbolRefAttr callee,
                           mlir::Type returnType, mlir::ValueRange operands) {
    return create<cir::CallOp>(loc, callee, returnType, operands);
  }

  cir::CallOp createCallOp(mlir::Location loc, cir::FuncOp callee,
                           mlir::ValueRange operands) {
    return createCallOp(loc, mlir::SymbolRefAttr::get(callee),
                        callee.getFunctionType().getReturnType(), operands);
  }

  cir::CallOp createIndirectCallOp(mlir::Location loc,
                                   mlir::Value indirectTarget,
                                   cir::FuncType funcType,
                                   mlir::ValueRange operands) {
    return create<cir::CallOp>(loc, indirectTarget, funcType.getReturnType(),
                               operands);
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

  mlir::Value createPtrToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::ptr_to_int, src, newTy);
  }

  mlir::Value createPtrToBoolCast(mlir::Value v) {
    return createCast(cir::CastKind::ptr_to_bool, v, getBoolTy());
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

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  mlir::Value createBinop(mlir::Location loc, mlir::Value lhs,
                          cir::BinOpKind kind, mlir::Value rhs) {
    return create<cir::BinOp>(loc, lhs.getType(), kind, lhs, rhs);
  }

  mlir::Value createLowBitsSet(mlir::Location loc, unsigned size,
                               unsigned bits) {
    llvm::APInt val = llvm::APInt::getLowBitsSet(size, bits);
    auto type = cir::IntType::get(getContext(), size, /*isSigned=*/false);
    return getConstAPInt(loc, type, val);
  }

  mlir::Value createAnd(mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    return createBinop(loc, lhs, cir::BinOpKind::And, rhs);
  }

  mlir::Value createOr(mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    return createBinop(loc, lhs, cir::BinOpKind::Or, rhs);
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

  mlir::Value createMul(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                        OverflowBehavior ob = OverflowBehavior::None) {
    auto op =
        create<cir::BinOp>(loc, lhs.getType(), cir::BinOpKind::Mul, lhs, rhs);
    op.setNoUnsignedWrap(
        llvm::to_underlying(ob & OverflowBehavior::NoUnsignedWrap));
    op.setNoSignedWrap(
        llvm::to_underlying(ob & OverflowBehavior::NoSignedWrap));
    return op;
  }
  mlir::Value createNSWMul(mlir::Location loc, mlir::Value lhs,
                           mlir::Value rhs) {
    return createMul(loc, lhs, rhs, OverflowBehavior::NoSignedWrap);
  }
  mlir::Value createNUWAMul(mlir::Location loc, mlir::Value lhs,
                            mlir::Value rhs) {
    return createMul(loc, lhs, rhs, OverflowBehavior::NoUnsignedWrap);
  }

  mlir::Value createSub(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                        OverflowBehavior ob = OverflowBehavior::Saturated) {
    auto op =
        create<cir::BinOp>(loc, lhs.getType(), cir::BinOpKind::Sub, lhs, rhs);
    op.setNoUnsignedWrap(
        llvm::to_underlying(ob & OverflowBehavior::NoUnsignedWrap));
    op.setNoSignedWrap(
        llvm::to_underlying(ob & OverflowBehavior::NoSignedWrap));
    op.setSaturated(llvm::to_underlying(ob & OverflowBehavior::Saturated));
    return op;
  }

  mlir::Value createNSWSub(mlir::Location loc, mlir::Value lhs,
                           mlir::Value rhs) {
    return createSub(loc, lhs, rhs, OverflowBehavior::NoSignedWrap);
  }

  mlir::Value createNUWSub(mlir::Location loc, mlir::Value lhs,
                           mlir::Value rhs) {
    return createSub(loc, lhs, rhs, OverflowBehavior::NoUnsignedWrap);
  }

  mlir::Value createAdd(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                        OverflowBehavior ob = OverflowBehavior::None) {
    auto op =
        create<cir::BinOp>(loc, lhs.getType(), cir::BinOpKind::Add, lhs, rhs);
    op.setNoUnsignedWrap(
        llvm::to_underlying(ob & OverflowBehavior::NoUnsignedWrap));
    op.setNoSignedWrap(
        llvm::to_underlying(ob & OverflowBehavior::NoSignedWrap));
    op.setSaturated(llvm::to_underlying(ob & OverflowBehavior::Saturated));
    return op;
  }

  mlir::Value createNSWAdd(mlir::Location loc, mlir::Value lhs,
                           mlir::Value rhs) {
    return createAdd(loc, lhs, rhs, OverflowBehavior::NoSignedWrap);
  }

  mlir::Value createNUWAdd(mlir::Location loc, mlir::Value lhs,
                           mlir::Value rhs) {
    return createAdd(loc, lhs, rhs, OverflowBehavior::NoUnsignedWrap);
  }

  cir::CmpOp createCompare(mlir::Location loc, cir::CmpOpKind kind,
                           mlir::Value lhs, mlir::Value rhs) {
    return create<cir::CmpOp>(loc, getBoolTy(), kind, lhs, rhs);
  }

  mlir::Value createShift(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                          bool isShiftLeft) {
    return create<cir::ShiftOp>(loc, lhs.getType(), lhs, rhs, isShiftLeft);
  }

  mlir::Value createShift(mlir::Location loc, mlir::Value lhs,
                          const llvm::APInt &rhs, bool isShiftLeft) {
    return createShift(loc, lhs, getConstAPInt(loc, lhs.getType(), rhs),
                       isShiftLeft);
  }

  mlir::Value createShift(mlir::Location loc, mlir::Value lhs, unsigned bits,
                          bool isShiftLeft) {
    auto width = mlir::dyn_cast<cir::IntType>(lhs.getType()).getWidth();
    auto shift = llvm::APInt(width, bits);
    return createShift(loc, lhs, shift, isShiftLeft);
  }

  mlir::Value createShiftLeft(mlir::Location loc, mlir::Value lhs,
                              unsigned bits) {
    return createShift(loc, lhs, bits, true);
  }

  mlir::Value createShiftRight(mlir::Location loc, mlir::Value lhs,
                               unsigned bits) {
    return createShift(loc, lhs, bits, false);
  }

  mlir::Value createShiftLeft(mlir::Location loc, mlir::Value lhs,
                              mlir::Value rhs) {
    return createShift(loc, lhs, rhs, true);
  }

  mlir::Value createShiftRight(mlir::Location loc, mlir::Value lhs,
                               mlir::Value rhs) {
    return createShift(loc, lhs, rhs, false);
  }

  //
  // Block handling helpers
  // ----------------------
  //
  static OpBuilder::InsertPoint getBestAllocaInsertPoint(mlir::Block *block) {
    auto last =
        std::find_if(block->rbegin(), block->rend(), [](mlir::Operation &op) {
          // TODO: Add LabelOp missing feature here
          return mlir::isa<cir::AllocaOp>(&op);
        });

    if (last != block->rend())
      return OpBuilder::InsertPoint(block, ++mlir::Block::iterator(&*last));
    return OpBuilder::InsertPoint(block, block->begin());
  };

  //
  // Alignment and size helpers
  //

  // Note that mlir::IntegerType is used instead of cir::IntType here because we
  // don't need sign information for these to be useful, so keep it simple.

  // For 0 alignment, any overload of `getAlignmentAttr` returns an empty
  // attribute.
  mlir::IntegerAttr getAlignmentAttr(clang::CharUnits alignment) {
    return getAlignmentAttr(alignment.getQuantity());
  }

  mlir::IntegerAttr getAlignmentAttr(llvm::Align alignment) {
    return getAlignmentAttr(alignment.value());
  }

  mlir::IntegerAttr getAlignmentAttr(int64_t alignment) {
    return alignment ? getI64IntegerAttr(alignment) : mlir::IntegerAttr();
  }

  mlir::IntegerAttr getSizeFromCharUnits(clang::CharUnits size) {
    return getI64IntegerAttr(size.getQuantity());
  }

  /// Create a loop condition.
  cir::ConditionOp createCondition(mlir::Value condition) {
    return create<cir::ConditionOp>(condition.getLoc(), condition);
  }

  /// Create a yield operation.
  cir::YieldOp createYield(mlir::Location loc, mlir::ValueRange value = {}) {
    return create<cir::YieldOp>(loc, value);
  }
};

} // namespace cir

#endif
