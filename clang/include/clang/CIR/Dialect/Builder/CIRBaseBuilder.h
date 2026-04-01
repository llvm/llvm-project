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
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/Support/ErrorHandling.h"

#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/Location.h"
#include "aiir/IR/OperationSupport.h"
#include "aiir/IR/Types.h"

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

constexpr bool testFlag(OverflowBehavior ob, OverflowBehavior flag) {
  return (ob & flag) != OverflowBehavior::None;
}

class CIRBaseBuilderTy : public aiir::OpBuilder {

public:
  CIRBaseBuilderTy(aiir::AIIRContext &aiirContext)
      : aiir::OpBuilder(&aiirContext) {}
  CIRBaseBuilderTy(aiir::OpBuilder &builder) : aiir::OpBuilder(builder) {}

  aiir::Value getConstAPInt(aiir::Location loc, aiir::Type typ,
                            const llvm::APInt &val) {
    return cir::ConstantOp::create(*this, loc, cir::IntAttr::get(typ, val));
  }

  cir::ConstantOp getConstant(aiir::Location loc, aiir::TypedAttr attr) {
    return cir::ConstantOp::create(*this, loc, attr);
  }

  cir::ConstantOp getConstantInt(aiir::Location loc, aiir::Type ty,
                                 int64_t value) {
    return getConstant(loc, cir::IntAttr::get(ty, value));
  }

  aiir::Value getSignedInt(aiir::Location loc, int64_t val, unsigned numBits) {
    auto type = cir::IntType::get(getContext(), numBits, /*isSigned=*/true);
    return getConstAPInt(loc, type,
                         llvm::APInt(numBits, val, /*isSigned=*/true));
  }

  aiir::Value getUnsignedInt(aiir::Location loc, uint64_t val,
                             unsigned numBits) {
    auto type = cir::IntType::get(getContext(), numBits, /*isSigned=*/false);
    return getConstAPInt(loc, type, llvm::APInt(numBits, val));
  }

  // Creates constant null value for integral type ty.
  cir::ConstantOp getNullValue(aiir::Type ty, aiir::Location loc) {
    return getConstant(loc, getZeroInitAttr(ty));
  }

  aiir::TypedAttr getConstNullPtrAttr(aiir::Type t) {
    assert(aiir::isa<cir::PointerType>(t) && "expected cir.ptr");
    return getConstPtrAttr(t, 0);
  }

  aiir::TypedAttr getNullDataMemberAttr(cir::DataMemberType ty) {
    return cir::DataMemberAttr::get(ty);
  }

  aiir::TypedAttr getZeroInitAttr(aiir::Type ty) {
    if (aiir::isa<cir::IntType>(ty))
      return cir::IntAttr::get(ty, 0);
    if (cir::isAnyFloatingPointType(ty))
      return cir::FPAttr::getZero(ty);
    if (auto complexType = aiir::dyn_cast<cir::ComplexType>(ty))
      return cir::ZeroAttr::get(complexType);
    if (auto arrTy = aiir::dyn_cast<cir::ArrayType>(ty))
      return cir::ZeroAttr::get(arrTy);
    if (auto vecTy = aiir::dyn_cast<cir::VectorType>(ty))
      return cir::ZeroAttr::get(vecTy);
    if (auto ptrTy = aiir::dyn_cast<cir::PointerType>(ty))
      return getConstNullPtrAttr(ptrTy);
    if (auto recordTy = aiir::dyn_cast<cir::RecordType>(ty))
      return cir::ZeroAttr::get(recordTy);
    if (auto dataMemberTy = aiir::dyn_cast<cir::DataMemberType>(ty))
      return getNullDataMemberAttr(dataMemberTy);
    if (aiir::isa<cir::BoolType>(ty)) {
      return getFalseAttr();
    }
    llvm_unreachable("Zero initializer for given type is NYI");
  }

  cir::ConstantOp getBool(bool state, aiir::Location loc) {
    return cir::ConstantOp::create(*this, loc, getCIRBoolAttr(state));
  }
  cir::ConstantOp getFalse(aiir::Location loc) { return getBool(false, loc); }
  cir::ConstantOp getTrue(aiir::Location loc) { return getBool(true, loc); }

  cir::BoolType getBoolTy() { return cir::BoolType::get(getContext()); }
  cir::VoidType getVoidTy() { return cir::VoidType::get(getContext()); }

  cir::IntType getUIntNTy(int n) {
    return cir::IntType::get(getContext(), n, false);
  }

  static unsigned getCIRIntOrFloatBitWidth(aiir::Type eltTy) {
    if (auto intType = aiir::dyn_cast<cir::IntTypeInterface>(eltTy))
      return intType.getWidth();
    if (auto floatType = aiir::dyn_cast<cir::FPTypeInterface>(eltTy))
      return floatType.getWidth();

    llvm_unreachable("Unsupported type in getCIRIntOrFloatBitWidth");
  }
  cir::IntType getSIntNTy(int n) {
    return cir::IntType::get(getContext(), n, true);
  }

  cir::PointerType getPointerTo(aiir::Type ty) {
    return cir::PointerType::get(ty);
  }

  cir::PointerType getPointerTo(aiir::Type ty,
                                aiir::ptr::MemorySpaceAttrInterface as) {
    return cir::PointerType::get(ty, as);
  }

  cir::PointerType getPointerTo(aiir::Type ty, clang::LangAS langAS) {
    if (langAS == clang::LangAS::Default)
      return getPointerTo(ty);

    aiir::ptr::MemorySpaceAttrInterface addrSpaceAttr =
        cir::toCIRAddressSpaceAttr(*getContext(), langAS);
    return getPointerTo(ty, addrSpaceAttr);
  }

  cir::PointerType getVoidPtrTy(clang::LangAS langAS = clang::LangAS::Default) {
    return getPointerTo(cir::VoidType::get(getContext()), langAS);
  }

  cir::PointerType getVoidPtrTy(aiir::ptr::MemorySpaceAttrInterface as) {
    return getPointerTo(cir::VoidType::get(getContext()), as);
  }

  cir::MethodAttr getMethodAttr(cir::MethodType ty, cir::FuncOp methodFuncOp) {
    auto methodFuncSymbolRef = aiir::FlatSymbolRefAttr::get(methodFuncOp);
    return cir::MethodAttr::get(ty, methodFuncSymbolRef);
  }

  cir::MethodAttr getNullMethodAttr(cir::MethodType ty) {
    return cir::MethodAttr::get(ty);
  }

  cir::BoolAttr getCIRBoolAttr(bool state) {
    return cir::BoolAttr::get(getContext(), state);
  }

  cir::BoolAttr getTrueAttr() { return getCIRBoolAttr(true); }
  cir::BoolAttr getFalseAttr() { return getCIRBoolAttr(false); }

  aiir::Value createComplexCreate(aiir::Location loc, aiir::Value real,
                                  aiir::Value imag) {
    auto resultComplexTy = cir::ComplexType::get(real.getType());
    return cir::ComplexCreateOp::create(*this, loc, resultComplexTy, real,
                                        imag);
  }

  aiir::Value createComplexReal(aiir::Location loc, aiir::Value operand) {
    auto resultType = operand.getType();
    if (auto complexResultType = aiir::dyn_cast<cir::ComplexType>(resultType))
      resultType = complexResultType.getElementType();
    return cir::ComplexRealOp::create(*this, loc, resultType, operand);
  }

  aiir::Value createComplexImag(aiir::Location loc, aiir::Value operand) {
    auto resultType = operand.getType();
    if (auto complexResultType = aiir::dyn_cast<cir::ComplexType>(resultType))
      resultType = complexResultType.getElementType();
    return cir::ComplexImagOp::create(*this, loc, resultType, operand);
  }

  cir::LoadOp createLoad(aiir::Location loc, aiir::Value ptr,
                         bool isVolatile = false, uint64_t alignment = 0) {
    aiir::IntegerAttr alignmentAttr = getAlignmentAttr(alignment);
    return cir::LoadOp::create(*this, loc, ptr, /*isDeref=*/false, isVolatile,
                               alignmentAttr, cir::SyncScopeKindAttr{},
                               cir::MemOrderAttr{});
  }

  aiir::Value createAlignedLoad(aiir::Location loc, aiir::Value ptr,
                                uint64_t alignment) {
    return createLoad(loc, ptr, /*isVolatile=*/false, alignment);
  }

  aiir::Value createNot(aiir::Location loc, aiir::Value value) {
    return cir::NotOp::create(*this, loc, value);
  }

  aiir::Value createNot(aiir::Value value) {
    return createNot(value.getLoc(), value);
  }

  /// Create a do-while operation.
  cir::DoWhileOp createDoWhile(
      aiir::Location loc,
      llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)> condBuilder,
      llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)> bodyBuilder) {
    return cir::DoWhileOp::create(*this, loc, condBuilder, bodyBuilder);
  }

  /// Create a while operation.
  cir::WhileOp createWhile(
      aiir::Location loc,
      llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)> condBuilder,
      llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)> bodyBuilder) {
    return cir::WhileOp::create(*this, loc, condBuilder, bodyBuilder);
  }

  /// Create a for operation.
  cir::ForOp createFor(
      aiir::Location loc,
      llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)> condBuilder,
      llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)> bodyBuilder,
      llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)> stepBuilder) {
    return cir::ForOp::create(*this, loc, condBuilder, bodyBuilder,
                              stepBuilder);
  }

  /// Create a break operation.
  cir::BreakOp createBreak(aiir::Location loc) {
    return cir::BreakOp::create(*this, loc);
  }

  /// Create a continue operation.
  cir::ContinueOp createContinue(aiir::Location loc) {
    return cir::ContinueOp::create(*this, loc);
  }

  aiir::Value createInc(aiir::Location loc, aiir::Value input,
                        bool nsw = false) {
    return cir::IncOp::create(*this, loc, input, nsw);
  }

  aiir::Value createDec(aiir::Location loc, aiir::Value input,
                        bool nsw = false) {
    return cir::DecOp::create(*this, loc, input, nsw);
  }

  aiir::Value createMinus(aiir::Location loc, aiir::Value input,
                          bool nsw = false) {
    return cir::MinusOp::create(*this, loc, input, nsw);
  }

  aiir::TypedAttr getConstPtrAttr(aiir::Type type, int64_t value) {
    return cir::ConstPtrAttr::get(type, getI64IntegerAttr(value));
  }

  aiir::Value createAlloca(aiir::Location loc, cir::PointerType addrType,
                           aiir::Type type, llvm::StringRef name,
                           aiir::IntegerAttr alignment,
                           aiir::Value dynAllocSize) {
    return cir::AllocaOp::create(*this, loc, addrType, type, name, alignment,
                                 dynAllocSize);
  }

  aiir::Value createAlloca(aiir::Location loc, cir::PointerType addrType,
                           aiir::Type type, llvm::StringRef name,
                           clang::CharUnits alignment,
                           aiir::Value dynAllocSize) {
    aiir::IntegerAttr alignmentAttr = getAlignmentAttr(alignment);
    return createAlloca(loc, addrType, type, name, alignmentAttr, dynAllocSize);
  }

  aiir::Value createAlloca(aiir::Location loc, cir::PointerType addrType,
                           aiir::Type type, llvm::StringRef name,
                           aiir::IntegerAttr alignment) {
    return cir::AllocaOp::create(*this, loc, addrType, type, name, alignment);
  }

  aiir::Value createAlloca(aiir::Location loc, cir::PointerType addrType,
                           aiir::Type type, llvm::StringRef name,
                           clang::CharUnits alignment) {
    aiir::IntegerAttr alignmentAttr = getAlignmentAttr(alignment);
    return createAlloca(loc, addrType, type, name, alignmentAttr);
  }

  /// Get constant address of a global variable as an AIIR attribute.
  /// This wrapper infers the attribute type through the global op.
  cir::GlobalViewAttr getGlobalViewAttr(cir::GlobalOp globalOp,
                                        aiir::ArrayAttr indices = {}) {
    cir::PointerType type = getPointerTo(globalOp.getSymType());
    return getGlobalViewAttr(type, globalOp, indices);
  }

  /// Get constant address of a global variable as an AIIR attribute.
  cir::GlobalViewAttr getGlobalViewAttr(cir::PointerType type,
                                        cir::GlobalOp globalOp,
                                        aiir::ArrayAttr indices = {}) {
    auto symbol = aiir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
    return cir::GlobalViewAttr::get(type, symbol, indices);
  }

  /// Get constant address of a global variable as an AIIR attribute.
  /// This overload converts raw int64_t indices to an ArrayAttr.
  cir::GlobalViewAttr getGlobalViewAttr(cir::PointerType type,
                                        cir::GlobalOp globalOp,
                                        llvm::ArrayRef<int64_t> indices) {
    llvm::SmallVector<aiir::Attribute> attrs;
    for (int64_t ind : indices)
      attrs.push_back(getI64IntegerAttr(ind));
    aiir::ArrayAttr arAttr = aiir::ArrayAttr::get(getContext(), attrs);
    return getGlobalViewAttr(type, globalOp, arAttr);
  }

  aiir::Value createGetGlobal(aiir::Location loc, cir::GlobalOp global,
                              bool threadLocal = false) {
    assert(!cir::MissingFeatures::addressSpace());
    return cir::GetGlobalOp::create(*this, loc,
                                    getPointerTo(global.getSymType()),
                                    global.getSymNameAttr(), threadLocal);
  }

  aiir::Value createGetGlobal(cir::GlobalOp global, bool threadLocal = false) {
    return createGetGlobal(global.getLoc(), global, threadLocal);
  }

  /// Create a copy with inferred length.
  cir::CopyOp createCopy(aiir::Value dst, aiir::Value src,
                         bool isVolatile = false) {
    return cir::CopyOp::create(*this, dst.getLoc(), dst, src, isVolatile);
  }

  cir::StoreOp createStore(aiir::Location loc, aiir::Value val, aiir::Value dst,
                           bool isVolatile = false,
                           aiir::IntegerAttr align = {},
                           cir::SyncScopeKindAttr scope = {},
                           cir::MemOrderAttr order = {}) {
    if (aiir::cast<cir::PointerType>(dst.getType()).getPointee() !=
        val.getType())
      dst = createPtrBitcast(dst, val.getType());
    return cir::StoreOp::create(*this, loc, val, dst, isVolatile, align, scope,
                                order);
  }

  /// Emit a load from an boolean flag variable.
  cir::LoadOp createFlagLoad(aiir::Location loc, aiir::Value addr) {
    aiir::Type boolTy = getBoolTy();
    if (boolTy != aiir::cast<cir::PointerType>(addr.getType()).getPointee())
      addr = createPtrBitcast(addr, boolTy);
    return createLoad(loc, addr, /*isVolatile=*/false, /*alignment=*/1);
  }

  cir::StoreOp createFlagStore(aiir::Location loc, bool val, aiir::Value dst) {
    aiir::Value flag = getBool(val, loc);
    return CIRBaseBuilderTy::createStore(loc, flag, dst);
  }

  [[nodiscard]] cir::GlobalOp
  createGlobal(aiir::ModuleOp aiirModule, aiir::Location loc,
               aiir::StringRef name, aiir::Type type, bool isConstant,
               cir::GlobalLinkageKind linkage,
               aiir::ptr::MemorySpaceAttrInterface addrSpace) {
    aiir::OpBuilder::InsertionGuard guard(*this);
    setInsertionPointToStart(aiirModule.getBody());
    return cir::GlobalOp::create(*this, loc, name, type, isConstant, addrSpace,
                                 linkage);
  }

  cir::GetMemberOp createGetMember(aiir::Location loc, aiir::Type resultTy,
                                   aiir::Value base, llvm::StringRef name,
                                   unsigned index) {
    return cir::GetMemberOp::create(*this, loc, resultTy, base, name, index);
  }

  aiir::Value createDummyValue(aiir::Location loc, aiir::Type type,
                               clang::CharUnits alignment) {
    aiir::IntegerAttr alignmentAttr = getAlignmentAttr(alignment);
    auto addr = createAlloca(loc, getPointerTo(type), type, {}, alignmentAttr);
    return cir::LoadOp::create(*this, loc, addr, /*isDeref=*/false,
                               /*isVolatile=*/false, alignmentAttr,
                               /*sync_scope=*/{}, /*mem_order=*/{});
  }

  cir::PtrStrideOp createPtrStride(aiir::Location loc, aiir::Value base,
                                   aiir::Value stride) {
    return cir::PtrStrideOp::create(*this, loc, base.getType(), base, stride);
  }

  //===--------------------------------------------------------------------===//
  // Call operators
  //===--------------------------------------------------------------------===//

  cir::CallOp createCallOp(aiir::Location loc, aiir::SymbolRefAttr callee,
                           aiir::Type returnType, aiir::ValueRange operands,
                           llvm::ArrayRef<aiir::NamedAttribute> attrs = {},
                           llvm::ArrayRef<aiir::NamedAttrList> argAttrs = {},
                           llvm::ArrayRef<aiir::NamedAttribute> resAttrs = {}) {
    auto op = cir::CallOp::create(*this, loc, callee, returnType, operands);
    op->setAttrs(attrs);

    if (!argAttrs.empty()) {
      llvm::SmallVector<aiir::Attribute> argDictAttrs;
      argDictAttrs.reserve(argAttrs.size());

      llvm::transform(
          argAttrs, std::back_inserter(argDictAttrs),
          [this](llvm::ArrayRef<aiir::NamedAttribute> singleArgAttrs) {
            return aiir::DictionaryAttr::get(getContext(), singleArgAttrs);
          });

      op.setArgAttrsAttr(aiir::ArrayAttr::get(getContext(), argDictAttrs));
    }

    if (!resAttrs.empty()) {
      auto resultDictAttr = aiir::DictionaryAttr::get(getContext(), resAttrs);
      op.setResAttrsAttr(aiir::ArrayAttr::get(getContext(), resultDictAttr));
    }
    return op;
  }

  cir::CallOp createCallOp(aiir::Location loc, cir::FuncOp callee,
                           aiir::ValueRange operands,
                           llvm::ArrayRef<aiir::NamedAttribute> attrs = {},
                           llvm::ArrayRef<aiir::NamedAttrList> argAttrs = {},
                           llvm::ArrayRef<aiir::NamedAttribute> resAttrs = {}) {
    return createCallOp(loc, aiir::SymbolRefAttr::get(callee),
                        callee.getFunctionType().getReturnType(), operands,
                        attrs, argAttrs, resAttrs);
  }

  cir::CallOp
  createIndirectCallOp(aiir::Location loc, aiir::Value indirectTarget,
                       cir::FuncType funcType, aiir::ValueRange operands,
                       llvm::ArrayRef<aiir::NamedAttribute> attrs = {},
                       llvm::ArrayRef<aiir::NamedAttrList> argAttrs = {},
                       llvm::ArrayRef<aiir::NamedAttribute> resAttrs = {}) {
    llvm::SmallVector<aiir::Value> resOperands{indirectTarget};
    resOperands.append(operands.begin(), operands.end());

    return createCallOp(loc, aiir::SymbolRefAttr(), funcType.getReturnType(),
                        resOperands, attrs, argAttrs, resAttrs);
  }

  cir::CallOp createCallOp(aiir::Location loc, aiir::SymbolRefAttr callee,
                           aiir::ValueRange operands = aiir::ValueRange(),
                           llvm::ArrayRef<aiir::NamedAttribute> attrs = {},
                           llvm::ArrayRef<aiir::NamedAttrList> argAttrs = {},
                           llvm::ArrayRef<aiir::NamedAttribute> resAttrs = {}) {
    return createCallOp(loc, callee, cir::VoidType(), operands, attrs, argAttrs,
                        resAttrs);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  aiir::Value createCast(aiir::Location loc, cir::CastKind kind,
                         aiir::Value src, aiir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return cir::CastOp::create(*this, loc, newTy, kind, src);
  }

  aiir::Value createCast(cir::CastKind kind, aiir::Value src,
                         aiir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return createCast(src.getLoc(), kind, src, newTy);
  }

  aiir::Value createIntCast(aiir::Value src, aiir::Type newTy) {
    return createCast(cir::CastKind::integral, src, newTy);
  }

  aiir::Value createIntToPtr(aiir::Value src, aiir::Type newTy) {
    return createCast(cir::CastKind::int_to_ptr, src, newTy);
  }

  aiir::Value createPtrToInt(aiir::Value src, aiir::Type newTy) {
    return createCast(cir::CastKind::ptr_to_int, src, newTy);
  }

  aiir::Value createPtrToBoolCast(aiir::Value v) {
    return createCast(cir::CastKind::ptr_to_bool, v, getBoolTy());
  }

  aiir::Value createBoolToInt(aiir::Value src, aiir::Type newTy) {
    return createCast(cir::CastKind::bool_to_int, src, newTy);
  }

  aiir::Value createBitcast(aiir::Value src, aiir::Type newTy) {
    return createCast(cir::CastKind::bitcast, src, newTy);
  }

  aiir::Value createBitcast(aiir::Location loc, aiir::Value src,
                            aiir::Type newTy) {
    return createCast(loc, cir::CastKind::bitcast, src, newTy);
  }

  aiir::Value createPtrBitcast(aiir::Value src, aiir::Type newPointeeTy) {
    assert(aiir::isa<cir::PointerType>(src.getType()) && "expected ptr src");
    return createBitcast(src, getPointerTo(newPointeeTy));
  }

  aiir::Value createPtrIsNull(aiir::Value ptr) {
    aiir::Value nullPtr = getNullPtr(ptr.getType(), ptr.getLoc());
    return createCompare(ptr.getLoc(), cir::CmpOpKind::eq, ptr, nullPtr);
  }

  aiir::Value createPtrIsNotNull(aiir::Value ptr) {
    aiir::Value nullPtr = getNullPtr(ptr.getType(), ptr.getLoc());
    return createCompare(ptr.getLoc(), cir::CmpOpKind::ne, ptr, nullPtr);
  }

  aiir::Value createAddrSpaceCast(aiir::Location loc, aiir::Value src,
                                  aiir::Type newTy) {
    return createCast(loc, cir::CastKind::address_space, src, newTy);
  }

  aiir::Value createAddrSpaceCast(aiir::Value src, aiir::Type newTy) {
    return createAddrSpaceCast(src.getLoc(), src, newTy);
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  aiir::Value createExtractElement(aiir::Location loc, aiir::Value vec,
                                   uint64_t idx) {
    aiir::Value idxVal =
        getConstAPInt(loc, getUIntNTy(64), llvm::APInt(64, idx));
    return cir::VecExtractOp::create(*this, loc, vec, idxVal);
  }

  aiir::Value createInsertElement(aiir::Location loc, aiir::Value vec,
                                  aiir::Value newElt, uint64_t idx) {
    aiir::Value idxVal =
        getConstAPInt(loc, getUIntNTy(64), llvm::APInt(64, idx));
    return cir::VecInsertOp::create(*this, loc, vec, newElt, idxVal);
  }

  cir::SignBitOp createSignBit(aiir::Location loc, aiir::Value val) {
    auto resTy = cir::BoolType::get(getContext());
    return cir::SignBitOp::create(*this, loc, resTy, val);
  }

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  aiir::Value createLowBitsSet(aiir::Location loc, unsigned size,
                               unsigned bits) {
    llvm::APInt val = llvm::APInt::getLowBitsSet(size, bits);
    auto type = cir::IntType::get(getContext(), size, /*isSigned=*/false);
    return getConstAPInt(loc, type, val);
  }

  aiir::Value createAnd(aiir::Location loc, aiir::Value lhs, aiir::Value rhs) {
    return cir::AndOp::create(*this, loc, lhs, rhs);
  }

  aiir::Value createOr(aiir::Location loc, aiir::Value lhs, aiir::Value rhs) {
    return cir::OrOp::create(*this, loc, lhs, rhs);
  }

  aiir::Value createSelect(aiir::Location loc, aiir::Value condition,
                           aiir::Value trueValue, aiir::Value falseValue) {
    assert(trueValue.getType() == falseValue.getType() &&
           "trueValue and falseValue should have the same type");
    return cir::SelectOp::create(*this, loc, trueValue.getType(), condition,
                                 trueValue, falseValue);
  }

  aiir::Value createLogicalAnd(aiir::Location loc, aiir::Value lhs,
                               aiir::Value rhs) {
    return createSelect(loc, lhs, rhs, getBool(false, loc));
  }

  aiir::Value createLogicalOr(aiir::Location loc, aiir::Value lhs,
                              aiir::Value rhs) {
    return createSelect(loc, lhs, getBool(true, loc), rhs);
  }

  aiir::Value createMul(aiir::Location loc, aiir::Value lhs, aiir::Value rhs,
                        OverflowBehavior ob = OverflowBehavior::None) {
    auto op = cir::MulOp::create(*this, loc, lhs, rhs);
    op.setNoUnsignedWrap(testFlag(ob, OverflowBehavior::NoUnsignedWrap));
    op.setNoSignedWrap(testFlag(ob, OverflowBehavior::NoSignedWrap));
    return op;
  }
  aiir::Value createNSWMul(aiir::Location loc, aiir::Value lhs,
                           aiir::Value rhs) {
    return createMul(loc, lhs, rhs, OverflowBehavior::NoSignedWrap);
  }
  aiir::Value createNUWAMul(aiir::Location loc, aiir::Value lhs,
                            aiir::Value rhs) {
    return createMul(loc, lhs, rhs, OverflowBehavior::NoUnsignedWrap);
  }

  aiir::Value createSub(aiir::Location loc, aiir::Value lhs, aiir::Value rhs,
                        OverflowBehavior ob = OverflowBehavior::None) {
    auto op = cir::SubOp::create(*this, loc, lhs, rhs);
    op.setNoUnsignedWrap(testFlag(ob, OverflowBehavior::NoUnsignedWrap));
    op.setNoSignedWrap(testFlag(ob, OverflowBehavior::NoSignedWrap));
    op.setSaturated(testFlag(ob, OverflowBehavior::Saturated));
    return op;
  }

  aiir::Value createNSWSub(aiir::Location loc, aiir::Value lhs,
                           aiir::Value rhs) {
    return createSub(loc, lhs, rhs, OverflowBehavior::NoSignedWrap);
  }

  aiir::Value createNUWSub(aiir::Location loc, aiir::Value lhs,
                           aiir::Value rhs) {
    return createSub(loc, lhs, rhs, OverflowBehavior::NoUnsignedWrap);
  }

  aiir::Value createAdd(aiir::Location loc, aiir::Value lhs, aiir::Value rhs,
                        OverflowBehavior ob = OverflowBehavior::None) {
    auto op = cir::AddOp::create(*this, loc, lhs, rhs);
    op.setNoUnsignedWrap(testFlag(ob, OverflowBehavior::NoUnsignedWrap));
    op.setNoSignedWrap(testFlag(ob, OverflowBehavior::NoSignedWrap));
    op.setSaturated(testFlag(ob, OverflowBehavior::Saturated));
    return op;
  }

  aiir::Value createNSWAdd(aiir::Location loc, aiir::Value lhs,
                           aiir::Value rhs) {
    return createAdd(loc, lhs, rhs, OverflowBehavior::NoSignedWrap);
  }

  aiir::Value createNUWAdd(aiir::Location loc, aiir::Value lhs,
                           aiir::Value rhs) {
    return createAdd(loc, lhs, rhs, OverflowBehavior::NoUnsignedWrap);
  }

  aiir::Value createDiv(aiir::Location loc, aiir::Value lhs, aiir::Value rhs) {
    return cir::DivOp::create(*this, loc, lhs, rhs);
  }

  aiir::Value createRem(aiir::Location loc, aiir::Value lhs, aiir::Value rhs) {
    return cir::RemOp::create(*this, loc, lhs, rhs);
  }

  aiir::Value createXor(aiir::Location loc, aiir::Value lhs, aiir::Value rhs) {
    return cir::XorOp::create(*this, loc, lhs, rhs);
  }

  aiir::Value createMax(aiir::Location loc, aiir::Value lhs, aiir::Value rhs) {
    return cir::MaxOp::create(*this, loc, lhs, rhs);
  }

  cir::CmpOp createCompare(aiir::Location loc, cir::CmpOpKind kind,
                           aiir::Value lhs, aiir::Value rhs) {
    return cir::CmpOp::create(*this, loc, kind, lhs, rhs);
  }

  cir::VecCmpOp createVecCompare(aiir::Location loc, cir::CmpOpKind kind,
                                 aiir::Value lhs, aiir::Value rhs) {
    VectorType vecCast = aiir::cast<VectorType>(lhs.getType());
    IntType integralTy =
        getSIntNTy(getCIRIntOrFloatBitWidth(vecCast.getElementType()));
    VectorType integralVecTy =
        cir::VectorType::get(integralTy, vecCast.getSize());
    return cir::VecCmpOp::create(*this, loc, integralVecTy, kind, lhs, rhs);
  }

  aiir::Value createIsNaN(aiir::Location loc, aiir::Value operand) {
    return createCompare(loc, cir::CmpOpKind::ne, operand, operand);
  }

  aiir::Value createShift(aiir::Location loc, aiir::Value lhs, aiir::Value rhs,
                          bool isShiftLeft) {
    return cir::ShiftOp::create(*this, loc, lhs.getType(), lhs, rhs,
                                isShiftLeft);
  }

  aiir::Value createShift(aiir::Location loc, aiir::Value lhs,
                          const llvm::APInt &rhs, bool isShiftLeft) {
    return createShift(loc, lhs, getConstAPInt(loc, lhs.getType(), rhs),
                       isShiftLeft);
  }

  aiir::Value createShift(aiir::Location loc, aiir::Value lhs, unsigned bits,
                          bool isShiftLeft) {
    auto width = aiir::dyn_cast<cir::IntType>(lhs.getType()).getWidth();
    auto shift = llvm::APInt(width, bits);
    return createShift(loc, lhs, shift, isShiftLeft);
  }

  aiir::Value createShiftLeft(aiir::Location loc, aiir::Value lhs,
                              unsigned bits) {
    return createShift(loc, lhs, bits, true);
  }

  aiir::Value createShiftRight(aiir::Location loc, aiir::Value lhs,
                               unsigned bits) {
    return createShift(loc, lhs, bits, false);
  }

  aiir::Value createShiftLeft(aiir::Location loc, aiir::Value lhs,
                              aiir::Value rhs) {
    return createShift(loc, lhs, rhs, true);
  }

  aiir::Value createShiftRight(aiir::Location loc, aiir::Value lhs,
                               aiir::Value rhs) {
    return createShift(loc, lhs, rhs, false);
  }

  //
  // Block handling helpers
  // ----------------------
  //
  static OpBuilder::InsertPoint getBestAllocaInsertPoint(aiir::Block *block) {
    auto last =
        std::find_if(block->rbegin(), block->rend(), [](aiir::Operation &op) {
          return aiir::isa<cir::AllocaOp, cir::LabelOp>(&op);
        });

    if (last != block->rend())
      return OpBuilder::InsertPoint(block, ++aiir::Block::iterator(&*last));
    return OpBuilder::InsertPoint(block, block->begin());
  };

  //
  // Alignment and size helpers
  //

  // Note that aiir::IntegerType is used instead of cir::IntType here because we
  // don't need sign information for these to be useful, so keep it simple.

  // For 0 alignment, any overload of `getAlignmentAttr` returns an empty
  // attribute.
  aiir::IntegerAttr getAlignmentAttr(clang::CharUnits alignment) {
    return getAlignmentAttr(alignment.getQuantity());
  }

  aiir::IntegerAttr getAlignmentAttr(llvm::Align alignment) {
    return getAlignmentAttr(alignment.value());
  }

  aiir::IntegerAttr getAlignmentAttr(int64_t alignment) {
    return alignment ? getI64IntegerAttr(alignment) : aiir::IntegerAttr();
  }

  aiir::IntegerAttr getSizeFromCharUnits(clang::CharUnits size) {
    return getI64IntegerAttr(size.getQuantity());
  }

  // Creates constant nullptr for pointer type ty.
  cir::ConstantOp getNullPtr(aiir::Type ty, aiir::Location loc) {
    assert(!cir::MissingFeatures::targetCodeGenInfoGetNullPointer());
    return cir::ConstantOp::create(*this, loc, getConstPtrAttr(ty, 0));
  }

  /// Create a loop condition.
  cir::ConditionOp createCondition(aiir::Value condition) {
    return cir::ConditionOp::create(*this, condition.getLoc(), condition);
  }

  /// Create a yield operation.
  cir::YieldOp createYield(aiir::Location loc, aiir::ValueRange value = {}) {
    return cir::YieldOp::create(*this, loc, value);
  }

  struct GetMethodResults {
    aiir::Value callee;
    aiir::Value adjustedThis;
  };

  GetMethodResults createGetMethod(aiir::Location loc, aiir::Value method,
                                   aiir::Value objectPtr) {
    // Build the callee function type.
    auto methodFuncTy =
        aiir::cast<cir::MethodType>(method.getType()).getMemberFuncTy();
    auto methodFuncInputTypes = methodFuncTy.getInputs();

    auto objectPtrTy = aiir::cast<cir::PointerType>(objectPtr.getType());
    aiir::Type adjustedThisTy = getVoidPtrTy(objectPtrTy.getAddrSpace());

    llvm::SmallVector<aiir::Type> calleeFuncInputTypes{adjustedThisTy};
    calleeFuncInputTypes.insert(calleeFuncInputTypes.end(),
                                methodFuncInputTypes.begin(),
                                methodFuncInputTypes.end());
    cir::FuncType calleeFuncTy =
        methodFuncTy.clone(calleeFuncInputTypes, methodFuncTy.getReturnType());
    // TODO(cir): consider the address space of the callee.
    assert(!cir::MissingFeatures::addressSpace());
    cir::PointerType calleeTy = getPointerTo(calleeFuncTy);

    auto op = cir::GetMethodOp::create(*this, loc, calleeTy, adjustedThisTy,
                                       method, objectPtr);
    return {op.getCallee(), op.getAdjustedThis()};
  }
};

} // namespace cir

#endif
