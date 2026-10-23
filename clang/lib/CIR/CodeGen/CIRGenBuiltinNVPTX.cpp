//===---- CIRGenBuiltinNVPTX.cpp - Emit CIR for NVPTX builtins ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit NVPTX Builtin calls.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

#include "mlir/IR/Value.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/NVPTXAddrSpace.h"

#include <algorithm>

using namespace clang;
using namespace clang::CIRGen;

static mlir::Value makeLdu(CIRGenFunction &cgf, const CallExpr *expr,
                           llvm::StringRef intrinsicName) {
  auto &builder = cgf.getBuilder();
  Address ptr = cgf.emitPointerWithAlignment(expr->getArg(0));
  QualType argType = expr->getArg(0)->getType();
  mlir::Type elemTy = cgf.convertTypeForMem(argType->getPointeeType());
  clang::CharUnits align = ptr.getAlignment();
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());
  mlir::Value alignVal =
      builder.getConstantInt(loc, builder.getSInt32Ty(), align.getQuantity());
  return cir::LLVMIntrinsicCallOp::create(
             builder, loc, builder.getStringAttr(intrinsicName), elemTy,
             {ptr.emitRawPointer(), alignVal})
      .getResult();
}

static mlir::Value makeLdg(CIRGenFunction &cgf, const CallExpr *expr) {
  auto &builder = cgf.getBuilder();
  Address ptr = cgf.emitPointerWithAlignment(expr->getArg(0));
  QualType argType = expr->getArg(0)->getType();
  mlir::Type elemTy = cgf.convertTypeForMem(argType->getPointeeType());
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());

  // Use addrspace(1) for NVPTX ADDRESS_SPACE_GLOBAL.
  mlir::Type globalPtrTy = cir::PointerType::get(
      elemTy, cir::TargetAddressSpaceAttr::get(
                  builder.getContext(), llvm::NVPTXAS::ADDRESS_SPACE_GLOBAL));
  mlir::Value asc =
      builder.createAddrSpaceCast(loc, ptr.getPointer(), globalPtrTy);
  cir::LoadOp load =
      builder.createAlignedLoad(loc, elemTy, asc, ptr.getAlignment());
  load.setInvariant(true);
  return load.getResult();
}

/// Emit a CIR LLVMIntrinsicCallOp for a unary NVVM intrinsic.
/// The result type is inferred from the single argument.
static mlir::Value emitUnaryNVVMIntrinsic(CIRGenFunction &cgf,
                                          const CallExpr *expr,
                                          llvm::StringRef intrinsicName) {
  auto &builder = cgf.getBuilder();
  mlir::Value arg = cgf.emitScalarExpr(expr->getArg(0));
  return cir::LLVMIntrinsicCallOp::create(
             builder, cgf.getLoc(expr->getExprLoc()),
             builder.getStringAttr(intrinsicName), arg.getType(), {arg})
      .getResult();
}

/// Emit a CIR LLVMIntrinsicCallOp for an NVVM fadd/fmul intrinsic, which takes
/// the rounding mode as a trailing operand.
static mlir::Value emitNVVMFPArith(CIRGenFunction &cgf, const CallExpr *expr,
                                   llvm::StringRef intrinsicName,
                                   llvm::APFloat::roundingMode rm) {
  auto &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());
  mlir::Value lhs = cgf.emitScalarExpr(expr->getArg(0));
  mlir::Value rhs = cgf.emitScalarExpr(expr->getArg(1));
  mlir::Value rnd =
      builder.getConstInt(loc, builder.getSInt32Ty(), static_cast<int>(rm));
  return cir::LLVMIntrinsicCallOp::create(builder, loc,
                                          builder.getStringAttr(intrinsicName),
                                          lhs.getType(), {lhs, rhs, rnd})
      .getResult();
}

/// Emit a CIR LLVMIntrinsicCallOp for an NVVM fma intrinsic, which takes the
/// rounding mode as a trailing operand.
static mlir::Value emitNVVMFMA(CIRGenFunction &cgf, const CallExpr *expr,
                               llvm::StringRef intrinsicName,
                               llvm::APFloat::roundingMode rm) {
  auto &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());
  mlir::Value a = cgf.emitScalarExpr(expr->getArg(0));
  mlir::Value b = cgf.emitScalarExpr(expr->getArg(1));
  mlir::Value c = cgf.emitScalarExpr(expr->getArg(2));
  mlir::Value rnd =
      builder.getConstInt(loc, builder.getSInt32Ty(), static_cast<int>(rm));
  return cir::LLVMIntrinsicCallOp::create(builder, loc,
                                          builder.getStringAttr(intrinsicName),
                                          a.getType(), {a, b, c, rnd})
      .getResult();
}

static mlir::Value emitBar0Reduction(CIRGenFunction &cgf, const CallExpr *expr,
                                     llvm::StringRef intrinsicName,
                                     bool returnsPred) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());
  mlir::Type si32Ty = builder.getSInt32Ty();
  mlir::Value zero = builder.getNullValue(si32Ty, loc);
  mlir::Value pred = builder.createCompare(
      loc, cir::CmpOpKind::ne, cgf.emitScalarExpr(expr->getArg(0)), zero);
  mlir::Type resultTy = returnsPred ? mlir::Type(builder.getBoolTy()) : si32Ty;
  mlir::Value result = builder.emitIntrinsicCallOp(
      loc, intrinsicName, resultTy, mlir::ValueRange{zero, pred});
  if (returnsPred)
    result = builder.createBoolToInt(result, si32Ty);
  return result;
}

static mlir::Value makeScopedAtomicRMW(CIRGenFunction &cgf,
                                       const CallExpr *expr,
                                       cir::AtomicFetchKind kind,
                                       cir::SyncScopeKind scope) {
  auto &builder = cgf.getBuilder();
  Address destAddr = cgf.emitPointerWithAlignment(expr->getArg(0));
  mlir::Value destValue = destAddr.emitRawPointer();
  mlir::Value val = cgf.emitScalarExpr(expr->getArg(1));
  auto rmwi = cir::AtomicFetchOp::create(
      builder, cgf.getLoc(expr->getSourceRange()), destValue, val, kind,
      cir::MemOrder::Relaxed, scope, /*is_volatile=*/false,
      /*fetch_first=*/true);
  return rmwi->getResult(0);
}

static mlir::Value makeScopedAtomicXchg(CIRGenFunction &cgf,
                                        const CallExpr *expr,
                                        cir::SyncScopeKind scope) {
  auto &builder = cgf.getBuilder();
  Address destAddr = cgf.emitPointerWithAlignment(expr->getArg(0));
  mlir::Value destValue = destAddr.emitRawPointer();
  mlir::Value val = cgf.emitScalarExpr(expr->getArg(1));
  auto xchg = cir::AtomicXchgOp::create(
      builder, cgf.getLoc(expr->getSourceRange()), destValue, val,
      cir::MemOrder::Relaxed, scope, /*is_volatile=*/false);
  return xchg.getResult();
}

// FP16 WMMA fragments are stored in i32 slots, but their intrinsic register
// representation is <2 x half>. FP32 fragments use float in both
// representations.
static mlir::Type getWMMARegisterType(mlir::Type storageType) {
  if (mlir::isa<cir::SingleType>(storageType))
    return storageType;
  return cir::VectorType::getFixed(cir::FP16Type::get(storageType.getContext()),
                                   2);
}

static cir::StructType getWMMAResultType(CIRGenBuilderTy &builder,
                                         mlir::Type registerType,
                                         unsigned numRegisters) {
  llvm::SmallVector<mlir::Type, 8> members(numRegisters, registerType);
  return builder.getAnonRecordTy(members, /*packed=*/false,
                                 cir::RecordType::getAllDataKinds(members));
}

static void loadWMMAFragment(CIRGenBuilderTy &builder, mlir::Location loc,
                             Address src, unsigned numRegisters,
                             llvm::SmallVectorImpl<mlir::Value> &values) {
  mlir::Type registerType = getWMMARegisterType(src.getElementType());
  // Every slot is four bytes, even when the fragment contains packed halves.
  CharUnits alignment =
      std::min(src.getAlignment(), CharUnits::fromQuantity(4));
  for (unsigned i = 0; i < numRegisters; ++i) {
    mlir::Value ptr = builder.createPtrStride(loc, src.emitRawPointer(),
                                              builder.getSInt32(i, loc));
    mlir::Value value =
        builder.createAlignedLoad(loc, src.getElementType(), ptr, alignment);
    values.push_back(builder.createBitcast(loc, value, registerType));
  }
}

static void storeWMMAFragment(CIRGenBuilderTy &builder, mlir::Location loc,
                              mlir::Value result, Address dst,
                              unsigned numRegisters) {
  CharUnits alignment =
      std::min(dst.getAlignment(), CharUnits::fromQuantity(4));
  for (unsigned i = 0; i < numRegisters; ++i) {
    mlir::Value value = cir::ExtractMemberOp::create(builder, loc, result, i);
    value = builder.createBitcast(loc, value, dst.getElementType());
    mlir::Value ptr = builder.createPtrStride(loc, dst.emitRawPointer(),
                                              builder.getSInt32(i, loc));
    builder.createStore(loc, value, Address(ptr, alignment));
  }
}

static std::optional<mlir::Value> emitWMMALoad(CIRGenFunction &cgf,
                                               const CallExpr *expr,
                                               llvm::StringRef fragment) {
  auto &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());
  Address dst = cgf.emitPointerWithAlignment(expr->getArg(0));
  mlir::Value src = cgf.emitScalarExpr(expr->getArg(1));
  mlir::Value ldm = cgf.emitScalarExpr(expr->getArg(2));
  auto isColMajor = expr->getArg(3)->getIntegerConstantExpr(cgf.getContext());
  if (!isColMajor)
    return std::nullopt;

  bool isFloat = mlir::isa<cir::SingleType>(dst.getElementType());
  unsigned numRegisters = fragment == "c" && !isFloat ? 4 : 8;
  std::string name = ("nvvm.wmma.m16n16k16.load." + fragment +
                      (*isColMajor != 0 ? ".col.stride." : ".row.stride.") +
                      (isFloat ? "f32" : "f16"))
                         .str();
  mlir::Type resultType = getWMMAResultType(
      builder, getWMMARegisterType(dst.getElementType()), numRegisters);
  mlir::Value result = builder.emitIntrinsicCallOp(loc, name, resultType,
                                                   mlir::ValueRange{src, ldm});
  storeWMMAFragment(builder, loc, result, dst, numRegisters);
  return mlir::Value{};
}

static std::optional<mlir::Value> emitWMMAStore(CIRGenFunction &cgf,
                                                const CallExpr *expr) {
  auto &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());
  mlir::Value dst = cgf.emitScalarExpr(expr->getArg(0));
  Address src = cgf.emitPointerWithAlignment(expr->getArg(1));
  mlir::Value ldm = cgf.emitScalarExpr(expr->getArg(2));
  auto isColMajor = expr->getArg(3)->getIntegerConstantExpr(cgf.getContext());
  if (!isColMajor)
    return std::nullopt;

  bool isFloat = mlir::isa<cir::SingleType>(src.getElementType());
  std::string name = (llvm::Twine("nvvm.wmma.m16n16k16.store.d.") +
                      (*isColMajor != 0 ? "col.stride." : "row.stride.") +
                      (isFloat ? "f32" : "f16"))
                         .str();
  llvm::SmallVector<mlir::Value, 10> values{dst};
  loadWMMAFragment(builder, loc, src, isFloat ? 8 : 4, values);
  values.push_back(ldm);
  cir::LLVMIntrinsicCallOp::create(builder, loc, builder.getStringAttr(name),
                                   /*resType=*/mlir::Type{}, values);
  return mlir::Value{};
}

static std::optional<mlir::Value> emitWMMAMma(CIRGenFunction &cgf,
                                              const CallExpr *expr) {
  auto &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());
  Address dst = cgf.emitPointerWithAlignment(expr->getArg(0));
  Address srcA = cgf.emitPointerWithAlignment(expr->getArg(1));
  Address srcB = cgf.emitPointerWithAlignment(expr->getArg(2));
  Address srcC = cgf.emitPointerWithAlignment(expr->getArg(3));
  auto layout = expr->getArg(4)->getIntegerConstantExpr(cgf.getContext());
  auto satf = expr->getArg(5)->getIntegerConstantExpr(cgf.getContext());
  if (!layout || layout->getSExtValue() < 0 || layout->getSExtValue() > 3 ||
      !satf)
    return std::nullopt;

  bool isFloatC = mlir::isa<cir::SingleType>(srcC.getElementType());
  bool isFloatD = mlir::isa<cir::SingleType>(dst.getElementType());
  const char *layouts[] = {"row.row", "row.col", "col.row", "col.col"};
  std::string name =
      (llvm::Twine("nvvm.wmma.m16n16k16.mma.") +
       layouts[layout->getSExtValue()] + (isFloatD ? ".f32" : ".f16") +
       (isFloatC ? ".f32" : ".f16") + (*satf != 0 ? ".satfinite" : ""))
          .str();

  // Read every input before writing D, since the fragments may alias.
  llvm::SmallVector<mlir::Value, 24> values;
  loadWMMAFragment(builder, loc, srcA, 8, values);
  loadWMMAFragment(builder, loc, srcB, 8, values);
  loadWMMAFragment(builder, loc, srcC, isFloatC ? 8 : 4, values);
  unsigned numResults = isFloatD ? 8 : 4;
  mlir::Type resultType = getWMMAResultType(
      builder, getWMMARegisterType(dst.getElementType()), numResults);
  mlir::Value result =
      builder.emitIntrinsicCallOp(loc, name, resultType, values);
  storeWMMAFragment(builder, loc, result, dst, numResults);
  return mlir::Value{};
}

std::optional<mlir::Value>
CIRGenFunction::emitNVPTXBuiltinExpr(unsigned builtinId, const CallExpr *expr) {
  switch (builtinId) {
  case NVPTX::BI__nvvm_atom_add_gen_i:
  case NVPTX::BI__nvvm_atom_add_gen_l:
  case NVPTX::BI__nvvm_atom_add_gen_ll:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Add, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_sub_gen_i:
  case NVPTX::BI__nvvm_atom_sub_gen_l:
  case NVPTX::BI__nvvm_atom_sub_gen_ll:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Sub, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_and_gen_i:
  case NVPTX::BI__nvvm_atom_and_gen_l:
  case NVPTX::BI__nvvm_atom_and_gen_ll:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::And, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_or_gen_i:
  case NVPTX::BI__nvvm_atom_or_gen_l:
  case NVPTX::BI__nvvm_atom_or_gen_ll:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Or, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_xor_gen_i:
  case NVPTX::BI__nvvm_atom_xor_gen_l:
  case NVPTX::BI__nvvm_atom_xor_gen_ll:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Xor, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_xchg_gen_ll:
    return makeScopedAtomicXchg(*this, expr, cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_max_gen_i:
  case NVPTX::BI__nvvm_atom_max_gen_l:
  case NVPTX::BI__nvvm_atom_max_gen_ll:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Max, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_max_gen_ui:
  case NVPTX::BI__nvvm_atom_max_gen_ul:
  case NVPTX::BI__nvvm_atom_max_gen_ull:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Max, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_min_gen_i:
  case NVPTX::BI__nvvm_atom_min_gen_l:
  case NVPTX::BI__nvvm_atom_min_gen_ll:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Min, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_min_gen_ui:
  case NVPTX::BI__nvvm_atom_min_gen_ul:
  case NVPTX::BI__nvvm_atom_min_gen_ull:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::Min, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cas_gen_ll:
    return emitAtomicCmpXchg(expr, /*returnBool=*/false, cir::MemOrder::Relaxed,
                             cir::MemOrder::Relaxed,
                             cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_add_gen_f:
  case NVPTX::BI__nvvm_atom_add_gen_d:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Add,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_inc_gen_ui:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::UIncWrap, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_atom_dec_gen_ui:
    return makeBinaryAtomicValue(cir::AtomicFetchKind::UDecWrap, expr,
                                 /*originalArgType=*/nullptr,
                                 /*emittedArgValue=*/nullptr,
                                 cir::MemOrder::Relaxed);
  case NVPTX::BI__nvvm_ldg_c:
  case NVPTX::BI__nvvm_ldg_sc:
  case NVPTX::BI__nvvm_ldg_c2:
  case NVPTX::BI__nvvm_ldg_sc2:
  case NVPTX::BI__nvvm_ldg_c4:
  case NVPTX::BI__nvvm_ldg_sc4:
  case NVPTX::BI__nvvm_ldg_s:
  case NVPTX::BI__nvvm_ldg_s2:
  case NVPTX::BI__nvvm_ldg_s4:
  case NVPTX::BI__nvvm_ldg_i:
  case NVPTX::BI__nvvm_ldg_i2:
  case NVPTX::BI__nvvm_ldg_i4:
  case NVPTX::BI__nvvm_ldg_l:
  case NVPTX::BI__nvvm_ldg_l2:
  case NVPTX::BI__nvvm_ldg_ll:
  case NVPTX::BI__nvvm_ldg_ll2:
  case NVPTX::BI__nvvm_ldg_uc:
  case NVPTX::BI__nvvm_ldg_uc2:
  case NVPTX::BI__nvvm_ldg_uc4:
  case NVPTX::BI__nvvm_ldg_us:
  case NVPTX::BI__nvvm_ldg_us2:
  case NVPTX::BI__nvvm_ldg_us4:
  case NVPTX::BI__nvvm_ldg_ui:
  case NVPTX::BI__nvvm_ldg_ui2:
  case NVPTX::BI__nvvm_ldg_ui4:
  case NVPTX::BI__nvvm_ldg_ul:
  case NVPTX::BI__nvvm_ldg_ul2:
  case NVPTX::BI__nvvm_ldg_ull:
  case NVPTX::BI__nvvm_ldg_ull2:
  case NVPTX::BI__nvvm_ldg_f:
  case NVPTX::BI__nvvm_ldg_f2:
  case NVPTX::BI__nvvm_ldg_f4:
  case NVPTX::BI__nvvm_ldg_d:
  case NVPTX::BI__nvvm_ldg_d2:
    return makeLdg(*this, expr);
  case NVPTX::BI__nvvm_ldu_c:
  case NVPTX::BI__nvvm_ldu_sc:
  case NVPTX::BI__nvvm_ldu_c2:
  case NVPTX::BI__nvvm_ldu_sc2:
  case NVPTX::BI__nvvm_ldu_c4:
  case NVPTX::BI__nvvm_ldu_sc4:
  case NVPTX::BI__nvvm_ldu_s:
  case NVPTX::BI__nvvm_ldu_s2:
  case NVPTX::BI__nvvm_ldu_s4:
  case NVPTX::BI__nvvm_ldu_i:
  case NVPTX::BI__nvvm_ldu_i2:
  case NVPTX::BI__nvvm_ldu_i4:
  case NVPTX::BI__nvvm_ldu_l:
  case NVPTX::BI__nvvm_ldu_l2:
  case NVPTX::BI__nvvm_ldu_ll:
  case NVPTX::BI__nvvm_ldu_ll2:
  case NVPTX::BI__nvvm_ldu_uc:
  case NVPTX::BI__nvvm_ldu_uc2:
  case NVPTX::BI__nvvm_ldu_uc4:
  case NVPTX::BI__nvvm_ldu_us:
  case NVPTX::BI__nvvm_ldu_us2:
  case NVPTX::BI__nvvm_ldu_us4:
  case NVPTX::BI__nvvm_ldu_ui:
  case NVPTX::BI__nvvm_ldu_ui2:
  case NVPTX::BI__nvvm_ldu_ui4:
  case NVPTX::BI__nvvm_ldu_ul:
  case NVPTX::BI__nvvm_ldu_ul2:
  case NVPTX::BI__nvvm_ldu_ull:
  case NVPTX::BI__nvvm_ldu_ull2:
    return makeLdu(*this, expr, "nvvm.ldu.global.i");
  case NVPTX::BI__nvvm_ldu_f:
  case NVPTX::BI__nvvm_ldu_f2:
  case NVPTX::BI__nvvm_ldu_f4:
  case NVPTX::BI__nvvm_ldu_d:
  case NVPTX::BI__nvvm_ldu_d2:
    return makeLdu(*this, expr, "nvvm.ldu.global.f");
  case NVPTX::BI__nvvm_atom_cta_add_gen_i:
  case NVPTX::BI__nvvm_atom_cta_add_gen_l:
  case NVPTX::BI__nvvm_atom_cta_add_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Add,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_add_gen_i:
  case NVPTX::BI__nvvm_atom_sys_add_gen_l:
  case NVPTX::BI__nvvm_atom_sys_add_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Add,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_add_gen_f:
  case NVPTX::BI__nvvm_atom_cta_add_gen_d:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Add,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_add_gen_f:
  case NVPTX::BI__nvvm_atom_sys_add_gen_d:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Add,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_ll:
    return makeScopedAtomicXchg(*this, expr, cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_ll:
    return makeScopedAtomicXchg(*this, expr, cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_max_gen_i:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_max_gen_l:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ull:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Max,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_max_gen_i:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_max_gen_l:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ull:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Max,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_min_gen_i:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_min_gen_l:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ull:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Min,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_min_gen_i:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_min_gen_l:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ull:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Min,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_inc_gen_ui:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::UIncWrap,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_cta_dec_gen_ui:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::UDecWrap,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_inc_gen_ui:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::UIncWrap,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_sys_dec_gen_ui:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::UDecWrap,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_and_gen_i:
  case NVPTX::BI__nvvm_atom_cta_and_gen_l:
  case NVPTX::BI__nvvm_atom_cta_and_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::And,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_and_gen_i:
  case NVPTX::BI__nvvm_atom_sys_and_gen_l:
  case NVPTX::BI__nvvm_atom_sys_and_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::And,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_or_gen_i:
  case NVPTX::BI__nvvm_atom_cta_or_gen_l:
  case NVPTX::BI__nvvm_atom_cta_or_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Or,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_or_gen_i:
  case NVPTX::BI__nvvm_atom_sys_or_gen_l:
  case NVPTX::BI__nvvm_atom_sys_or_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Or,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_xor_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Xor,
                               cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_xor_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_ll:
    return makeScopedAtomicRMW(*this, expr, cir::AtomicFetchKind::Xor,
                               cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_atom_cta_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_ll:
    return emitAtomicCmpXchg(expr, /*returnBool=*/false, cir::MemOrder::Relaxed,
                             cir::MemOrder::Relaxed,
                             cir::SyncScopeKind::Workgroup);
  case NVPTX::BI__nvvm_atom_sys_cas_gen_us:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_i:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_l:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_ll:
    return emitAtomicCmpXchg(expr, /*returnBool=*/false, cir::MemOrder::Relaxed,
                             cir::MemOrder::Relaxed,
                             cir::SyncScopeKind::System);
  case NVPTX::BI__nvvm_match_all_sync_i32p:
  case NVPTX::BI__nvvm_match_all_sync_i64p:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  // FP MMA loads
  case NVPTX::BI__hmma_m16n16k16_ld_a:
    return emitWMMALoad(*this, expr, "a");
  case NVPTX::BI__hmma_m16n16k16_ld_b:
    return emitWMMALoad(*this, expr, "b");
  case NVPTX::BI__hmma_m16n16k16_ld_c_f16:
  case NVPTX::BI__hmma_m16n16k16_ld_c_f32:
    return emitWMMALoad(*this, expr, "c");
  case NVPTX::BI__hmma_m32n8k16_ld_a:
  case NVPTX::BI__hmma_m32n8k16_ld_b:
  case NVPTX::BI__hmma_m32n8k16_ld_c_f16:
  case NVPTX::BI__hmma_m32n8k16_ld_c_f32:
  case NVPTX::BI__hmma_m8n32k16_ld_a:
  case NVPTX::BI__hmma_m8n32k16_ld_b:
  case NVPTX::BI__hmma_m8n32k16_ld_c_f16:
  case NVPTX::BI__hmma_m8n32k16_ld_c_f32:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__imma_m16n16k16_ld_a_s8:
  case NVPTX::BI__imma_m16n16k16_ld_a_u8:
  case NVPTX::BI__imma_m16n16k16_ld_b_s8:
  case NVPTX::BI__imma_m16n16k16_ld_b_u8:
  case NVPTX::BI__imma_m16n16k16_ld_c:
  case NVPTX::BI__imma_m32n8k16_ld_a_s8:
  case NVPTX::BI__imma_m32n8k16_ld_a_u8:
  case NVPTX::BI__imma_m32n8k16_ld_b_s8:
  case NVPTX::BI__imma_m32n8k16_ld_b_u8:
  case NVPTX::BI__imma_m32n8k16_ld_c:
  case NVPTX::BI__imma_m8n32k16_ld_a_s8:
  case NVPTX::BI__imma_m8n32k16_ld_a_u8:
  case NVPTX::BI__imma_m8n32k16_ld_b_s8:
  case NVPTX::BI__imma_m8n32k16_ld_b_u8:
  case NVPTX::BI__imma_m8n32k16_ld_c:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__imma_m8n8k32_ld_a_s4:
  case NVPTX::BI__imma_m8n8k32_ld_a_u4:
  case NVPTX::BI__imma_m8n8k32_ld_b_s4:
  case NVPTX::BI__imma_m8n8k32_ld_b_u4:
  case NVPTX::BI__imma_m8n8k32_ld_c:
  case NVPTX::BI__bmma_m8n8k128_ld_a_b1:
  case NVPTX::BI__bmma_m8n8k128_ld_b_b1:
  case NVPTX::BI__bmma_m8n8k128_ld_c:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__dmma_m8n8k4_ld_a:
  case NVPTX::BI__dmma_m8n8k4_ld_b:
  case NVPTX::BI__dmma_m8n8k4_ld_c:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__mma_bf16_m16n16k16_ld_a:
  case NVPTX::BI__mma_bf16_m16n16k16_ld_b:
  case NVPTX::BI__mma_bf16_m8n32k16_ld_a:
  case NVPTX::BI__mma_bf16_m8n32k16_ld_b:
  case NVPTX::BI__mma_bf16_m32n8k16_ld_a:
  case NVPTX::BI__mma_bf16_m32n8k16_ld_b:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_a:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_b:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_c:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__hmma_m16n16k16_st_c_f16:
  case NVPTX::BI__hmma_m16n16k16_st_c_f32:
    return emitWMMAStore(*this, expr);
  case NVPTX::BI__hmma_m32n8k16_st_c_f16:
  case NVPTX::BI__hmma_m32n8k16_st_c_f32:
  case NVPTX::BI__hmma_m8n32k16_st_c_f16:
  case NVPTX::BI__hmma_m8n32k16_st_c_f32:
  case NVPTX::BI__imma_m16n16k16_st_c_i32:
  case NVPTX::BI__imma_m32n8k16_st_c_i32:
  case NVPTX::BI__imma_m8n32k16_st_c_i32:
  case NVPTX::BI__imma_m8n8k32_st_c_i32:
  case NVPTX::BI__bmma_m8n8k128_st_c_i32:
  case NVPTX::BI__dmma_m8n8k4_st_c_f64:
  case NVPTX::BI__mma_m16n16k8_st_c_f32:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  // BI__hmma_m16n16k16_mma_<Dtype><CType>(d, a, b, c, layout, satf) -->
  // Intrinsic::nvvm_wmma_m16n16k16_mma_sync<layout A,B><DType><CType><Satf>
  case NVPTX::BI__hmma_m16n16k16_mma_f16f16:
  case NVPTX::BI__hmma_m16n16k16_mma_f32f16:
  case NVPTX::BI__hmma_m16n16k16_mma_f32f32:
  case NVPTX::BI__hmma_m16n16k16_mma_f16f32:
    return emitWMMAMma(*this, expr);
  case NVPTX::BI__hmma_m32n8k16_mma_f16f16:
  case NVPTX::BI__hmma_m32n8k16_mma_f32f16:
  case NVPTX::BI__hmma_m32n8k16_mma_f32f32:
  case NVPTX::BI__hmma_m32n8k16_mma_f16f32:
  case NVPTX::BI__hmma_m8n32k16_mma_f16f16:
  case NVPTX::BI__hmma_m8n32k16_mma_f32f16:
  case NVPTX::BI__hmma_m8n32k16_mma_f32f32:
  case NVPTX::BI__hmma_m8n32k16_mma_f16f32:
  case NVPTX::BI__imma_m16n16k16_mma_s8:
  case NVPTX::BI__imma_m16n16k16_mma_u8:
  case NVPTX::BI__imma_m32n8k16_mma_s8:
  case NVPTX::BI__imma_m32n8k16_mma_u8:
  case NVPTX::BI__imma_m8n32k16_mma_s8:
  case NVPTX::BI__imma_m8n32k16_mma_u8:
  case NVPTX::BI__imma_m8n8k32_mma_s4:
  case NVPTX::BI__imma_m8n8k32_mma_u4:
  case NVPTX::BI__bmma_m8n8k128_mma_xor_popc_b1:
  case NVPTX::BI__bmma_m8n8k128_mma_and_popc_b1:
  case NVPTX::BI__dmma_m8n8k4_mma_f64:
  case NVPTX::BI__mma_bf16_m16n16k16_mma_f32:
  case NVPTX::BI__mma_bf16_m8n32k16_mma_f32:
  case NVPTX::BI__mma_bf16_m32n8k16_mma_f32:
  case NVPTX::BI__mma_tf32_m16n16k8_mma_f32:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  // The following builtins require half type support
  case NVPTX::BI__nvvm_ff2f16x2_rn:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ff2f16x2_rn_relu:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ff2f16x2_rz:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ff2f16x2_rz_relu:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_f16:
  case NVPTX::BI__nvvm_fma_rn_f16x2:
  case NVPTX::BI__nvvm_fma_rn_bf16:
  case NVPTX::BI__nvvm_fma_rn_bf16x2:
  case NVPTX::BI__nvvm_fma_rn_f:
  case NVPTX::BI__nvvm_fma_rn_d:
    return emitNVVMFMA(*this, expr, "nvvm.fma",
                       llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_fma_rz_f:
  case NVPTX::BI__nvvm_fma_rz_d:
    return emitNVVMFMA(*this, expr, "nvvm.fma", llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_fma_rm_f:
  case NVPTX::BI__nvvm_fma_rm_d:
    return emitNVVMFMA(*this, expr, "nvvm.fma",
                       llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_fma_rp_f:
  case NVPTX::BI__nvvm_fma_rp_d:
    return emitNVVMFMA(*this, expr, "nvvm.fma",
                       llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_fma_rn_ftz_f16:
  case NVPTX::BI__nvvm_fma_rn_ftz_f16x2:
  case NVPTX::BI__nvvm_fma_rn_ftz_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz",
                       llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_fma_rz_ftz_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz",
                       llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_fma_rm_ftz_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz",
                       llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_fma_rp_ftz_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz",
                       llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_fma_rn_sat_f16:
  case NVPTX::BI__nvvm_fma_rn_sat_f16x2:
  case NVPTX::BI__nvvm_fma_rn_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.sat",
                       llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_fma_rz_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.sat",
                       llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_fma_rm_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.sat",
                       llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_fma_rp_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.sat",
                       llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16:
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16x2:
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz.sat",
                       llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_fma_rz_ftz_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz.sat",
                       llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_fma_rm_ftz_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz.sat",
                       llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_fma_rp_ftz_sat_f:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz.sat",
                       llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_fma_rn_relu_f16:
  case NVPTX::BI__nvvm_fma_rn_relu_f16x2:
  case NVPTX::BI__nvvm_fma_rn_relu_bf16:
  case NVPTX::BI__nvvm_fma_rn_relu_bf16x2:
    return emitNVVMFMA(*this, expr, "nvvm.fma.relu",
                       llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16:
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16x2:
    return emitNVVMFMA(*this, expr, "nvvm.fma.ftz.relu",
                       llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_fma_rn_oob_f16:
  case NVPTX::BI__nvvm_fma_rn_oob_f16x2:
  case NVPTX::BI__nvvm_fma_rn_oob_bf16:
  case NVPTX::BI__nvvm_fma_rn_oob_bf16x2:
    return emitBuiltinWithOneOverloadedType<3>(expr, "nvvm.fma.rn.oob")
        .getValue();
  case NVPTX::BI__nvvm_fma_rn_oob_relu_f16:
  case NVPTX::BI__nvvm_fma_rn_oob_relu_f16x2:
  case NVPTX::BI__nvvm_fma_rn_oob_relu_bf16:
  case NVPTX::BI__nvvm_fma_rn_oob_relu_bf16x2:
    return emitBuiltinWithOneOverloadedType<3>(expr, "nvvm.fma.rn.oob.relu")
        .getValue();
  case NVPTX::BI__nvvm_fmax_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.ftz.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.ftz.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.ftz.nan.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.ftz.nan.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmax.ftz.nan.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmax.ftz.nan.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmax.ftz.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmax.ftz.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmax_nan_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.nan.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_nan_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmax.nan.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmax.nan.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmax.nan.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmax.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmax.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.ftz.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.ftz.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.ftz.nan.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.ftz.nan.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmin.ftz.nan.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmin.ftz.nan.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmin.ftz.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmin.ftz.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_nan_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.nan.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_nan_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr, "nvvm.fmin.nan.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmin.nan.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(
               expr, "nvvm.fmin.nan.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmin.xorsign.abs.f16")
        .getValue();
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16x2:
    return emitBuiltinWithOneOverloadedType<2>(expr,
                                               "nvvm.fmin.xorsign.abs.f16x2")
        .getValue();
  case NVPTX::BI__nvvm_fabs_f:
  case NVPTX::BI__nvvm_abs_bf16:
  case NVPTX::BI__nvvm_abs_bf16x2:
  case NVPTX::BI__nvvm_fabs_f16:
  case NVPTX::BI__nvvm_fabs_f16x2:
    return emitUnaryNVVMIntrinsic(*this, expr, "nvvm.fabs");
  case NVPTX::BI__nvvm_fabs_ftz_f:
  case NVPTX::BI__nvvm_fabs_ftz_f16:
  case NVPTX::BI__nvvm_fabs_ftz_f16x2:
    return emitUnaryNVVMIntrinsic(*this, expr, "nvvm.fabs.ftz");
  case NVPTX::BI__nvvm_fabs_d:
    return emitUnaryNVVMIntrinsic(*this, expr, "fabs");
  case NVPTX::BI__nvvm_ex2_approx_d:
  case NVPTX::BI__nvvm_ex2_approx_f:
  case NVPTX::BI__nvvm_ex2_approx_f16:
  case NVPTX::BI__nvvm_ex2_approx_f16x2:
    return emitUnaryNVVMIntrinsic(*this, expr, "nvvm.ex2.approx");
  case NVPTX::BI__nvvm_ex2_approx_ftz_f:
    return emitUnaryNVVMIntrinsic(*this, expr, "nvvm.ex2.approx.ftz");
  case NVPTX::BI__nvvm_add_rn_f:
  case NVPTX::BI__nvvm_add_rn_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_add_rz_f:
  case NVPTX::BI__nvvm_add_rz_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd",
                           llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_add_rm_f:
  case NVPTX::BI__nvvm_add_rm_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd",
                           llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_add_rp_f:
  case NVPTX::BI__nvvm_add_rp_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd",
                           llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_add_rn_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_add_rz_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz",
                           llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_add_rm_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz",
                           llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_add_rp_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz",
                           llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_add_rn_sat_f:
  case NVPTX::BI__nvvm_add_rn_sat_f16:
  case NVPTX::BI__nvvm_add_rn_sat_v2f16:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.sat",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_add_rz_sat_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.sat",
                           llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_add_rm_sat_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.sat",
                           llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_add_rp_sat_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.sat",
                           llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_add_rn_ftz_sat_f:
  case NVPTX::BI__nvvm_add_rn_ftz_sat_f16:
  case NVPTX::BI__nvvm_add_rn_ftz_sat_v2f16:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz.sat",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_add_rz_ftz_sat_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz.sat",
                           llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_add_rm_ftz_sat_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz.sat",
                           llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_add_rp_ftz_sat_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fadd.ftz.sat",
                           llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_mul_rn_f:
  case NVPTX::BI__nvvm_mul_rn_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_mul_rz_f:
  case NVPTX::BI__nvvm_mul_rz_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul",
                           llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_mul_rm_f:
  case NVPTX::BI__nvvm_mul_rm_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul",
                           llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_mul_rp_f:
  case NVPTX::BI__nvvm_mul_rp_d:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul",
                           llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_mul_rn_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul.ftz",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_mul_rz_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul.ftz",
                           llvm::APFloat::rmTowardZero);
  case NVPTX::BI__nvvm_mul_rm_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul.ftz",
                           llvm::APFloat::rmTowardNegative);
  case NVPTX::BI__nvvm_mul_rp_ftz_f:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul.ftz",
                           llvm::APFloat::rmTowardPositive);
  case NVPTX::BI__nvvm_mul_rn_sat_f16:
  case NVPTX::BI__nvvm_mul_rn_sat_v2f16:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul.sat",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_mul_rn_ftz_sat_f16:
  case NVPTX::BI__nvvm_mul_rn_ftz_sat_v2f16:
    return emitNVVMFPArith(*this, expr, "nvvm.fmul.ftz.sat",
                           llvm::APFloat::rmNearestTiesToEven);
  case NVPTX::BI__nvvm_ldg_h:
  case NVPTX::BI__nvvm_ldg_h2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ldu_h:
  case NVPTX::BI__nvvm_ldu_h2:
    return makeLdu(*this, expr, "nvvm.ldu.global.f");
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_4:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_8:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_cp_async_cg_shared_global_16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_x:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_y:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_z:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_w:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_x:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_y:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_z:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_w:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_x:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_y:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_z:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_w:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_x:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_y:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_z:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_w:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctarank:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctarank:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_is_explicit_cluster:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_isspacep_shared_cluster:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_mapa:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_mapa_shared_cluster:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_getctarank:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_getctarank_shared_cluster:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_barrier_cluster_arrive:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "nvvm.barrier.cluster.arrive",
                                       builder.getVoidTy());
  case NVPTX::BI__nvvm_barrier_cluster_arrive_relaxed:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "nvvm.barrier.cluster.arrive.relaxed",
                                       builder.getVoidTy());
  case NVPTX::BI__nvvm_barrier_cluster_wait:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "nvvm.barrier.cluster.wait",
                                       builder.getVoidTy());
  case NVPTX::BI__nvvm_fence_sc_cluster:
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "nvvm.fence.sc.cluster",
                                       builder.getVoidTy());
  case NVPTX::BI__nvvm_bar_sync:
    return builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), "nvvm.barrier.cta.sync.aligned.all",
        builder.getVoidTy(), mlir::ValueRange{emitScalarExpr(expr->getArg(0))});
  case NVPTX::BI__syncthreads:
    return builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), "nvvm.barrier.cta.sync.aligned.all",
        builder.getVoidTy(),
        mlir::ValueRange{builder.getConstInt(getLoc(expr->getExprLoc()),
                                             builder.getSInt32Ty(), 0)});
  case NVPTX::BI__nvvm_barrier_sync:
    return builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), "nvvm.barrier.cta.sync.all",
        builder.getVoidTy(), mlir::ValueRange{emitScalarExpr(expr->getArg(0))});
  case NVPTX::BI__nvvm_barrier_sync_cnt:
    return builder.emitIntrinsicCallOp(
        getLoc(expr->getExprLoc()), "nvvm.barrier.cta.sync.count",
        builder.getVoidTy(),
        mlir::ValueRange{emitScalarExpr(expr->getArg(0)),
                         emitScalarExpr(expr->getArg(1))});
  case NVPTX::BI__nvvm_bar0_and:
    return emitBar0Reduction(*this, expr,
                             "nvvm.barrier.cta.red.and.aligned.all",
                             /*returnsPred=*/true);
  case NVPTX::BI__nvvm_bar0_or:
    return emitBar0Reduction(*this, expr, "nvvm.barrier.cta.red.or.aligned.all",
                             /*returnsPred=*/true);
  case NVPTX::BI__nvvm_bar0_popc:
    return emitBar0Reduction(*this, expr,
                             "nvvm.barrier.cta.red.popc.aligned.all",
                             /*returnsPred=*/false);

  default:
    return std::nullopt;
  }
}
