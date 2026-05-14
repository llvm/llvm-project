//===---- CIRGenBuiltinAMDGPU.cpp - Emit CIR for AMDGPU builtins ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit AMDGPU Builtin calls.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

#include "mlir/IR/Value.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

static mlir::Value emitBinaryExpMaybeConstrainedFPBuiltin(
    CIRGenFunction &cgf, const CallExpr *e, llvm::StringRef intrinsicName,
    llvm::StringRef constrainedIntrinsicName) {
  mlir::Value src0 = cgf.emitScalarExpr(e->getArg(0));
  mlir::Value src1 = cgf.emitScalarExpr(e->getArg(1));
  mlir::Location loc = cgf.getLoc(e->getExprLoc());

  CIRGenBuilderTy &builder = cgf.getBuilder();

  CIRGenFunction::CIRGenFPOptionsRAII fpOptsRAII(cgf, e);

  if (builder.getIsFPConstrained()) {
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "constrained FP intrinsic support is NYI.");
  }

  return builder.emitIntrinsicCallOp(loc, intrinsicName, src0.getType(),
                                     mlir::ValueRange{src0, src1});
}

static mlir::Value emitLogbBuiltin(CIRGenFunction &cgf, const CallExpr *e,
                                   const llvm::fltSemantics &fSem) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(e->getExprLoc());

  mlir::Value src0 = cgf.emitScalarExpr(e->getArg(0));
  mlir::Type srcTy = src0.getType();
  mlir::Type int32Ty = builder.getSInt32Ty();

  cir::RecordType frExpResTy =
      builder.getAnonRecordTy({srcTy, int32Ty}, false, false);

  mlir::Value frExpResult = builder.emitIntrinsicCallOp(
      loc, "frexp", frExpResTy, mlir::ValueRange{src0});

  mlir::Value exp =
      cir::ExtractMemberOp::create(builder, loc, int32Ty, frExpResult, 1);

  mlir::Value negativeOne =
      builder.getConstant(loc, cir::IntAttr::get(int32Ty, -1));
  mlir::Value expMinus1 = builder.createAdd(loc, exp, negativeOne);

  mlir::Value siToFp = cir::CastOp::create(
      builder, loc, srcTy, cir::CastKind::int_to_float, expMinus1);

  mlir::Value fabs = cir::FAbsOp::create(builder, loc, srcTy, src0);

  llvm::APFloat infVal = llvm::APFloat::getInf(fSem);
  mlir::Value inf = builder.getConstant(loc, cir::FPAttr::get(srcTy, infVal));

  mlir::Value fabsNegInf =
      builder.createCompare(loc, cir::CmpOpKind::ne, fabs, inf);

  mlir::Value sel = builder.createSelect(loc, fabsNegInf, siToFp, fabs);

  llvm::APFloat zeroValue = llvm::APFloat::getZero(fSem);
  mlir::Value zero =
      builder.getConstant(loc, cir::FPAttr::get(srcTy, zeroValue));

  mlir::Value srcEqZero =
      builder.createCompare(loc, cir::CmpOpKind::eq, src0, zero);

  llvm::APFloat negInfVal = llvm::APFloat::getInf(fSem, true);
  mlir::Value negInf =
      builder.getConstant(loc, cir::FPAttr::get(srcTy, negInfVal));

  mlir::Value res = builder.createSelect(loc, srcEqZero, negInf, sel);

  return res;
}

std::optional<mlir::Value>
CIRGenFunction::emitAMDGPUBuiltinExpr(unsigned builtinId,
                                      const CallExpr *expr) {
  switch (builtinId) {
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_div_scale:
  case AMDGPU::BI__builtin_amdgcn_div_scalef: {
    Address flagOutPtr = emitPointerWithAlignment(expr->getArg(3));
    llvm::StringRef intrinsicName = "amdgcn.div.scale";
    mlir::Value x = emitScalarExpr(expr->getArg(0));
    mlir::Value y = emitScalarExpr(expr->getArg(1));
    mlir::Value z = emitScalarExpr(expr->getArg(2));

    auto i1Ty = builder.getUIntNTy(1);
    cir::RecordType resTy = builder.getAnonRecordTy(
        {x.getType(), i1Ty}, /*packed=*/false, /*padded=*/false);

    mlir::Value structResult =
        cir::LLVMIntrinsicCallOp::create(builder, getLoc(expr->getExprLoc()),
                                         builder.getStringAttr(intrinsicName),
                                         resTy, {x, y, z})
            .getResult();

    mlir::Value result = cir::ExtractMemberOp::create(
        builder, getLoc(expr->getExprLoc()), x.getType(), structResult, 0);
    mlir::Value flag = cir::ExtractMemberOp::create(
        builder, getLoc(expr->getExprLoc()), i1Ty, structResult, 1);

    mlir::Type flagType = flagOutPtr.getElementType();
    mlir::Value flagToStore =
        cir::CastOp::create(builder, getLoc(expr->getExprLoc()), flagType,
                            cir::CastKind::int_to_bool, flag);
    builder.createStore(getLoc(expr->getExprLoc()), flagToStore, flagOutPtr);
    return result;
  }
  case AMDGPU::BI__builtin_amdgcn_div_fmas:
  case AMDGPU::BI__builtin_amdgcn_div_fmasf: {
    mlir::Value src0 = emitScalarExpr(expr->getArg(0));
    mlir::Value src1 = emitScalarExpr(expr->getArg(1));
    mlir::Value src2 = emitScalarExpr(expr->getArg(2));
    mlir::Value src3 = emitScalarExpr(expr->getArg(3));
    mlir::Value result = cir::LLVMIntrinsicCallOp::create(
                             builder, getLoc(expr->getExprLoc()),
                             builder.getStringAttr("amdgcn.div.fmas"),
                             src0.getType(), {src0, src1, src2, src3})
                             .getResult();
    return result;
  }
  case AMDGPU::BI__builtin_amdgcn_ds_swizzle: {
    mlir::Value src0 = emitScalarExpr(expr->getArg(0));
    mlir::Value src1 = emitScalarExpr(expr->getArg(1));
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "amdgcn.ds.swizzle", src0.getType(),
                                       mlir::ValueRange{src0, src1});
  }
  case AMDGPU::BI__builtin_amdgcn_mov_dpp8:
  case AMDGPU::BI__builtin_amdgcn_mov_dpp:
  case AMDGPU::BI__builtin_amdgcn_update_dpp: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_permlane16:
  case AMDGPU::BI__builtin_amdgcn_permlanex16:
  case AMDGPU::BI__builtin_amdgcn_permlane64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_readlane: {
    mlir::Value src0 = emitScalarExpr(expr->getArg(0));
    mlir::Value src1 = emitScalarExpr(expr->getArg(1));
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "amdgcn.readlane", src0.getType(),
                                       mlir::ValueRange{src0, src1});
  }
  case AMDGPU::BI__builtin_amdgcn_readfirstlane: {
    mlir::Value src0 = emitScalarExpr(expr->getArg(0));
    return builder.emitIntrinsicCallOp(getLoc(expr->getExprLoc()),
                                       "amdgcn.readfirstlane", src0.getType(),
                                       mlir::ValueRange{src0});
  }
  case AMDGPU::BI__builtin_amdgcn_wave_shuffle: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_div_fixup:
  case AMDGPU::BI__builtin_amdgcn_div_fixupf:
  case AMDGPU::BI__builtin_amdgcn_div_fixuph: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_trig_preop:
  case AMDGPU::BI__builtin_amdgcn_trig_preopf: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_rcp:
  case AMDGPU::BI__builtin_amdgcn_rcpf:
  case AMDGPU::BI__builtin_amdgcn_rcph:
  case AMDGPU::BI__builtin_amdgcn_rcp_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_sqrt:
  case AMDGPU::BI__builtin_amdgcn_sqrtf:
  case AMDGPU::BI__builtin_amdgcn_sqrth:
  case AMDGPU::BI__builtin_amdgcn_sqrt_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_rsq:
  case AMDGPU::BI__builtin_amdgcn_rsqf:
  case AMDGPU::BI__builtin_amdgcn_rsqh:
  case AMDGPU::BI__builtin_amdgcn_rsq_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_rsq_clamp:
  case AMDGPU::BI__builtin_amdgcn_rsq_clampf: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_sinf:
  case AMDGPU::BI__builtin_amdgcn_sinh:
  case AMDGPU::BI__builtin_amdgcn_sin_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_cosf:
  case AMDGPU::BI__builtin_amdgcn_cosh:
  case AMDGPU::BI__builtin_amdgcn_cos_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_dispatch_ptr: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_logf:
  case AMDGPU::BI__builtin_amdgcn_log_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_exp2f:
  case AMDGPU::BI__builtin_amdgcn_exp2_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_log_clampf: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_ldexp:
  case AMDGPU::BI__builtin_amdgcn_ldexpf:
  case AMDGPU::BI__builtin_amdgcn_ldexph: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_frexp_mant:
  case AMDGPU::BI__builtin_amdgcn_frexp_mantf:
  case AMDGPU::BI__builtin_amdgcn_frexp_manth: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_frexp_exp:
  case AMDGPU::BI__builtin_amdgcn_frexp_expf:
  case AMDGPU::BI__builtin_amdgcn_frexp_exph: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_fract:
  case AMDGPU::BI__builtin_amdgcn_fractf:
  case AMDGPU::BI__builtin_amdgcn_fracth: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_lerp: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_ubfe: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_sbfe: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_ballot_w32:
  case AMDGPU::BI__builtin_amdgcn_ballot_w64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_inverse_ballot_w32:
  case AMDGPU::BI__builtin_amdgcn_inverse_ballot_w64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_tanhf:
  case AMDGPU::BI__builtin_amdgcn_tanhh:
  case AMDGPU::BI__builtin_amdgcn_tanh_bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_uicmp:
  case AMDGPU::BI__builtin_amdgcn_uicmpl:
  case AMDGPU::BI__builtin_amdgcn_sicmp:
  case AMDGPU::BI__builtin_amdgcn_sicmpl: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_fcmp:
  case AMDGPU::BI__builtin_amdgcn_fcmpf: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_class:
  case AMDGPU::BI__builtin_amdgcn_classf:
  case AMDGPU::BI__builtin_amdgcn_classh: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_fmed3f:
  case AMDGPU::BI__builtin_amdgcn_fmed3h: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_ds_append:
  case AMDGPU::BI__builtin_amdgcn_ds_consume: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4bf16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8bf16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8bf16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4f16:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4i16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b32:
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b64:
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b128:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b32:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b64:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b128: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b32:
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b64:
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b128: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_load_to_lds: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_32x4B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_32x4B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_16x8B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_16x8B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_8x16B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_8x16B: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_get_fpenv:
  case AMDGPU::BI__builtin_amdgcn_set_fpenv: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_read_exec:
  case AMDGPU::BI__builtin_amdgcn_read_exec_lo:
  case AMDGPU::BI__builtin_amdgcn_read_exec_hi: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_h:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_l:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_lh: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_image_bvh8_intersect_ray:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_dual_intersect_ray: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_image_load_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_1d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_1darray_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_3d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_cube_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_3d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_cube_v4f16_i32: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_image_store_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_1d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_1darray_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_3d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_cube_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1darray_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_3d_v4f16_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_cube_v4f16_i32: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_image_sample_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_cube_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_cube_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_cube_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_cube_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_cube_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_cube_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2darray_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2darray_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2darray_f32_f32: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_image_gather4_lz_2d_v4f32_f32: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x4_f32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_f16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x32_f16:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x64_iu8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_32x16x128_f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_32x16x128_f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_f16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x64_f16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x128_iu8: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  // amdgcn workgroup size
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_x:
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_y:
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_z: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_grid_size_x:
  case AMDGPU::BI__builtin_amdgcn_grid_size_y:
  case AMDGPU::BI__builtin_amdgcn_grid_size_z: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_r600_recipsqrt_ieee:
  case AMDGPU::BI__builtin_r600_recipsqrt_ieeef: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_alignbit: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_fence: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_atomic_inc32:
  case AMDGPU::BI__builtin_amdgcn_atomic_inc64:
  case AMDGPU::BI__builtin_amdgcn_atomic_dec32:
  case AMDGPU::BI__builtin_amdgcn_atomic_dec64:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_faddf:
  case AMDGPU::BI__builtin_amdgcn_ds_fminf:
  case AMDGPU::BI__builtin_amdgcn_ds_fmaxf:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fmin_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fmax_f64:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmin_f64:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmax_f64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_s_sendmsg_rtn:
  case AMDGPU::BI__builtin_amdgcn_s_sendmsg_rtnl: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_permlane16_swap:
  case AMDGPU::BI__builtin_amdgcn_permlane32_swap: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_bitop3_b32:
  case AMDGPU::BI__builtin_amdgcn_bitop3_b16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_make_buffer_rsrc: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b8:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b16:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b32:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b64:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b96:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b128: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b8:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b16:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b32:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b64:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b96:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b128: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_add_i32: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case AMDGPU::BI__builtin_amdgcn_s_prefetch_data: {
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented AMDGPU builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  }
  case Builtin::BIlogbf:
  case Builtin::BI__builtin_logbf:
    return emitLogbBuiltin(*this, expr, llvm::APFloat::IEEEsingle());
  case Builtin::BIlogb:
  case Builtin::BI__builtin_logb:
    return emitLogbBuiltin(*this, expr, llvm::APFloat::IEEEdouble());
  case Builtin::BIscalbnf:
  case Builtin::BI__builtin_scalbnf:
  case Builtin::BIscalbn:
  case Builtin::BI__builtin_scalbn: {
    return emitBinaryExpMaybeConstrainedFPBuiltin(
        *this, expr, "ldexp", "experimental.constrained.ldexp");
  }
  default:
    return std::nullopt;
  }
}
