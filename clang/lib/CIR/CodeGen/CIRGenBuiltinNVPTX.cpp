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

using namespace clang;
using namespace clang::CIRGen;

std::optional<mlir::Value>
CIRGenFunction::emitNVPTXBuiltinExpr(unsigned builtinId, const CallExpr *expr) {
  switch (builtinId) {
  case NVPTX::BI__nvvm_atom_add_gen_i:
  case NVPTX::BI__nvvm_atom_add_gen_l:
  case NVPTX::BI__nvvm_atom_add_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sub_gen_i:
  case NVPTX::BI__nvvm_atom_sub_gen_l:
  case NVPTX::BI__nvvm_atom_sub_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_and_gen_i:
  case NVPTX::BI__nvvm_atom_and_gen_l:
  case NVPTX::BI__nvvm_atom_and_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_or_gen_i:
  case NVPTX::BI__nvvm_atom_or_gen_l:
  case NVPTX::BI__nvvm_atom_or_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_xor_gen_i:
  case NVPTX::BI__nvvm_atom_xor_gen_l:
  case NVPTX::BI__nvvm_atom_xor_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_xchg_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_max_gen_i:
  case NVPTX::BI__nvvm_atom_max_gen_l:
  case NVPTX::BI__nvvm_atom_max_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_max_gen_ui:
  case NVPTX::BI__nvvm_atom_max_gen_ul:
  case NVPTX::BI__nvvm_atom_max_gen_ull:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_min_gen_i:
  case NVPTX::BI__nvvm_atom_min_gen_l:
  case NVPTX::BI__nvvm_atom_min_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_min_gen_ui:
  case NVPTX::BI__nvvm_atom_min_gen_ul:
  case NVPTX::BI__nvvm_atom_min_gen_ull:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cas_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
    // success flag.
  case NVPTX::BI__nvvm_atom_add_gen_f:
  case NVPTX::BI__nvvm_atom_add_gen_d:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_inc_gen_ui:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_dec_gen_ui:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
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
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
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
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ldu_f:
  case NVPTX::BI__nvvm_ldu_f2:
  case NVPTX::BI__nvvm_ldu_f4:
  case NVPTX::BI__nvvm_ldu_d:
  case NVPTX::BI__nvvm_ldu_d2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_add_gen_i:
  case NVPTX::BI__nvvm_atom_cta_add_gen_l:
  case NVPTX::BI__nvvm_atom_cta_add_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_add_gen_i:
  case NVPTX::BI__nvvm_atom_sys_add_gen_l:
  case NVPTX::BI__nvvm_atom_sys_add_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_add_gen_f:
  case NVPTX::BI__nvvm_atom_cta_add_gen_d:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_add_gen_f:
  case NVPTX::BI__nvvm_atom_sys_add_gen_d:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_max_gen_i:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_max_gen_l:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ull:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_max_gen_i:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_max_gen_l:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ull:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_min_gen_i:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_min_gen_l:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ull:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_min_gen_i:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_min_gen_l:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ull:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_inc_gen_ui:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_dec_gen_ui:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_inc_gen_ui:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_dec_gen_ui:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_and_gen_i:
  case NVPTX::BI__nvvm_atom_cta_and_gen_l:
  case NVPTX::BI__nvvm_atom_cta_and_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_and_gen_i:
  case NVPTX::BI__nvvm_atom_sys_and_gen_l:
  case NVPTX::BI__nvvm_atom_sys_and_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_or_gen_i:
  case NVPTX::BI__nvvm_atom_cta_or_gen_l:
  case NVPTX::BI__nvvm_atom_cta_or_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_or_gen_i:
  case NVPTX::BI__nvvm_atom_sys_or_gen_l:
  case NVPTX::BI__nvvm_atom_sys_or_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_xor_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_xor_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_cta_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_atom_sys_cas_gen_us:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_i:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_l:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_ll:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_match_all_sync_i32p:
  case NVPTX::BI__nvvm_match_all_sync_i64p:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  // FP MMA loads
  case NVPTX::BI__hmma_m16n16k16_ld_a:
  case NVPTX::BI__hmma_m16n16k16_ld_b:
  case NVPTX::BI__hmma_m16n16k16_ld_c_f16:
  case NVPTX::BI__hmma_m16n16k16_ld_c_f32:
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
  case NVPTX::BI__nvvm_ex2_approx_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ex2_approx_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
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
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_ftz_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_ftz_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_relu_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_relu_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_sat_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_sat_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_bf16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_bf16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_relu_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_relu_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_relu_bf16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fma_rn_oob_relu_bf16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_nan_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_nan_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_nan_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_nan_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fabs_f:
  case NVPTX::BI__nvvm_abs_bf16:
  case NVPTX::BI__nvvm_abs_bf16x2:
  case NVPTX::BI__nvvm_fabs_f16:
  case NVPTX::BI__nvvm_fabs_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fabs_ftz_f:
  case NVPTX::BI__nvvm_fabs_ftz_f16:
  case NVPTX::BI__nvvm_fabs_ftz_f16x2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_fabs_d:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ex2_approx_d:
  case NVPTX::BI__nvvm_ex2_approx_f:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ex2_approx_ftz_f:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ldg_h:
  case NVPTX::BI__nvvm_ldg_h2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_ldu_h:
  case NVPTX::BI__nvvm_ldu_h2:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
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
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_bar0_or:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};
  case NVPTX::BI__nvvm_bar0_popc:
    cgm.errorNYI(expr->getSourceRange(),
                 std::string("unimplemented NVPTX builtin call: ") +
                     getContext().BuiltinInfo.getName(builtinId));
    return mlir::Value{};

  default:
    return std::nullopt;
  }
}
