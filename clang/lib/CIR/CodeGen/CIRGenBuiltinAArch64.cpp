//===---- CIRGenBuiltinAArch64.cpp - Emit CIR for AArch64 builtins --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit ARM64 Builtin calls as CIR or a function call
// to be later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "UnimplementedFeatureGuarding.h"

// TODO(cir): we shouldn't need this but we currently reuse intrinsic IDs for
// convenience.
#include "llvm/IR/Intrinsics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

mlir::Value
CIRGenFunction::buildAArch64BuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                        llvm::Triple::ArchType Arch) {
  if (BuiltinID >= clang::AArch64::FirstSVEBuiltin &&
      BuiltinID <= clang::AArch64::LastSVEBuiltin)
    llvm_unreachable("NYI");

  if (BuiltinID >= clang::AArch64::FirstSMEBuiltin &&
      BuiltinID <= clang::AArch64::LastSMEBuiltin)
    llvm_unreachable("NYI");

  if (BuiltinID == Builtin::BI__builtin_cpu_supports)
    llvm_unreachable("NYI");

  unsigned HintID = static_cast<unsigned>(-1);
  switch (BuiltinID) {
  default:
    break;
  case clang::AArch64::BI__builtin_arm_nop:
    HintID = 0;
    break;
  case clang::AArch64::BI__builtin_arm_yield:
  case clang::AArch64::BI__yield:
    HintID = 1;
    break;
  case clang::AArch64::BI__builtin_arm_wfe:
  case clang::AArch64::BI__wfe:
    HintID = 2;
    break;
  case clang::AArch64::BI__builtin_arm_wfi:
  case clang::AArch64::BI__wfi:
    HintID = 3;
    break;
  case clang::AArch64::BI__builtin_arm_sev:
  case clang::AArch64::BI__sev:
    HintID = 4;
    break;
  case clang::AArch64::BI__builtin_arm_sevl:
  case clang::AArch64::BI__sevl:
    HintID = 5;
    break;
  }

  if (HintID != static_cast<unsigned>(-1)) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_trap) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_get_sme_state) {
    // Create call to __arm_sme_state and store the results to the two pointers.
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rbit) {
    assert((getContext().getTypeSize(E->getType()) == 32) &&
           "rbit of unusual size!");
    llvm_unreachable("NYI");
  }
  if (BuiltinID == clang::AArch64::BI__builtin_arm_rbit64) {
    assert((getContext().getTypeSize(E->getType()) == 64) &&
           "rbit of unusual size!");
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_clz ||
      BuiltinID == clang::AArch64::BI__builtin_arm_clz64) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_cls) {
    llvm_unreachable("NYI");
  }
  if (BuiltinID == clang::AArch64::BI__builtin_arm_cls64) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint32zf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint32z) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint64zf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint64z) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint32xf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint32x) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint64xf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint64x) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_jcvt) {
    assert((getContext().getTypeSize(E->getType()) == 32) &&
           "__jcvt of unusual size!");
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_ld64b ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64b ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64bv ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64bv0) {
    llvm_unreachable("NYI");

    if (BuiltinID == clang::AArch64::BI__builtin_arm_ld64b) {
      // Load from the address via an LLVM intrinsic, receiving a
      // tuple of 8 i64 words, and store each one to ValPtr.
      llvm_unreachable("NYI");
    } else {
      // Load 8 i64 words from ValPtr, and store them to the address
      // via an LLVM intrinsic.
      llvm_unreachable("NYI");
    }
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rndr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rndrrs) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__clear_cache) {
    assert(E->getNumArgs() == 2 && "__clear_cache takes 2 arguments");
    llvm_unreachable("NYI");
  }

  if ((BuiltinID == clang::AArch64::BI__builtin_arm_ldrex ||
       BuiltinID == clang::AArch64::BI__builtin_arm_ldaex) &&
      getContext().getTypeSize(E->getType()) == 128) {
    llvm_unreachable("NYI");
  } else if (BuiltinID == clang::AArch64::BI__builtin_arm_ldrex ||
             BuiltinID == clang::AArch64::BI__builtin_arm_ldaex) {
    llvm_unreachable("NYI");
  }

  if ((BuiltinID == clang::AArch64::BI__builtin_arm_strex ||
       BuiltinID == clang::AArch64::BI__builtin_arm_stlex) &&
      getContext().getTypeSize(E->getArg(0)->getType()) == 128) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_strex ||
      BuiltinID == clang::AArch64::BI__builtin_arm_stlex) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__getReg) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__break) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_clrex) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI_ReadWriteBarrier)
    llvm_unreachable("NYI");

  // CRC32
  // FIXME(cir): get rid of LLVM when this gets implemented.
  llvm::Intrinsic::ID CRCIntrinsicID = llvm::Intrinsic::not_intrinsic;
  switch (BuiltinID) {
  case clang::AArch64::BI__builtin_arm_crc32b:
  case clang::AArch64::BI__builtin_arm_crc32cb:
  case clang::AArch64::BI__builtin_arm_crc32h:
  case clang::AArch64::BI__builtin_arm_crc32ch:
  case clang::AArch64::BI__builtin_arm_crc32w:
  case clang::AArch64::BI__builtin_arm_crc32cw:
  case clang::AArch64::BI__builtin_arm_crc32d:
  case clang::AArch64::BI__builtin_arm_crc32cd:
    llvm_unreachable("NYI");
  }

  if (CRCIntrinsicID != llvm::Intrinsic::not_intrinsic) {
    llvm_unreachable("NYI");
  }

  // Memory Operations (MOPS)
  if (BuiltinID == AArch64::BI__builtin_arm_mops_memset_tag) {
    llvm_unreachable("NYI");
  }

  // Memory Tagging Extensions (MTE) Intrinsics
  // FIXME(cir): get rid of LLVM when this gets implemented.
  llvm::Intrinsic::ID MTEIntrinsicID = llvm::Intrinsic::not_intrinsic;
  switch (BuiltinID) {
  case clang::AArch64::BI__builtin_arm_irg:
  case clang::AArch64::BI__builtin_arm_addg:
  case clang::AArch64::BI__builtin_arm_gmi:
  case clang::AArch64::BI__builtin_arm_ldg:
  case clang::AArch64::BI__builtin_arm_stg:
  case clang::AArch64::BI__builtin_arm_subp:
    llvm_unreachable("NYI");
  }

  if (MTEIntrinsicID != llvm::Intrinsic::not_intrinsic) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsr64 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsrp ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr64 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr128 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsrp) {

    llvm_unreachable("NYI");
    if (BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsr64 ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsrp)
      llvm_unreachable("NYI");

    bool IsPointerBuiltin = BuiltinID == clang::AArch64::BI__builtin_arm_rsrp ||
                            BuiltinID == clang::AArch64::BI__builtin_arm_wsrp;

    bool Is32Bit = BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
                   BuiltinID == clang::AArch64::BI__builtin_arm_wsr;

    bool Is128Bit = BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
                    BuiltinID == clang::AArch64::BI__builtin_arm_wsr128;

    if (Is32Bit) {
      llvm_unreachable("NYI");
    } else if (Is128Bit) {
      llvm_unreachable("NYI");
    } else if (IsPointerBuiltin) {
      llvm_unreachable("NYI");
    } else {
      llvm_unreachable("NYI");
    };

    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI_ReadStatusReg ||
      BuiltinID == clang::AArch64::BI_WriteStatusReg) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI_AddressOfReturnAddress) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_sponentry) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__mulh ||
      BuiltinID == clang::AArch64::BI__umulh) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == AArch64::BI__writex18byte ||
      BuiltinID == AArch64::BI__writex18word ||
      BuiltinID == AArch64::BI__writex18dword ||
      BuiltinID == AArch64::BI__writex18qword) {
    // Read x18 as i8*
    llvm_unreachable("NYI");
  }

  if (BuiltinID == AArch64::BI__readx18byte ||
      BuiltinID == AArch64::BI__readx18word ||
      BuiltinID == AArch64::BI__readx18dword ||
      BuiltinID == AArch64::BI__readx18qword) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == AArch64::BI_CopyDoubleFromInt64 ||
      BuiltinID == AArch64::BI_CopyFloatFromInt32 ||
      BuiltinID == AArch64::BI_CopyInt32FromFloat ||
      BuiltinID == AArch64::BI_CopyInt64FromDouble) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == AArch64::BI_CountLeadingOnes ||
      BuiltinID == AArch64::BI_CountLeadingOnes64 ||
      BuiltinID == AArch64::BI_CountLeadingZeros ||
      BuiltinID == AArch64::BI_CountLeadingZeros64) {
    llvm_unreachable("NYI");

    if (BuiltinID == AArch64::BI_CountLeadingOnes ||
        BuiltinID == AArch64::BI_CountLeadingOnes64)
      llvm_unreachable("NYI");

    llvm_unreachable("NYI");
  }

  if (BuiltinID == AArch64::BI_CountLeadingSigns ||
      BuiltinID == AArch64::BI_CountLeadingSigns64) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == AArch64::BI_CountOneBits ||
      BuiltinID == AArch64::BI_CountOneBits64) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == AArch64::BI__prefetch) {
    llvm_unreachable("NYI");
  }

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  assert(!UnimplementedFeature::translateAarch64ToMsvcIntrin());

  // Some intrinsics are equivalent - if they are use the base intrinsic ID.
  assert(!UnimplementedFeature::neonEquivalentIntrinsicMap());

  // Find out if any arguments are required to be integer constant
  // expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  llvm::SmallVector<mlir::Value, 4> Ops;
  for (unsigned i = 0, e = E->getNumArgs() - 1; i != e; i++) {
    if (i == 0) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld1_v:
      case NEON::BI__builtin_neon_vld1q_v:
      case NEON::BI__builtin_neon_vld1_dup_v:
      case NEON::BI__builtin_neon_vld1q_dup_v:
      case NEON::BI__builtin_neon_vld1_lane_v:
      case NEON::BI__builtin_neon_vld1q_lane_v:
      case NEON::BI__builtin_neon_vst1_v:
      case NEON::BI__builtin_neon_vst1q_v:
      case NEON::BI__builtin_neon_vst1_lane_v:
      case NEON::BI__builtin_neon_vst1q_lane_v:
      case NEON::BI__builtin_neon_vldap1_lane_s64:
      case NEON::BI__builtin_neon_vldap1q_lane_s64:
      case NEON::BI__builtin_neon_vstl1_lane_s64:
      case NEON::BI__builtin_neon_vstl1q_lane_s64:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        llvm_unreachable("NYI");
      }
    }
    llvm_unreachable("NYI");
  }

  assert(!UnimplementedFeature::arm64SISDIntrinsicMap());

  const Expr *Arg = E->getArg(E->getNumArgs() - 1);
  NeonTypeFlags Type(0);
  if (std::optional<llvm::APSInt> Result =
          Arg->getIntegerConstantExpr(getContext()))
    // Determine the type of this overloaded NEON intrinsic.
    Type = NeonTypeFlags(Result->getZExtValue());

  bool usgn = Type.isUnsigned();

  // Handle non-overloaded intrinsics first.
  switch (BuiltinID) {
  default:
    break;
  case NEON::BI__builtin_neon_vabsh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vaddq_p128: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vldrq_p128: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vstrq_p128: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcvts_f32_u32:
  case NEON::BI__builtin_neon_vcvtd_f64_u64:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vcvts_f32_s32:
  case NEON::BI__builtin_neon_vcvtd_f64_s64: {
    if (usgn)
      llvm_unreachable("NYI");
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcvth_f16_u16:
  case NEON::BI__builtin_neon_vcvth_f16_u32:
  case NEON::BI__builtin_neon_vcvth_f16_u64:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vcvth_f16_s16:
  case NEON::BI__builtin_neon_vcvth_f16_s32:
  case NEON::BI__builtin_neon_vcvth_f16_s64: {
    if (usgn)
      llvm_unreachable("NYI");
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcvtah_u16_f16:
  case NEON::BI__builtin_neon_vcvtmh_u16_f16:
  case NEON::BI__builtin_neon_vcvtnh_u16_f16:
  case NEON::BI__builtin_neon_vcvtph_u16_f16:
  case NEON::BI__builtin_neon_vcvth_u16_f16:
  case NEON::BI__builtin_neon_vcvtah_s16_f16:
  case NEON::BI__builtin_neon_vcvtmh_s16_f16:
  case NEON::BI__builtin_neon_vcvtnh_s16_f16:
  case NEON::BI__builtin_neon_vcvtph_s16_f16:
  case NEON::BI__builtin_neon_vcvth_s16_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcaleh_f16:
  case NEON::BI__builtin_neon_vcalth_f16:
  case NEON::BI__builtin_neon_vcageh_f16:
  case NEON::BI__builtin_neon_vcagth_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcvth_n_s16_f16:
  case NEON::BI__builtin_neon_vcvth_n_u16_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcvth_n_f16_s16:
  case NEON::BI__builtin_neon_vcvth_n_f16_u16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vpaddd_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vpaddd_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vpadds_f32: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vceqzd_s64:
  case NEON::BI__builtin_neon_vceqzd_f64:
  case NEON::BI__builtin_neon_vceqzs_f32:
  case NEON::BI__builtin_neon_vceqzh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vcgezd_s64:
  case NEON::BI__builtin_neon_vcgezd_f64:
  case NEON::BI__builtin_neon_vcgezs_f32:
  case NEON::BI__builtin_neon_vcgezh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vclezd_s64:
  case NEON::BI__builtin_neon_vclezd_f64:
  case NEON::BI__builtin_neon_vclezs_f32:
  case NEON::BI__builtin_neon_vclezh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vcgtzd_s64:
  case NEON::BI__builtin_neon_vcgtzd_f64:
  case NEON::BI__builtin_neon_vcgtzs_f32:
  case NEON::BI__builtin_neon_vcgtzh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vcltzd_s64:
  case NEON::BI__builtin_neon_vcltzd_f64:
  case NEON::BI__builtin_neon_vcltzs_f32:
  case NEON::BI__builtin_neon_vcltzh_f16:
    llvm_unreachable("NYI");

  case NEON::BI__builtin_neon_vceqzd_u64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vceqd_f64:
  case NEON::BI__builtin_neon_vcled_f64:
  case NEON::BI__builtin_neon_vcltd_f64:
  case NEON::BI__builtin_neon_vcged_f64:
  case NEON::BI__builtin_neon_vcgtd_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vceqs_f32:
  case NEON::BI__builtin_neon_vcles_f32:
  case NEON::BI__builtin_neon_vclts_f32:
  case NEON::BI__builtin_neon_vcges_f32:
  case NEON::BI__builtin_neon_vcgts_f32: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vceqh_f16:
  case NEON::BI__builtin_neon_vcleh_f16:
  case NEON::BI__builtin_neon_vclth_f16:
  case NEON::BI__builtin_neon_vcgeh_f16:
  case NEON::BI__builtin_neon_vcgth_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vceqd_s64:
  case NEON::BI__builtin_neon_vceqd_u64:
  case NEON::BI__builtin_neon_vcgtd_s64:
  case NEON::BI__builtin_neon_vcgtd_u64:
  case NEON::BI__builtin_neon_vcltd_s64:
  case NEON::BI__builtin_neon_vcltd_u64:
  case NEON::BI__builtin_neon_vcged_u64:
  case NEON::BI__builtin_neon_vcged_s64:
  case NEON::BI__builtin_neon_vcled_u64:
  case NEON::BI__builtin_neon_vcled_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtstd_s64:
  case NEON::BI__builtin_neon_vtstd_u64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_bf16:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_bf16:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vset_lane_f64:
    // The vector type needs a cast for the v1f64 variant.
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vsetq_lane_f64:
    // The vector type needs a cast for the v2f64 variant.
    llvm_unreachable("NYI");

  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vdupb_lane_i8:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vdupb_laneq_i8:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vduph_lane_i16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vduph_laneq_i16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vdups_lane_i32:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vdups_lane_f32:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vdups_laneq_i32:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vdupd_lane_i64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vdupd_lane_f64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vdupd_laneq_i64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vget_lane_f32:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vget_lane_f64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vgetq_lane_f32:
  case NEON::BI__builtin_neon_vdups_laneq_f32:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vgetq_lane_f64:
  case NEON::BI__builtin_neon_vdupd_laneq_f64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vaddh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vsubh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vmulh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vdivh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vfmah_f16:
    // NEON intrinsic puts accumulator first, unlike the LLVM fma.
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vfmsh_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddd_s64:
  case NEON::BI__builtin_neon_vaddd_u64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vsubd_s64:
  case NEON::BI__builtin_neon_vsubd_u64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vqdmlalh_s16:
  case NEON::BI__builtin_neon_vqdmlslh_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqshlud_n_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqshld_n_u64:
  case NEON::BI__builtin_neon_vqshld_n_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrshrd_n_u64:
  case NEON::BI__builtin_neon_vrshrd_n_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrsrad_n_u64:
  case NEON::BI__builtin_neon_vrsrad_n_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vshld_n_s64:
  case NEON::BI__builtin_neon_vshld_n_u64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vshrd_n_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vshrd_n_u64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsrad_n_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsrad_n_u64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqdmlalh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlalh_laneq_s16:
  case NEON::BI__builtin_neon_vqdmlslh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlslh_laneq_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqdmlals_s32:
  case NEON::BI__builtin_neon_vqdmlsls_s32: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqdmlals_lane_s32:
  case NEON::BI__builtin_neon_vqdmlals_laneq_s32:
  case NEON::BI__builtin_neon_vqdmlsls_lane_s32:
  case NEON::BI__builtin_neon_vqdmlsls_laneq_s32: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vget_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vgetq_lane_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_f16: {
    llvm_unreachable("NYI");
  }

  case clang::AArch64::BI_InterlockedAdd:
  case clang::AArch64::BI_InterlockedAdd64: {
    llvm_unreachable("NYI");
  }
  }

  // From here on it's pure NEON based
  assert(UnimplementedFeature::getNeonType() && "NYI");
  return {};
}
