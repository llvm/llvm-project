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

#include <utility>

#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"

// TODO(cir): once all builtins are covered, decide whether we still
// need to use LLVM intrinsics or if there's a better approach to follow. Right
// now the intrinsics are reused to make it convenient to encode all thousands
// of them and passing down to LLVM lowering.
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace llvm;

enum {
  AddRetType = (1 << 0),
  Add1ArgType = (1 << 1),
  Add2ArgTypes = (1 << 2),

  VectorizeRetType = (1 << 3),
  VectorizeArgTypes = (1 << 4),

  InventFloatType = (1 << 5),
  UnsignedAlts = (1 << 6),

  Use64BitVectors = (1 << 7),
  Use128BitVectors = (1 << 8),

  Vectorize1ArgType = Add1ArgType | VectorizeArgTypes,
  VectorRet = AddRetType | VectorizeRetType,
  VectorRetGetArgs01 =
      AddRetType | Add2ArgTypes | VectorizeRetType | VectorizeArgTypes,
  FpCmpzModifiers =
      AddRetType | VectorizeRetType | Add1ArgType | InventFloatType
};

namespace {
struct ARMVectorIntrinsicInfo {
  const char *NameHint;
  unsigned BuiltinID;
  unsigned LLVMIntrinsic;
  unsigned AltLLVMIntrinsic;
  uint64_t TypeModifier;

  bool operator<(unsigned RHSBuiltinID) const {
    return BuiltinID < RHSBuiltinID;
  }
  bool operator<(const ARMVectorIntrinsicInfo &TE) const {
    return BuiltinID < TE.BuiltinID;
  }
};
} // end anonymous namespace

#define NEONMAP0(NameBase)                                                     \
  {#NameBase, NEON::BI__builtin_neon_##NameBase, 0, 0, 0}

#define NEONMAP1(NameBase, LLVMIntrinsic, TypeModifier)                        \
  {#NameBase, NEON::BI__builtin_neon_##NameBase, Intrinsic::LLVMIntrinsic, 0,  \
   TypeModifier}

#define NEONMAP2(NameBase, LLVMIntrinsic, AltLLVMIntrinsic, TypeModifier)      \
  {#NameBase, NEON::BI__builtin_neon_##NameBase, Intrinsic::LLVMIntrinsic,     \
   Intrinsic::AltLLVMIntrinsic, TypeModifier}

static const ARMVectorIntrinsicInfo AArch64SIMDIntrinsicMap[] = {
    NEONMAP0(splat_lane_v),
    NEONMAP0(splat_laneq_v),
    NEONMAP0(splatq_lane_v),
    NEONMAP0(splatq_laneq_v),
    NEONMAP1(vabs_v, aarch64_neon_abs, 0),
    NEONMAP1(vabsq_v, aarch64_neon_abs, 0),
    NEONMAP0(vadd_v),
    NEONMAP0(vaddhn_v),
    NEONMAP0(vaddq_p128),
    NEONMAP0(vaddq_v),
    NEONMAP1(vaesdq_u8, aarch64_crypto_aesd, 0),
    NEONMAP1(vaeseq_u8, aarch64_crypto_aese, 0),
    NEONMAP1(vaesimcq_u8, aarch64_crypto_aesimc, 0),
    NEONMAP1(vaesmcq_u8, aarch64_crypto_aesmc, 0),
    NEONMAP2(vbcaxq_s16, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vbcaxq_s32, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vbcaxq_s64, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vbcaxq_s8, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vbcaxq_u16, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vbcaxq_u32, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vbcaxq_u64, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vbcaxq_u8, aarch64_crypto_bcaxu, aarch64_crypto_bcaxs,
             Add1ArgType | UnsignedAlts),
    NEONMAP1(vbfdot_f32, aarch64_neon_bfdot, 0),
    NEONMAP1(vbfdotq_f32, aarch64_neon_bfdot, 0),
    NEONMAP1(vbfmlalbq_f32, aarch64_neon_bfmlalb, 0),
    NEONMAP1(vbfmlaltq_f32, aarch64_neon_bfmlalt, 0),
    NEONMAP1(vbfmmlaq_f32, aarch64_neon_bfmmla, 0),
    NEONMAP1(vcadd_rot270_f16, aarch64_neon_vcadd_rot270, Add1ArgType),
    NEONMAP1(vcadd_rot270_f32, aarch64_neon_vcadd_rot270, Add1ArgType),
    NEONMAP1(vcadd_rot90_f16, aarch64_neon_vcadd_rot90, Add1ArgType),
    NEONMAP1(vcadd_rot90_f32, aarch64_neon_vcadd_rot90, Add1ArgType),
    NEONMAP1(vcaddq_rot270_f16, aarch64_neon_vcadd_rot270, Add1ArgType),
    NEONMAP1(vcaddq_rot270_f32, aarch64_neon_vcadd_rot270, Add1ArgType),
    NEONMAP1(vcaddq_rot270_f64, aarch64_neon_vcadd_rot270, Add1ArgType),
    NEONMAP1(vcaddq_rot90_f16, aarch64_neon_vcadd_rot90, Add1ArgType),
    NEONMAP1(vcaddq_rot90_f32, aarch64_neon_vcadd_rot90, Add1ArgType),
    NEONMAP1(vcaddq_rot90_f64, aarch64_neon_vcadd_rot90, Add1ArgType),
    NEONMAP1(vcage_v, aarch64_neon_facge, 0),
    NEONMAP1(vcageq_v, aarch64_neon_facge, 0),
    NEONMAP1(vcagt_v, aarch64_neon_facgt, 0),
    NEONMAP1(vcagtq_v, aarch64_neon_facgt, 0),
    NEONMAP1(vcale_v, aarch64_neon_facge, 0),
    NEONMAP1(vcaleq_v, aarch64_neon_facge, 0),
    NEONMAP1(vcalt_v, aarch64_neon_facgt, 0),
    NEONMAP1(vcaltq_v, aarch64_neon_facgt, 0),
    NEONMAP0(vceqz_v),
    NEONMAP0(vceqzq_v),
    NEONMAP0(vcgez_v),
    NEONMAP0(vcgezq_v),
    NEONMAP0(vcgtz_v),
    NEONMAP0(vcgtzq_v),
    NEONMAP0(vclez_v),
    NEONMAP0(vclezq_v),
    NEONMAP1(vcls_v, aarch64_neon_cls, Add1ArgType),
    NEONMAP1(vclsq_v, aarch64_neon_cls, Add1ArgType),
    NEONMAP0(vcltz_v),
    NEONMAP0(vcltzq_v),
    NEONMAP1(vclz_v, ctlz, Add1ArgType),
    NEONMAP1(vclzq_v, ctlz, Add1ArgType),
    NEONMAP1(vcmla_f16, aarch64_neon_vcmla_rot0, Add1ArgType),
    NEONMAP1(vcmla_f32, aarch64_neon_vcmla_rot0, Add1ArgType),
    NEONMAP1(vcmla_rot180_f16, aarch64_neon_vcmla_rot180, Add1ArgType),
    NEONMAP1(vcmla_rot180_f32, aarch64_neon_vcmla_rot180, Add1ArgType),
    NEONMAP1(vcmla_rot270_f16, aarch64_neon_vcmla_rot270, Add1ArgType),
    NEONMAP1(vcmla_rot270_f32, aarch64_neon_vcmla_rot270, Add1ArgType),
    NEONMAP1(vcmla_rot90_f16, aarch64_neon_vcmla_rot90, Add1ArgType),
    NEONMAP1(vcmla_rot90_f32, aarch64_neon_vcmla_rot90, Add1ArgType),
    NEONMAP1(vcmlaq_f16, aarch64_neon_vcmla_rot0, Add1ArgType),
    NEONMAP1(vcmlaq_f32, aarch64_neon_vcmla_rot0, Add1ArgType),
    NEONMAP1(vcmlaq_f64, aarch64_neon_vcmla_rot0, Add1ArgType),
    NEONMAP1(vcmlaq_rot180_f16, aarch64_neon_vcmla_rot180, Add1ArgType),
    NEONMAP1(vcmlaq_rot180_f32, aarch64_neon_vcmla_rot180, Add1ArgType),
    NEONMAP1(vcmlaq_rot180_f64, aarch64_neon_vcmla_rot180, Add1ArgType),
    NEONMAP1(vcmlaq_rot270_f16, aarch64_neon_vcmla_rot270, Add1ArgType),
    NEONMAP1(vcmlaq_rot270_f32, aarch64_neon_vcmla_rot270, Add1ArgType),
    NEONMAP1(vcmlaq_rot270_f64, aarch64_neon_vcmla_rot270, Add1ArgType),
    NEONMAP1(vcmlaq_rot90_f16, aarch64_neon_vcmla_rot90, Add1ArgType),
    NEONMAP1(vcmlaq_rot90_f32, aarch64_neon_vcmla_rot90, Add1ArgType),
    NEONMAP1(vcmlaq_rot90_f64, aarch64_neon_vcmla_rot90, Add1ArgType),
    NEONMAP1(vcnt_v, ctpop, Add1ArgType),
    NEONMAP1(vcntq_v, ctpop, Add1ArgType),
    NEONMAP1(vcvt_f16_f32, aarch64_neon_vcvtfp2hf, 0),
    NEONMAP0(vcvt_f16_s16),
    NEONMAP0(vcvt_f16_u16),
    NEONMAP1(vcvt_f32_f16, aarch64_neon_vcvthf2fp, 0),
    NEONMAP0(vcvt_f32_v),
    NEONMAP1(vcvt_n_f16_s16, aarch64_neon_vcvtfxs2fp, 0),
    NEONMAP1(vcvt_n_f16_u16, aarch64_neon_vcvtfxu2fp, 0),
    NEONMAP2(vcvt_n_f32_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp, 0),
    NEONMAP2(vcvt_n_f64_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp, 0),
    NEONMAP1(vcvt_n_s16_f16, aarch64_neon_vcvtfp2fxs, 0),
    NEONMAP1(vcvt_n_s32_v, aarch64_neon_vcvtfp2fxs, 0),
    NEONMAP1(vcvt_n_s64_v, aarch64_neon_vcvtfp2fxs, 0),
    NEONMAP1(vcvt_n_u16_f16, aarch64_neon_vcvtfp2fxu, 0),
    NEONMAP1(vcvt_n_u32_v, aarch64_neon_vcvtfp2fxu, 0),
    NEONMAP1(vcvt_n_u64_v, aarch64_neon_vcvtfp2fxu, 0),
    NEONMAP0(vcvtq_f16_s16),
    NEONMAP0(vcvtq_f16_u16),
    NEONMAP0(vcvtq_f32_v),
    NEONMAP0(vcvtq_high_bf16_f32),
    NEONMAP0(vcvtq_low_bf16_f32),
    NEONMAP1(vcvtq_n_f16_s16, aarch64_neon_vcvtfxs2fp, 0),
    NEONMAP1(vcvtq_n_f16_u16, aarch64_neon_vcvtfxu2fp, 0),
    NEONMAP2(vcvtq_n_f32_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp,
             0),
    NEONMAP2(vcvtq_n_f64_v, aarch64_neon_vcvtfxu2fp, aarch64_neon_vcvtfxs2fp,
             0),
    NEONMAP1(vcvtq_n_s16_f16, aarch64_neon_vcvtfp2fxs, 0),
    NEONMAP1(vcvtq_n_s32_v, aarch64_neon_vcvtfp2fxs, 0),
    NEONMAP1(vcvtq_n_s64_v, aarch64_neon_vcvtfp2fxs, 0),
    NEONMAP1(vcvtq_n_u16_f16, aarch64_neon_vcvtfp2fxu, 0),
    NEONMAP1(vcvtq_n_u32_v, aarch64_neon_vcvtfp2fxu, 0),
    NEONMAP1(vcvtq_n_u64_v, aarch64_neon_vcvtfp2fxu, 0),
    NEONMAP1(vcvtx_f32_v, aarch64_neon_fcvtxn, AddRetType | Add1ArgType),
    NEONMAP1(vdot_s32, aarch64_neon_sdot, 0),
    NEONMAP1(vdot_u32, aarch64_neon_udot, 0),
    NEONMAP1(vdotq_s32, aarch64_neon_sdot, 0),
    NEONMAP1(vdotq_u32, aarch64_neon_udot, 0),
    NEONMAP2(veor3q_s16, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(veor3q_s32, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(veor3q_s64, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(veor3q_s8, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(veor3q_u16, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(veor3q_u32, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(veor3q_u64, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(veor3q_u8, aarch64_crypto_eor3u, aarch64_crypto_eor3s,
             Add1ArgType | UnsignedAlts),
    NEONMAP0(vext_v),
    NEONMAP0(vextq_v),
    NEONMAP0(vfma_v),
    NEONMAP0(vfmaq_v),
    NEONMAP1(vfmlal_high_f16, aarch64_neon_fmlal2, 0),
    NEONMAP1(vfmlal_low_f16, aarch64_neon_fmlal, 0),
    NEONMAP1(vfmlalq_high_f16, aarch64_neon_fmlal2, 0),
    NEONMAP1(vfmlalq_low_f16, aarch64_neon_fmlal, 0),
    NEONMAP1(vfmlsl_high_f16, aarch64_neon_fmlsl2, 0),
    NEONMAP1(vfmlsl_low_f16, aarch64_neon_fmlsl, 0),
    NEONMAP1(vfmlslq_high_f16, aarch64_neon_fmlsl2, 0),
    NEONMAP1(vfmlslq_low_f16, aarch64_neon_fmlsl, 0),
    NEONMAP2(vhadd_v, aarch64_neon_uhadd, aarch64_neon_shadd,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vhaddq_v, aarch64_neon_uhadd, aarch64_neon_shadd,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vhsub_v, aarch64_neon_uhsub, aarch64_neon_shsub,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vhsubq_v, aarch64_neon_uhsub, aarch64_neon_shsub,
             Add1ArgType | UnsignedAlts),
    NEONMAP1(vld1_x2_v, aarch64_neon_ld1x2, 0),
    NEONMAP1(vld1_x3_v, aarch64_neon_ld1x3, 0),
    NEONMAP1(vld1_x4_v, aarch64_neon_ld1x4, 0),
    NEONMAP1(vld1q_x2_v, aarch64_neon_ld1x2, 0),
    NEONMAP1(vld1q_x3_v, aarch64_neon_ld1x3, 0),
    NEONMAP1(vld1q_x4_v, aarch64_neon_ld1x4, 0),
    NEONMAP1(vmmlaq_s32, aarch64_neon_smmla, 0),
    NEONMAP1(vmmlaq_u32, aarch64_neon_ummla, 0),
    NEONMAP0(vmovl_v),
    NEONMAP0(vmovn_v),
    NEONMAP1(vmul_v, aarch64_neon_pmul, Add1ArgType),
    NEONMAP1(vmulq_v, aarch64_neon_pmul, Add1ArgType),
    NEONMAP1(vpadd_v, aarch64_neon_addp, Add1ArgType),
    NEONMAP2(vpaddl_v, aarch64_neon_uaddlp, aarch64_neon_saddlp, UnsignedAlts),
    NEONMAP2(vpaddlq_v, aarch64_neon_uaddlp, aarch64_neon_saddlp, UnsignedAlts),
    NEONMAP1(vpaddq_v, aarch64_neon_addp, Add1ArgType),
    NEONMAP1(vqabs_v, aarch64_neon_sqabs, Add1ArgType),
    NEONMAP1(vqabsq_v, aarch64_neon_sqabs, Add1ArgType),
    NEONMAP2(vqadd_v, aarch64_neon_uqadd, aarch64_neon_sqadd,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vqaddq_v, aarch64_neon_uqadd, aarch64_neon_sqadd,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vqdmlal_v, aarch64_neon_sqdmull, aarch64_neon_sqadd, 0),
    NEONMAP2(vqdmlsl_v, aarch64_neon_sqdmull, aarch64_neon_sqsub, 0),
    NEONMAP1(vqdmulh_lane_v, aarch64_neon_sqdmulh_lane, 0),
    NEONMAP1(vqdmulh_laneq_v, aarch64_neon_sqdmulh_laneq, 0),
    NEONMAP1(vqdmulh_v, aarch64_neon_sqdmulh, Add1ArgType),
    NEONMAP1(vqdmulhq_lane_v, aarch64_neon_sqdmulh_lane, 0),
    NEONMAP1(vqdmulhq_laneq_v, aarch64_neon_sqdmulh_laneq, 0),
    NEONMAP1(vqdmulhq_v, aarch64_neon_sqdmulh, Add1ArgType),
    NEONMAP1(vqdmull_v, aarch64_neon_sqdmull, Add1ArgType),
    NEONMAP2(vqmovn_v, aarch64_neon_uqxtn, aarch64_neon_sqxtn,
             Add1ArgType | UnsignedAlts),
    NEONMAP1(vqmovun_v, aarch64_neon_sqxtun, Add1ArgType),
    NEONMAP1(vqneg_v, aarch64_neon_sqneg, Add1ArgType),
    NEONMAP1(vqnegq_v, aarch64_neon_sqneg, Add1ArgType),
    NEONMAP1(vqrdmlah_s16, aarch64_neon_sqrdmlah, Add1ArgType),
    NEONMAP1(vqrdmlah_s32, aarch64_neon_sqrdmlah, Add1ArgType),
    NEONMAP1(vqrdmlahq_s16, aarch64_neon_sqrdmlah, Add1ArgType),
    NEONMAP1(vqrdmlahq_s32, aarch64_neon_sqrdmlah, Add1ArgType),
    NEONMAP1(vqrdmlsh_s16, aarch64_neon_sqrdmlsh, Add1ArgType),
    NEONMAP1(vqrdmlsh_s32, aarch64_neon_sqrdmlsh, Add1ArgType),
    NEONMAP1(vqrdmlshq_s16, aarch64_neon_sqrdmlsh, Add1ArgType),
    NEONMAP1(vqrdmlshq_s32, aarch64_neon_sqrdmlsh, Add1ArgType),
    NEONMAP1(vqrdmulh_lane_v, aarch64_neon_sqrdmulh_lane, 0),
    NEONMAP1(vqrdmulh_laneq_v, aarch64_neon_sqrdmulh_laneq, 0),
    NEONMAP1(vqrdmulh_v, aarch64_neon_sqrdmulh, Add1ArgType),
    NEONMAP1(vqrdmulhq_lane_v, aarch64_neon_sqrdmulh_lane, 0),
    NEONMAP1(vqrdmulhq_laneq_v, aarch64_neon_sqrdmulh_laneq, 0),
    NEONMAP1(vqrdmulhq_v, aarch64_neon_sqrdmulh, Add1ArgType),
    NEONMAP2(vqrshl_v, aarch64_neon_uqrshl, aarch64_neon_sqrshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vqrshlq_v, aarch64_neon_uqrshl, aarch64_neon_sqrshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vqshl_n_v, aarch64_neon_uqshl, aarch64_neon_sqshl, UnsignedAlts),
    NEONMAP2(vqshl_v, aarch64_neon_uqshl, aarch64_neon_sqshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vqshlq_n_v, aarch64_neon_uqshl, aarch64_neon_sqshl, UnsignedAlts),
    NEONMAP2(vqshlq_v, aarch64_neon_uqshl, aarch64_neon_sqshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP1(vqshlu_n_v, aarch64_neon_sqshlu, 0),
    NEONMAP1(vqshluq_n_v, aarch64_neon_sqshlu, 0),
    NEONMAP2(vqsub_v, aarch64_neon_uqsub, aarch64_neon_sqsub,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vqsubq_v, aarch64_neon_uqsub, aarch64_neon_sqsub,
             Add1ArgType | UnsignedAlts),
    NEONMAP1(vraddhn_v, aarch64_neon_raddhn, Add1ArgType),
    NEONMAP1(vrax1q_u64, aarch64_crypto_rax1, 0),
    NEONMAP2(vrecpe_v, aarch64_neon_frecpe, aarch64_neon_urecpe, 0),
    NEONMAP2(vrecpeq_v, aarch64_neon_frecpe, aarch64_neon_urecpe, 0),
    NEONMAP1(vrecps_v, aarch64_neon_frecps, Add1ArgType),
    NEONMAP1(vrecpsq_v, aarch64_neon_frecps, Add1ArgType),
    NEONMAP2(vrhadd_v, aarch64_neon_urhadd, aarch64_neon_srhadd,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vrhaddq_v, aarch64_neon_urhadd, aarch64_neon_srhadd,
             Add1ArgType | UnsignedAlts),
    NEONMAP1(vrnd32x_f32, aarch64_neon_frint32x, Add1ArgType),
    NEONMAP1(vrnd32x_f64, aarch64_neon_frint32x, Add1ArgType),
    NEONMAP1(vrnd32xq_f32, aarch64_neon_frint32x, Add1ArgType),
    NEONMAP1(vrnd32xq_f64, aarch64_neon_frint32x, Add1ArgType),
    NEONMAP1(vrnd32z_f32, aarch64_neon_frint32z, Add1ArgType),
    NEONMAP1(vrnd32z_f64, aarch64_neon_frint32z, Add1ArgType),
    NEONMAP1(vrnd32zq_f32, aarch64_neon_frint32z, Add1ArgType),
    NEONMAP1(vrnd32zq_f64, aarch64_neon_frint32z, Add1ArgType),
    NEONMAP1(vrnd64x_f32, aarch64_neon_frint64x, Add1ArgType),
    NEONMAP1(vrnd64x_f64, aarch64_neon_frint64x, Add1ArgType),
    NEONMAP1(vrnd64xq_f32, aarch64_neon_frint64x, Add1ArgType),
    NEONMAP1(vrnd64xq_f64, aarch64_neon_frint64x, Add1ArgType),
    NEONMAP1(vrnd64z_f32, aarch64_neon_frint64z, Add1ArgType),
    NEONMAP1(vrnd64z_f64, aarch64_neon_frint64z, Add1ArgType),
    NEONMAP1(vrnd64zq_f32, aarch64_neon_frint64z, Add1ArgType),
    NEONMAP1(vrnd64zq_f64, aarch64_neon_frint64z, Add1ArgType),
    NEONMAP0(vrndi_v),
    NEONMAP0(vrndiq_v),
    NEONMAP2(vrshl_v, aarch64_neon_urshl, aarch64_neon_srshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vrshlq_v, aarch64_neon_urshl, aarch64_neon_srshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP2(vrshr_n_v, aarch64_neon_urshl, aarch64_neon_srshl, UnsignedAlts),
    NEONMAP2(vrshrq_n_v, aarch64_neon_urshl, aarch64_neon_srshl, UnsignedAlts),
    NEONMAP2(vrsqrte_v, aarch64_neon_frsqrte, aarch64_neon_ursqrte, 0),
    NEONMAP2(vrsqrteq_v, aarch64_neon_frsqrte, aarch64_neon_ursqrte, 0),
    NEONMAP1(vrsqrts_v, aarch64_neon_frsqrts, Add1ArgType),
    NEONMAP1(vrsqrtsq_v, aarch64_neon_frsqrts, Add1ArgType),
    NEONMAP1(vrsubhn_v, aarch64_neon_rsubhn, Add1ArgType),
    NEONMAP1(vsha1su0q_u32, aarch64_crypto_sha1su0, 0),
    NEONMAP1(vsha1su1q_u32, aarch64_crypto_sha1su1, 0),
    NEONMAP1(vsha256h2q_u32, aarch64_crypto_sha256h2, 0),
    NEONMAP1(vsha256hq_u32, aarch64_crypto_sha256h, 0),
    NEONMAP1(vsha256su0q_u32, aarch64_crypto_sha256su0, 0),
    NEONMAP1(vsha256su1q_u32, aarch64_crypto_sha256su1, 0),
    NEONMAP1(vsha512h2q_u64, aarch64_crypto_sha512h2, 0),
    NEONMAP1(vsha512hq_u64, aarch64_crypto_sha512h, 0),
    NEONMAP1(vsha512su0q_u64, aarch64_crypto_sha512su0, 0),
    NEONMAP1(vsha512su1q_u64, aarch64_crypto_sha512su1, 0),
    NEONMAP0(vshl_n_v),
    NEONMAP2(vshl_v, aarch64_neon_ushl, aarch64_neon_sshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP0(vshll_n_v),
    NEONMAP0(vshlq_n_v),
    NEONMAP2(vshlq_v, aarch64_neon_ushl, aarch64_neon_sshl,
             Add1ArgType | UnsignedAlts),
    NEONMAP0(vshr_n_v),
    NEONMAP0(vshrn_n_v),
    NEONMAP0(vshrq_n_v),
    NEONMAP1(vsm3partw1q_u32, aarch64_crypto_sm3partw1, 0),
    NEONMAP1(vsm3partw2q_u32, aarch64_crypto_sm3partw2, 0),
    NEONMAP1(vsm3ss1q_u32, aarch64_crypto_sm3ss1, 0),
    NEONMAP1(vsm3tt1aq_u32, aarch64_crypto_sm3tt1a, 0),
    NEONMAP1(vsm3tt1bq_u32, aarch64_crypto_sm3tt1b, 0),
    NEONMAP1(vsm3tt2aq_u32, aarch64_crypto_sm3tt2a, 0),
    NEONMAP1(vsm3tt2bq_u32, aarch64_crypto_sm3tt2b, 0),
    NEONMAP1(vsm4ekeyq_u32, aarch64_crypto_sm4ekey, 0),
    NEONMAP1(vsm4eq_u32, aarch64_crypto_sm4e, 0),
    NEONMAP1(vst1_x2_v, aarch64_neon_st1x2, 0),
    NEONMAP1(vst1_x3_v, aarch64_neon_st1x3, 0),
    NEONMAP1(vst1_x4_v, aarch64_neon_st1x4, 0),
    NEONMAP1(vst1q_x2_v, aarch64_neon_st1x2, 0),
    NEONMAP1(vst1q_x3_v, aarch64_neon_st1x3, 0),
    NEONMAP1(vst1q_x4_v, aarch64_neon_st1x4, 0),
    NEONMAP0(vsubhn_v),
    NEONMAP0(vtst_v),
    NEONMAP0(vtstq_v),
    NEONMAP1(vusdot_s32, aarch64_neon_usdot, 0),
    NEONMAP1(vusdotq_s32, aarch64_neon_usdot, 0),
    NEONMAP1(vusmmlaq_s32, aarch64_neon_usmmla, 0),
    NEONMAP1(vxarq_u64, aarch64_crypto_xar, 0),
};

static const ARMVectorIntrinsicInfo AArch64SISDIntrinsicMap[] = {
    NEONMAP1(vabdd_f64, aarch64_sisd_fabd, Add1ArgType),
    NEONMAP1(vabds_f32, aarch64_sisd_fabd, Add1ArgType),
    NEONMAP1(vabsd_s64, aarch64_neon_abs, Add1ArgType),
    NEONMAP1(vaddlv_s32, aarch64_neon_saddlv, AddRetType | Add1ArgType),
    NEONMAP1(vaddlv_u32, aarch64_neon_uaddlv, AddRetType | Add1ArgType),
    NEONMAP1(vaddlvq_s32, aarch64_neon_saddlv, AddRetType | Add1ArgType),
    NEONMAP1(vaddlvq_u32, aarch64_neon_uaddlv, AddRetType | Add1ArgType),
    NEONMAP1(vaddv_f32, aarch64_neon_faddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddv_s32, aarch64_neon_saddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddv_u32, aarch64_neon_uaddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddvq_f32, aarch64_neon_faddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddvq_f64, aarch64_neon_faddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddvq_s32, aarch64_neon_saddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddvq_s64, aarch64_neon_saddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddvq_u32, aarch64_neon_uaddv, AddRetType | Add1ArgType),
    NEONMAP1(vaddvq_u64, aarch64_neon_uaddv, AddRetType | Add1ArgType),
    NEONMAP1(vcaged_f64, aarch64_neon_facge, AddRetType | Add1ArgType),
    NEONMAP1(vcages_f32, aarch64_neon_facge, AddRetType | Add1ArgType),
    NEONMAP1(vcagtd_f64, aarch64_neon_facgt, AddRetType | Add1ArgType),
    NEONMAP1(vcagts_f32, aarch64_neon_facgt, AddRetType | Add1ArgType),
    NEONMAP1(vcaled_f64, aarch64_neon_facge, AddRetType | Add1ArgType),
    NEONMAP1(vcales_f32, aarch64_neon_facge, AddRetType | Add1ArgType),
    NEONMAP1(vcaltd_f64, aarch64_neon_facgt, AddRetType | Add1ArgType),
    NEONMAP1(vcalts_f32, aarch64_neon_facgt, AddRetType | Add1ArgType),
    NEONMAP1(vcvtad_s64_f64, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
    NEONMAP1(vcvtad_u64_f64, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
    NEONMAP1(vcvtas_s32_f32, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
    NEONMAP1(vcvtas_u32_f32, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
    NEONMAP1(vcvtd_n_f64_s64, aarch64_neon_vcvtfxs2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvtd_n_f64_u64, aarch64_neon_vcvtfxu2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvtd_n_s64_f64, aarch64_neon_vcvtfp2fxs,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvtd_n_u64_f64, aarch64_neon_vcvtfp2fxu,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvtd_s64_f64, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
    NEONMAP1(vcvtd_u64_f64, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
    NEONMAP0(vcvth_bf16_f32),
    NEONMAP1(vcvtmd_s64_f64, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
    NEONMAP1(vcvtmd_u64_f64, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtms_s32_f32, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
    NEONMAP1(vcvtms_u32_f32, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtnd_s64_f64, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
    NEONMAP1(vcvtnd_u64_f64, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtns_s32_f32, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
    NEONMAP1(vcvtns_u32_f32, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtpd_s64_f64, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
    NEONMAP1(vcvtpd_u64_f64, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtps_s32_f32, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
    NEONMAP1(vcvtps_u32_f32, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
    NEONMAP1(vcvts_n_f32_s32, aarch64_neon_vcvtfxs2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvts_n_f32_u32, aarch64_neon_vcvtfxu2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvts_n_s32_f32, aarch64_neon_vcvtfp2fxs,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvts_n_u32_f32, aarch64_neon_vcvtfp2fxu,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvts_s32_f32, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
    NEONMAP1(vcvts_u32_f32, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtxd_f32_f64, aarch64_sisd_fcvtxn, 0),
    NEONMAP1(vmaxnmv_f32, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxnmvq_f32, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxnmvq_f64, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxv_f32, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxv_s32, aarch64_neon_smaxv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxv_u32, aarch64_neon_umaxv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxvq_f32, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxvq_f64, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxvq_s32, aarch64_neon_smaxv, AddRetType | Add1ArgType),
    NEONMAP1(vmaxvq_u32, aarch64_neon_umaxv, AddRetType | Add1ArgType),
    NEONMAP1(vminnmv_f32, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
    NEONMAP1(vminnmvq_f32, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
    NEONMAP1(vminnmvq_f64, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
    NEONMAP1(vminv_f32, aarch64_neon_fminv, AddRetType | Add1ArgType),
    NEONMAP1(vminv_s32, aarch64_neon_sminv, AddRetType | Add1ArgType),
    NEONMAP1(vminv_u32, aarch64_neon_uminv, AddRetType | Add1ArgType),
    NEONMAP1(vminvq_f32, aarch64_neon_fminv, AddRetType | Add1ArgType),
    NEONMAP1(vminvq_f64, aarch64_neon_fminv, AddRetType | Add1ArgType),
    NEONMAP1(vminvq_s32, aarch64_neon_sminv, AddRetType | Add1ArgType),
    NEONMAP1(vminvq_u32, aarch64_neon_uminv, AddRetType | Add1ArgType),
    NEONMAP1(vmull_p64, aarch64_neon_pmull64, 0),
    NEONMAP1(vmulxd_f64, aarch64_neon_fmulx, Add1ArgType),
    NEONMAP1(vmulxs_f32, aarch64_neon_fmulx, Add1ArgType),
    NEONMAP1(vpaddd_s64, aarch64_neon_uaddv, AddRetType | Add1ArgType),
    NEONMAP1(vpaddd_u64, aarch64_neon_uaddv, AddRetType | Add1ArgType),
    NEONMAP1(vpmaxnmqd_f64, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
    NEONMAP1(vpmaxnms_f32, aarch64_neon_fmaxnmv, AddRetType | Add1ArgType),
    NEONMAP1(vpmaxqd_f64, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
    NEONMAP1(vpmaxs_f32, aarch64_neon_fmaxv, AddRetType | Add1ArgType),
    NEONMAP1(vpminnmqd_f64, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
    NEONMAP1(vpminnms_f32, aarch64_neon_fminnmv, AddRetType | Add1ArgType),
    NEONMAP1(vpminqd_f64, aarch64_neon_fminv, AddRetType | Add1ArgType),
    NEONMAP1(vpmins_f32, aarch64_neon_fminv, AddRetType | Add1ArgType),
    NEONMAP1(vqabsb_s8, aarch64_neon_sqabs,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqabsd_s64, aarch64_neon_sqabs, Add1ArgType),
    NEONMAP1(vqabsh_s16, aarch64_neon_sqabs,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqabss_s32, aarch64_neon_sqabs, Add1ArgType),
    NEONMAP1(vqaddb_s8, aarch64_neon_sqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqaddb_u8, aarch64_neon_uqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqaddd_s64, aarch64_neon_sqadd, Add1ArgType),
    NEONMAP1(vqaddd_u64, aarch64_neon_uqadd, Add1ArgType),
    NEONMAP1(vqaddh_s16, aarch64_neon_sqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqaddh_u16, aarch64_neon_uqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqadds_s32, aarch64_neon_sqadd, Add1ArgType),
    NEONMAP1(vqadds_u32, aarch64_neon_uqadd, Add1ArgType),
    NEONMAP1(vqdmulhh_s16, aarch64_neon_sqdmulh,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqdmulhs_s32, aarch64_neon_sqdmulh, Add1ArgType),
    NEONMAP1(vqdmullh_s16, aarch64_neon_sqdmull, VectorRet | Use128BitVectors),
    NEONMAP1(vqdmulls_s32, aarch64_neon_sqdmulls_scalar, 0),
    NEONMAP1(vqmovnd_s64, aarch64_neon_scalar_sqxtn, AddRetType | Add1ArgType),
    NEONMAP1(vqmovnd_u64, aarch64_neon_scalar_uqxtn, AddRetType | Add1ArgType),
    NEONMAP1(vqmovnh_s16, aarch64_neon_sqxtn, VectorRet | Use64BitVectors),
    NEONMAP1(vqmovnh_u16, aarch64_neon_uqxtn, VectorRet | Use64BitVectors),
    NEONMAP1(vqmovns_s32, aarch64_neon_sqxtn, VectorRet | Use64BitVectors),
    NEONMAP1(vqmovns_u32, aarch64_neon_uqxtn, VectorRet | Use64BitVectors),
    NEONMAP1(vqmovund_s64, aarch64_neon_scalar_sqxtun,
             AddRetType | Add1ArgType),
    NEONMAP1(vqmovunh_s16, aarch64_neon_sqxtun, VectorRet | Use64BitVectors),
    NEONMAP1(vqmovuns_s32, aarch64_neon_sqxtun, VectorRet | Use64BitVectors),
    NEONMAP1(vqnegb_s8, aarch64_neon_sqneg,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqnegd_s64, aarch64_neon_sqneg, Add1ArgType),
    NEONMAP1(vqnegh_s16, aarch64_neon_sqneg,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqnegs_s32, aarch64_neon_sqneg, Add1ArgType),
    NEONMAP1(vqrdmlahh_s16, aarch64_neon_sqrdmlah,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqrdmlahs_s32, aarch64_neon_sqrdmlah, Add1ArgType),
    NEONMAP1(vqrdmlshh_s16, aarch64_neon_sqrdmlsh,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqrdmlshs_s32, aarch64_neon_sqrdmlsh, Add1ArgType),
    NEONMAP1(vqrdmulhh_s16, aarch64_neon_sqrdmulh,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqrdmulhs_s32, aarch64_neon_sqrdmulh, Add1ArgType),
    NEONMAP1(vqrshlb_s8, aarch64_neon_sqrshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqrshlb_u8, aarch64_neon_uqrshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqrshld_s64, aarch64_neon_sqrshl, Add1ArgType),
    NEONMAP1(vqrshld_u64, aarch64_neon_uqrshl, Add1ArgType),
    NEONMAP1(vqrshlh_s16, aarch64_neon_sqrshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqrshlh_u16, aarch64_neon_uqrshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqrshls_s32, aarch64_neon_sqrshl, Add1ArgType),
    NEONMAP1(vqrshls_u32, aarch64_neon_uqrshl, Add1ArgType),
    NEONMAP1(vqrshrnd_n_s64, aarch64_neon_sqrshrn, AddRetType),
    NEONMAP1(vqrshrnd_n_u64, aarch64_neon_uqrshrn, AddRetType),
    NEONMAP1(vqrshrnh_n_s16, aarch64_neon_sqrshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqrshrnh_n_u16, aarch64_neon_uqrshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqrshrns_n_s32, aarch64_neon_sqrshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqrshrns_n_u32, aarch64_neon_uqrshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqrshrund_n_s64, aarch64_neon_sqrshrun, AddRetType),
    NEONMAP1(vqrshrunh_n_s16, aarch64_neon_sqrshrun,
             VectorRet | Use64BitVectors),
    NEONMAP1(vqrshruns_n_s32, aarch64_neon_sqrshrun,
             VectorRet | Use64BitVectors),
    NEONMAP1(vqshlb_n_s8, aarch64_neon_sqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshlb_n_u8, aarch64_neon_uqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshlb_s8, aarch64_neon_sqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshlb_u8, aarch64_neon_uqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshld_s64, aarch64_neon_sqshl, Add1ArgType),
    NEONMAP1(vqshld_u64, aarch64_neon_uqshl, Add1ArgType),
    NEONMAP1(vqshlh_n_s16, aarch64_neon_sqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshlh_n_u16, aarch64_neon_uqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshlh_s16, aarch64_neon_sqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshlh_u16, aarch64_neon_uqshl,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshls_n_s32, aarch64_neon_sqshl, Add1ArgType),
    NEONMAP1(vqshls_n_u32, aarch64_neon_uqshl, Add1ArgType),
    NEONMAP1(vqshls_s32, aarch64_neon_sqshl, Add1ArgType),
    NEONMAP1(vqshls_u32, aarch64_neon_uqshl, Add1ArgType),
    NEONMAP1(vqshlub_n_s8, aarch64_neon_sqshlu,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshluh_n_s16, aarch64_neon_sqshlu,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqshlus_n_s32, aarch64_neon_sqshlu, Add1ArgType),
    NEONMAP1(vqshrnd_n_s64, aarch64_neon_sqshrn, AddRetType),
    NEONMAP1(vqshrnd_n_u64, aarch64_neon_uqshrn, AddRetType),
    NEONMAP1(vqshrnh_n_s16, aarch64_neon_sqshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqshrnh_n_u16, aarch64_neon_uqshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqshrns_n_s32, aarch64_neon_sqshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqshrns_n_u32, aarch64_neon_uqshrn, VectorRet | Use64BitVectors),
    NEONMAP1(vqshrund_n_s64, aarch64_neon_sqshrun, AddRetType),
    NEONMAP1(vqshrunh_n_s16, aarch64_neon_sqshrun, VectorRet | Use64BitVectors),
    NEONMAP1(vqshruns_n_s32, aarch64_neon_sqshrun, VectorRet | Use64BitVectors),
    NEONMAP1(vqsubb_s8, aarch64_neon_sqsub,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqsubb_u8, aarch64_neon_uqsub,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqsubd_s64, aarch64_neon_sqsub, Add1ArgType),
    NEONMAP1(vqsubd_u64, aarch64_neon_uqsub, Add1ArgType),
    NEONMAP1(vqsubh_s16, aarch64_neon_sqsub,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqsubh_u16, aarch64_neon_uqsub,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vqsubs_s32, aarch64_neon_sqsub, Add1ArgType),
    NEONMAP1(vqsubs_u32, aarch64_neon_uqsub, Add1ArgType),
    NEONMAP1(vrecped_f64, aarch64_neon_frecpe, Add1ArgType),
    NEONMAP1(vrecpes_f32, aarch64_neon_frecpe, Add1ArgType),
    NEONMAP1(vrecpxd_f64, aarch64_neon_frecpx, Add1ArgType),
    NEONMAP1(vrecpxs_f32, aarch64_neon_frecpx, Add1ArgType),
    NEONMAP1(vrshld_s64, aarch64_neon_srshl, Add1ArgType),
    NEONMAP1(vrshld_u64, aarch64_neon_urshl, Add1ArgType),
    NEONMAP1(vrsqrted_f64, aarch64_neon_frsqrte, Add1ArgType),
    NEONMAP1(vrsqrtes_f32, aarch64_neon_frsqrte, Add1ArgType),
    NEONMAP1(vrsqrtsd_f64, aarch64_neon_frsqrts, Add1ArgType),
    NEONMAP1(vrsqrtss_f32, aarch64_neon_frsqrts, Add1ArgType),
    NEONMAP1(vsha1cq_u32, aarch64_crypto_sha1c, 0),
    NEONMAP1(vsha1h_u32, aarch64_crypto_sha1h, 0),
    NEONMAP1(vsha1mq_u32, aarch64_crypto_sha1m, 0),
    NEONMAP1(vsha1pq_u32, aarch64_crypto_sha1p, 0),
    NEONMAP1(vshld_s64, aarch64_neon_sshl, Add1ArgType),
    NEONMAP1(vshld_u64, aarch64_neon_ushl, Add1ArgType),
    NEONMAP1(vslid_n_s64, aarch64_neon_vsli, Vectorize1ArgType),
    NEONMAP1(vslid_n_u64, aarch64_neon_vsli, Vectorize1ArgType),
    NEONMAP1(vsqaddb_u8, aarch64_neon_usqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vsqaddd_u64, aarch64_neon_usqadd, Add1ArgType),
    NEONMAP1(vsqaddh_u16, aarch64_neon_usqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vsqadds_u32, aarch64_neon_usqadd, Add1ArgType),
    NEONMAP1(vsrid_n_s64, aarch64_neon_vsri, Vectorize1ArgType),
    NEONMAP1(vsrid_n_u64, aarch64_neon_vsri, Vectorize1ArgType),
    NEONMAP1(vuqaddb_s8, aarch64_neon_suqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vuqaddd_s64, aarch64_neon_suqadd, Add1ArgType),
    NEONMAP1(vuqaddh_s16, aarch64_neon_suqadd,
             Vectorize1ArgType | Use64BitVectors),
    NEONMAP1(vuqadds_s32, aarch64_neon_suqadd, Add1ArgType),
    // FP16 scalar intrinisics go here.
    NEONMAP1(vabdh_f16, aarch64_sisd_fabd, Add1ArgType),
    NEONMAP1(vcvtah_s32_f16, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
    NEONMAP1(vcvtah_s64_f16, aarch64_neon_fcvtas, AddRetType | Add1ArgType),
    NEONMAP1(vcvtah_u32_f16, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
    NEONMAP1(vcvtah_u64_f16, aarch64_neon_fcvtau, AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_f16_s32, aarch64_neon_vcvtfxs2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_f16_s64, aarch64_neon_vcvtfxs2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_f16_u32, aarch64_neon_vcvtfxu2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_f16_u64, aarch64_neon_vcvtfxu2fp,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_s32_f16, aarch64_neon_vcvtfp2fxs,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_s64_f16, aarch64_neon_vcvtfp2fxs,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_u32_f16, aarch64_neon_vcvtfp2fxu,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_n_u64_f16, aarch64_neon_vcvtfp2fxu,
             AddRetType | Add1ArgType),
    NEONMAP1(vcvth_s32_f16, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
    NEONMAP1(vcvth_s64_f16, aarch64_neon_fcvtzs, AddRetType | Add1ArgType),
    NEONMAP1(vcvth_u32_f16, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
    NEONMAP1(vcvth_u64_f16, aarch64_neon_fcvtzu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtmh_s32_f16, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
    NEONMAP1(vcvtmh_s64_f16, aarch64_neon_fcvtms, AddRetType | Add1ArgType),
    NEONMAP1(vcvtmh_u32_f16, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtmh_u64_f16, aarch64_neon_fcvtmu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtnh_s32_f16, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
    NEONMAP1(vcvtnh_s64_f16, aarch64_neon_fcvtns, AddRetType | Add1ArgType),
    NEONMAP1(vcvtnh_u32_f16, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtnh_u64_f16, aarch64_neon_fcvtnu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtph_s32_f16, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
    NEONMAP1(vcvtph_s64_f16, aarch64_neon_fcvtps, AddRetType | Add1ArgType),
    NEONMAP1(vcvtph_u32_f16, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
    NEONMAP1(vcvtph_u64_f16, aarch64_neon_fcvtpu, AddRetType | Add1ArgType),
    NEONMAP1(vmulxh_f16, aarch64_neon_fmulx, Add1ArgType),
    NEONMAP1(vrecpeh_f16, aarch64_neon_frecpe, Add1ArgType),
    NEONMAP1(vrecpxh_f16, aarch64_neon_frecpx, Add1ArgType),
    NEONMAP1(vrsqrteh_f16, aarch64_neon_frsqrte, Add1ArgType),
    NEONMAP1(vrsqrtsh_f16, aarch64_neon_frsqrts, Add1ArgType),
};

/// Get name of intrinsics in the AArch64SIMDIntrinsicMap defined above.
static std::string getAArch64SIMDIntrinsicString(unsigned int intrinsicID) {
  switch (intrinsicID) {
  default:
    return std::string("Unexpected intrinsic id " +
                       std::to_string(intrinsicID));
  case NEON::BI__builtin_neon_splat_lane_v:
    return "NEON::BI__builtin_neon_splat_lane_v";
  case NEON::BI__builtin_neon_splat_laneq_v:
    return "NEON::BI__builtin_neon_splat_laneq_v";
  case NEON::BI__builtin_neon_splatq_lane_v:
    return "NEON::BI__builtin_neon_splatq_lane_v";
  case NEON::BI__builtin_neon_splatq_laneq_v:
    return "NEON::BI__builtin_neon_splatq_laneq_v";
  case NEON::BI__builtin_neon_vabs_v:
    return "NEON::BI__builtin_neon_vabs_v";
  case NEON::BI__builtin_neon_vabsq_v:
    return "NEON::BI__builtin_neon_vabsq_v";
  case NEON::BI__builtin_neon_vadd_v:
    return "NEON::BI__builtin_neon_vadd_v";
  case NEON::BI__builtin_neon_vaddhn_v:
    return "NEON::BI__builtin_neon_vaddhn_v";
  case NEON::BI__builtin_neon_vaddq_p128:
    return "NEON::BI__builtin_neon_vaddq_p128";
  case NEON::BI__builtin_neon_vaddq_v:
    return "NEON::BI__builtin_neon_vaddq_v";
  case NEON::BI__builtin_neon_vaesdq_u8:
    return "NEON::BI__builtin_neon_vaesdq_u8";
  case NEON::BI__builtin_neon_vaeseq_u8:
    return "NEON::BI__builtin_neon_vaeseq_u8";
  case NEON::BI__builtin_neon_vaesimcq_u8:
    return "NEON::BI__builtin_neon_vaesimcq_u8";
  case NEON::BI__builtin_neon_vaesmcq_u8:
    return "NEON::BI__builtin_neon_vaesmcq_u8";
  case NEON::BI__builtin_neon_vbcaxq_s16:
    return "NEON::BI__builtin_neon_vbcaxq_s16";
  case NEON::BI__builtin_neon_vbcaxq_s32:
    return "NEON::BI__builtin_neon_vbcaxq_s32";
  case NEON::BI__builtin_neon_vbcaxq_s64:
    return "NEON::BI__builtin_neon_vbcaxq_s64";
  case NEON::BI__builtin_neon_vbcaxq_s8:
    return "NEON::BI__builtin_neon_vbcaxq_s8";
  case NEON::BI__builtin_neon_vbcaxq_u16:
    return "NEON::BI__builtin_neon_vbcaxq_u16";
  case NEON::BI__builtin_neon_vbcaxq_u32:
    return "NEON::BI__builtin_neon_vbcaxq_u32";
  case NEON::BI__builtin_neon_vbcaxq_u64:
    return "NEON::BI__builtin_neon_vbcaxq_u64";
  case NEON::BI__builtin_neon_vbcaxq_u8:
    return "NEON::BI__builtin_neon_vbcaxq_u8";
  case NEON::BI__builtin_neon_vbfdot_f32:
    return "NEON::BI__builtin_neon_vbfdot_f32";
  case NEON::BI__builtin_neon_vbfdotq_f32:
    return "NEON::BI__builtin_neon_vbfdotq_f32";
  case NEON::BI__builtin_neon_vbfmlalbq_f32:
    return "NEON::BI__builtin_neon_vbfmlalbq_f32";
  case NEON::BI__builtin_neon_vbfmlaltq_f32:
    return "NEON::BI__builtin_neon_vbfmlaltq_f32";
  case NEON::BI__builtin_neon_vbfmmlaq_f32:
    return "NEON::BI__builtin_neon_vbfmmlaq_f32";
  case NEON::BI__builtin_neon_vcadd_rot270_f16:
    return "NEON::BI__builtin_neon_vcadd_rot270_f16";
  case NEON::BI__builtin_neon_vcadd_rot270_f32:
    return "NEON::BI__builtin_neon_vcadd_rot270_f32";
  case NEON::BI__builtin_neon_vcadd_rot90_f16:
    return "NEON::BI__builtin_neon_vcadd_rot90_f16";
  case NEON::BI__builtin_neon_vcadd_rot90_f32:
    return "NEON::BI__builtin_neon_vcadd_rot90_f32";
  case NEON::BI__builtin_neon_vcaddq_rot270_f16:
    return "NEON::BI__builtin_neon_vcaddq_rot270_f16";
  case NEON::BI__builtin_neon_vcaddq_rot270_f32:
    return "NEON::BI__builtin_neon_vcaddq_rot270_f32";
  case NEON::BI__builtin_neon_vcaddq_rot270_f64:
    return "NEON::BI__builtin_neon_vcaddq_rot270_f64";
  case NEON::BI__builtin_neon_vcaddq_rot90_f16:
    return "NEON::BI__builtin_neon_vcaddq_rot90_f16";
  case NEON::BI__builtin_neon_vcaddq_rot90_f32:
    return "NEON::BI__builtin_neon_vcaddq_rot90_f32";
  case NEON::BI__builtin_neon_vcaddq_rot90_f64:
    return "NEON::BI__builtin_neon_vcaddq_rot90_f64";
  case NEON::BI__builtin_neon_vcage_v:
    return "NEON::BI__builtin_neon_vcage_v";
  case NEON::BI__builtin_neon_vcageq_v:
    return "NEON::BI__builtin_neon_vcageq_v";
  case NEON::BI__builtin_neon_vcagt_v:
    return "NEON::BI__builtin_neon_vcagt_v";
  case NEON::BI__builtin_neon_vcagtq_v:
    return "NEON::BI__builtin_neon_vcagtq_v";
  case NEON::BI__builtin_neon_vcale_v:
    return "NEON::BI__builtin_neon_vcale_v";
  case NEON::BI__builtin_neon_vcaleq_v:
    return "NEON::BI__builtin_neon_vcaleq_v";
  case NEON::BI__builtin_neon_vcalt_v:
    return "NEON::BI__builtin_neon_vcalt_v";
  case NEON::BI__builtin_neon_vcaltq_v:
    return "NEON::BI__builtin_neon_vcaltq_v";
  case NEON::BI__builtin_neon_vceqz_v:
    return "NEON::BI__builtin_neon_vceqz_v";
  case NEON::BI__builtin_neon_vceqzq_v:
    return "NEON::BI__builtin_neon_vceqzq_v";
  case NEON::BI__builtin_neon_vcgez_v:
    return "NEON::BI__builtin_neon_vcgez_v";
  case NEON::BI__builtin_neon_vcgezq_v:
    return "NEON::BI__builtin_neon_vcgezq_v";
  case NEON::BI__builtin_neon_vcgtz_v:
    return "NEON::BI__builtin_neon_vcgtz_v";
  case NEON::BI__builtin_neon_vcgtzq_v:
    return "NEON::BI__builtin_neon_vcgtzq_v";
  case NEON::BI__builtin_neon_vclez_v:
    return "NEON::BI__builtin_neon_vclez_v";
  case NEON::BI__builtin_neon_vclezq_v:
    return "NEON::BI__builtin_neon_vclezq_v";
  case NEON::BI__builtin_neon_vcls_v:
    return "NEON::BI__builtin_neon_vcls_v";
  case NEON::BI__builtin_neon_vclsq_v:
    return "NEON::BI__builtin_neon_vclsq_v";
  case NEON::BI__builtin_neon_vcltz_v:
    return "NEON::BI__builtin_neon_vcltz_v";
  case NEON::BI__builtin_neon_vcltzq_v:
    return "NEON::BI__builtin_neon_vcltzq_v";
  case NEON::BI__builtin_neon_vclz_v:
    return "NEON::BI__builtin_neon_vclz_v";
  case NEON::BI__builtin_neon_vclzq_v:
    return "NEON::BI__builtin_neon_vclzq_v";
  case NEON::BI__builtin_neon_vcmla_f16:
    return "NEON::BI__builtin_neon_vcmla_f16";
  case NEON::BI__builtin_neon_vcmla_f32:
    return "NEON::BI__builtin_neon_vcmla_f32";
  case NEON::BI__builtin_neon_vcmla_rot180_f16:
    return "NEON::BI__builtin_neon_vcmla_rot180_f16";
  case NEON::BI__builtin_neon_vcmla_rot180_f32:
    return "NEON::BI__builtin_neon_vcmla_rot180_f32";
  case NEON::BI__builtin_neon_vcmla_rot270_f16:
    return "NEON::BI__builtin_neon_vcmla_rot270_f16";
  case NEON::BI__builtin_neon_vcmla_rot270_f32:
    return "NEON::BI__builtin_neon_vcmla_rot270_f32";
  case NEON::BI__builtin_neon_vcmla_rot90_f16:
    return "NEON::BI__builtin_neon_vcmla_rot90_f16";
  case NEON::BI__builtin_neon_vcmla_rot90_f32:
    return "NEON::BI__builtin_neon_vcmla_rot90_f32";
  case NEON::BI__builtin_neon_vcmlaq_f16:
    return "NEON::BI__builtin_neon_vcmlaq_f16";
  case NEON::BI__builtin_neon_vcmlaq_f32:
    return "NEON::BI__builtin_neon_vcmlaq_f32";
  case NEON::BI__builtin_neon_vcmlaq_f64:
    return "NEON::BI__builtin_neon_vcmlaq_f64";
  case NEON::BI__builtin_neon_vcmlaq_rot180_f16:
    return "NEON::BI__builtin_neon_vcmlaq_rot180_f16";
  case NEON::BI__builtin_neon_vcmlaq_rot180_f32:
    return "NEON::BI__builtin_neon_vcmlaq_rot180_f32";
  case NEON::BI__builtin_neon_vcmlaq_rot180_f64:
    return "NEON::BI__builtin_neon_vcmlaq_rot180_f64";
  case NEON::BI__builtin_neon_vcmlaq_rot270_f16:
    return "NEON::BI__builtin_neon_vcmlaq_rot270_f16";
  case NEON::BI__builtin_neon_vcmlaq_rot270_f32:
    return "NEON::BI__builtin_neon_vcmlaq_rot270_f32";
  case NEON::BI__builtin_neon_vcmlaq_rot270_f64:
    return "NEON::BI__builtin_neon_vcmlaq_rot270_f64";
  case NEON::BI__builtin_neon_vcmlaq_rot90_f16:
    return "NEON::BI__builtin_neon_vcmlaq_rot90_f16";
  case NEON::BI__builtin_neon_vcmlaq_rot90_f32:
    return "NEON::BI__builtin_neon_vcmlaq_rot90_f32";
  case NEON::BI__builtin_neon_vcmlaq_rot90_f64:
    return "NEON::BI__builtin_neon_vcmlaq_rot90_f64";
  case NEON::BI__builtin_neon_vcnt_v:
    return "NEON::BI__builtin_neon_vcnt_v";
  case NEON::BI__builtin_neon_vcntq_v:
    return "NEON::BI__builtin_neon_vcntq_v";
  case NEON::BI__builtin_neon_vcvt_f16_f32:
    return "NEON::BI__builtin_neon_vcvt_f16_f32";
  case NEON::BI__builtin_neon_vcvt_f16_s16:
    return "NEON::BI__builtin_neon_vcvt_f16_s16";
  case NEON::BI__builtin_neon_vcvt_f16_u16:
    return "NEON::BI__builtin_neon_vcvt_f16_u16";
  case NEON::BI__builtin_neon_vcvt_f32_f16:
    return "NEON::BI__builtin_neon_vcvt_f32_f16";
  case NEON::BI__builtin_neon_vcvt_f32_v:
    return "NEON::BI__builtin_neon_vcvt_f32_v";
  case NEON::BI__builtin_neon_vcvt_n_f16_s16:
    return "NEON::BI__builtin_neon_vcvt_n_f16_s16";
  case NEON::BI__builtin_neon_vcvt_n_f16_u16:
    return "NEON::BI__builtin_neon_vcvt_n_f16_u16";
  case NEON::BI__builtin_neon_vcvt_n_f32_v:
    return "NEON::BI__builtin_neon_vcvt_n_f32_v";
  case NEON::BI__builtin_neon_vcvt_n_f64_v:
    return "NEON::BI__builtin_neon_vcvt_n_f64_v";
  case NEON::BI__builtin_neon_vcvt_n_s16_f16:
    return "NEON::BI__builtin_neon_vcvt_n_s16_f16";
  case NEON::BI__builtin_neon_vcvt_n_s32_v:
    return "NEON::BI__builtin_neon_vcvt_n_s32_v";
  case NEON::BI__builtin_neon_vcvt_n_s64_v:
    return "NEON::BI__builtin_neon_vcvt_n_s64_v";
  case NEON::BI__builtin_neon_vcvt_n_u16_f16:
    return "NEON::BI__builtin_neon_vcvt_n_u16_f16";
  case NEON::BI__builtin_neon_vcvt_n_u32_v:
    return "NEON::BI__builtin_neon_vcvt_n_u32_v";
  case NEON::BI__builtin_neon_vcvt_n_u64_v:
    return "NEON::BI__builtin_neon_vcvt_n_u64_v";
  case NEON::BI__builtin_neon_vcvtq_f16_s16:
    return "NEON::BI__builtin_neon_vcvtq_f16_s16";
  case NEON::BI__builtin_neon_vcvtq_f16_u16:
    return "NEON::BI__builtin_neon_vcvtq_f16_u16";
  case NEON::BI__builtin_neon_vcvtq_f32_v:
    return "NEON::BI__builtin_neon_vcvtq_f32_v";
  case NEON::BI__builtin_neon_vcvtq_high_bf16_f32:
    return "NEON::BI__builtin_neon_vcvtq_high_bf16_f32";
  case NEON::BI__builtin_neon_vcvtq_n_f16_s16:
    return "NEON::BI__builtin_neon_vcvtq_n_f16_s16";
  case NEON::BI__builtin_neon_vcvtq_n_f16_u16:
    return "NEON::BI__builtin_neon_vcvtq_n_f16_u16";
  case NEON::BI__builtin_neon_vcvtq_n_f32_v:
    return "NEON::BI__builtin_neon_vcvtq_n_f32_v";
  case NEON::BI__builtin_neon_vcvtq_n_f64_v:
    return "NEON::BI__builtin_neon_vcvtq_n_f64_v";
  case NEON::BI__builtin_neon_vcvtq_n_s16_f16:
    return "NEON::BI__builtin_neon_vcvtq_n_s16_f16";
  case NEON::BI__builtin_neon_vcvtq_n_s32_v:
    return "NEON::BI__builtin_neon_vcvtq_n_s32_v";
  case NEON::BI__builtin_neon_vcvtq_n_s64_v:
    return "NEON::BI__builtin_neon_vcvtq_n_s64_v";
  case NEON::BI__builtin_neon_vcvtq_n_u16_f16:
    return "NEON::BI__builtin_neon_vcvtq_n_u16_f16";
  case NEON::BI__builtin_neon_vcvtq_n_u32_v:
    return "NEON::BI__builtin_neon_vcvtq_n_u32_v";
  case NEON::BI__builtin_neon_vcvtq_n_u64_v:
    return "NEON::BI__builtin_neon_vcvtq_n_u64_v";
  case NEON::BI__builtin_neon_vcvtx_f32_v:
    return "NEON::BI__builtin_neon_vcvtx_f32_v";
  case NEON::BI__builtin_neon_vdot_s32:
    return "NEON::BI__builtin_neon_vdot_s32";
  case NEON::BI__builtin_neon_vdot_u32:
    return "NEON::BI__builtin_neon_vdot_u32";
  case NEON::BI__builtin_neon_vdotq_s32:
    return "NEON::BI__builtin_neon_vdotq_s32";
  case NEON::BI__builtin_neon_vdotq_u32:
    return "NEON::BI__builtin_neon_vdotq_u32";
  case NEON::BI__builtin_neon_veor3q_s16:
    return "NEON::BI__builtin_neon_veor3q_s16";
  case NEON::BI__builtin_neon_veor3q_s32:
    return "NEON::BI__builtin_neon_veor3q_s32";
  case NEON::BI__builtin_neon_veor3q_s64:
    return "NEON::BI__builtin_neon_veor3q_s64";
  case NEON::BI__builtin_neon_veor3q_s8:
    return "NEON::BI__builtin_neon_veor3q_s8";
  case NEON::BI__builtin_neon_veor3q_u16:
    return "NEON::BI__builtin_neon_veor3q_u16";
  case NEON::BI__builtin_neon_veor3q_u32:
    return "NEON::BI__builtin_neon_veor3q_u32";
  case NEON::BI__builtin_neon_veor3q_u64:
    return "NEON::BI__builtin_neon_veor3q_u64";
  case NEON::BI__builtin_neon_veor3q_u8:
    return "NEON::BI__builtin_neon_veor3q_u8";
  case NEON::BI__builtin_neon_vext_v:
    return "NEON::BI__builtin_neon_vext_v";
  case NEON::BI__builtin_neon_vextq_v:
    return "NEON::BI__builtin_neon_vextq_v";
  case NEON::BI__builtin_neon_vfma_v:
    return "NEON::BI__builtin_neon_vfma_v";
  case NEON::BI__builtin_neon_vfmaq_v:
    return "NEON::BI__builtin_neon_vfmaq_v";
  case NEON::BI__builtin_neon_vfmlal_high_f16:
    return "NEON::BI__builtin_neon_vfmlal_high_f16";
  case NEON::BI__builtin_neon_vfmlal_low_f16:
    return "NEON::BI__builtin_neon_vfmlal_low_f16";
  case NEON::BI__builtin_neon_vfmlalq_high_f16:
    return "NEON::BI__builtin_neon_vfmlalq_high_f16";
  case NEON::BI__builtin_neon_vfmlalq_low_f16:
    return "NEON::BI__builtin_neon_vfmlalq_low_f16";
  case NEON::BI__builtin_neon_vfmlsl_high_f16:
    return "NEON::BI__builtin_neon_vfmlsl_high_f16";
  case NEON::BI__builtin_neon_vfmlsl_low_f16:
    return "NEON::BI__builtin_neon_vfmlsl_low_f16";
  case NEON::BI__builtin_neon_vfmlslq_high_f16:
    return "NEON::BI__builtin_neon_vfmlslq_high_f16";
  case NEON::BI__builtin_neon_vfmlslq_low_f16:
    return "NEON::BI__builtin_neon_vfmlslq_low_f16";
  case NEON::BI__builtin_neon_vhadd_v:
    return "NEON::BI__builtin_neon_vhadd_v";
  case NEON::BI__builtin_neon_vhaddq_v:
    return "NEON::BI__builtin_neon_vhaddq_v";
  case NEON::BI__builtin_neon_vhsub_v:
    return "NEON::BI__builtin_neon_vhsub_v";
  case NEON::BI__builtin_neon_vhsubq_v:
    return "NEON::BI__builtin_neon_vhsubq_v";
  case NEON::BI__builtin_neon_vld1_x2_v:
    return "NEON::BI__builtin_neon_vld1_x2_v";
  case NEON::BI__builtin_neon_vld1_x3_v:
    return "NEON::BI__builtin_neon_vld1_x3_v";
  case NEON::BI__builtin_neon_vld1_x4_v:
    return "NEON::BI__builtin_neon_vld1_x4_v";
  case NEON::BI__builtin_neon_vld1q_x2_v:
    return "NEON::BI__builtin_neon_vld1q_x2_v";
  case NEON::BI__builtin_neon_vld1q_x3_v:
    return "NEON::BI__builtin_neon_vld1q_x3_v";
  case NEON::BI__builtin_neon_vld1q_x4_v:
    return "NEON::BI__builtin_neon_vld1q_x4_v";
  case NEON::BI__builtin_neon_vmmlaq_s32:
    return "NEON::BI__builtin_neon_vmmlaq_s32";
  case NEON::BI__builtin_neon_vmmlaq_u32:
    return "NEON::BI__builtin_neon_vmmlaq_u32";
  case NEON::BI__builtin_neon_vmovl_v:
    return "NEON::BI__builtin_neon_vmovl_v";
  case NEON::BI__builtin_neon_vmovn_v:
    return "NEON::BI__builtin_neon_vmovn_v";
  case NEON::BI__builtin_neon_vmul_v:
    return "NEON::BI__builtin_neon_vmul_v";
  case NEON::BI__builtin_neon_vmulq_v:
    return "NEON::BI__builtin_neon_vmulq_v";
  case NEON::BI__builtin_neon_vpadd_v:
    return "NEON::BI__builtin_neon_vpadd_v";
  case NEON::BI__builtin_neon_vpaddl_v:
    return "NEON::BI__builtin_neon_vpaddl_v";
  case NEON::BI__builtin_neon_vpaddlq_v:
    return "NEON::BI__builtin_neon_vpaddlq_v";
  case NEON::BI__builtin_neon_vpaddq_v:
    return "NEON::BI__builtin_neon_vpaddq_v";
  case NEON::BI__builtin_neon_vqabs_v:
    return "NEON::BI__builtin_neon_vqabs_v";
  case NEON::BI__builtin_neon_vqabsq_v:
    return "NEON::BI__builtin_neon_vqabsq_v";
  case NEON::BI__builtin_neon_vqadd_v:
    return "NEON::BI__builtin_neon_vqadd_v";
  case NEON::BI__builtin_neon_vqaddq_v:
    return "NEON::BI__builtin_neon_vqaddq_v";
  case NEON::BI__builtin_neon_vqdmlal_v:
    return "NEON::BI__builtin_neon_vqdmlal_v";
  case NEON::BI__builtin_neon_vqdmlsl_v:
    return "NEON::BI__builtin_neon_vqdmlsl_v";
  case NEON::BI__builtin_neon_vqdmulh_lane_v:
    return "NEON::BI__builtin_neon_vqdmulh_lane_v";
  case NEON::BI__builtin_neon_vqdmulh_laneq_v:
    return "NEON::BI__builtin_neon_vqdmulh_laneq_v";
  case NEON::BI__builtin_neon_vqdmulh_v:
    return "NEON::BI__builtin_neon_vqdmulh_v";
  case NEON::BI__builtin_neon_vqdmulhq_lane_v:
    return "NEON::BI__builtin_neon_vqdmulhq_lane_v";
  case NEON::BI__builtin_neon_vqdmulhq_laneq_v:
    return "NEON::BI__builtin_neon_vqdmulhq_laneq_v";
  case NEON::BI__builtin_neon_vqdmulhq_v:
    return "NEON::BI__builtin_neon_vqdmulhq_v";
  case NEON::BI__builtin_neon_vqdmull_v:
    return "NEON::BI__builtin_neon_vqdmull_v";
  case NEON::BI__builtin_neon_vqmovn_v:
    return "NEON::BI__builtin_neon_vqmovn_v";
  case NEON::BI__builtin_neon_vqmovun_v:
    return "NEON::BI__builtin_neon_vqmovun_v";
  case NEON::BI__builtin_neon_vqneg_v:
    return "NEON::BI__builtin_neon_vqneg_v";
  case NEON::BI__builtin_neon_vqnegq_v:
    return "NEON::BI__builtin_neon_vqnegq_v";
  case NEON::BI__builtin_neon_vqrdmlah_s16:
    return "NEON::BI__builtin_neon_vqrdmlah_s16";
  case NEON::BI__builtin_neon_vqrdmlah_s32:
    return "NEON::BI__builtin_neon_vqrdmlah_s32";
  case NEON::BI__builtin_neon_vqrdmlahq_s16:
    return "NEON::BI__builtin_neon_vqrdmlahq_s16";
  case NEON::BI__builtin_neon_vqrdmlahq_s32:
    return "NEON::BI__builtin_neon_vqrdmlahq_s32";
  case NEON::BI__builtin_neon_vqrdmlsh_s16:
    return "NEON::BI__builtin_neon_vqrdmlsh_s16";
  case NEON::BI__builtin_neon_vqrdmlsh_s32:
    return "NEON::BI__builtin_neon_vqrdmlsh_s32";
  case NEON::BI__builtin_neon_vqrdmlshq_s16:
    return "NEON::BI__builtin_neon_vqrdmlshq_s16";
  case NEON::BI__builtin_neon_vqrdmlshq_s32:
    return "NEON::BI__builtin_neon_vqrdmlshq_s32";
  case NEON::BI__builtin_neon_vqrdmulh_lane_v:
    return "NEON::BI__builtin_neon_vqrdmulh_lane_v";
  case NEON::BI__builtin_neon_vqrdmulh_laneq_v:
    return "NEON::BI__builtin_neon_vqrdmulh_laneq_v";
  case NEON::BI__builtin_neon_vqrdmulh_v:
    return "NEON::BI__builtin_neon_vqrdmulh_v";
  case NEON::BI__builtin_neon_vqrdmulhq_lane_v:
    return "NEON::BI__builtin_neon_vqrdmulhq_lane_v";
  case NEON::BI__builtin_neon_vqrdmulhq_laneq_v:
    return "NEON::BI__builtin_neon_vqrdmulhq_laneq_v";
  case NEON::BI__builtin_neon_vqrdmulhq_v:
    return "NEON::BI__builtin_neon_vqrdmulhq_v";
  case NEON::BI__builtin_neon_vqrshl_v:
    return "NEON::BI__builtin_neon_vqrshl_v";
  case NEON::BI__builtin_neon_vqrshlq_v:
    return "NEON::BI__builtin_neon_vqrshlq_v";
  case NEON::BI__builtin_neon_vqshl_n_v:
    return "NEON::BI__builtin_neon_vqshl_n_v";
  case NEON::BI__builtin_neon_vqshl_v:
    return "NEON::BI__builtin_neon_vqshl_v";
  case NEON::BI__builtin_neon_vqshlq_n_v:
    return "NEON::BI__builtin_neon_vqshlq_n_v";
  case NEON::BI__builtin_neon_vqshlq_v:
    return "NEON::BI__builtin_neon_vqshlq_v";
  case NEON::BI__builtin_neon_vqshlu_n_v:
    return "NEON::BI__builtin_neon_vqshlu_n_v";
  case NEON::BI__builtin_neon_vqshluq_n_v:
    return "NEON::BI__builtin_neon_vqshluq_n_v";
  case NEON::BI__builtin_neon_vqsub_v:
    return "NEON::BI__builtin_neon_vqsub_v";
  case NEON::BI__builtin_neon_vqsubq_v:
    return "NEON::BI__builtin_neon_vqsubq_v";
  case NEON::BI__builtin_neon_vraddhn_v:
    return "NEON::BI__builtin_neon_vraddhn_v";
  case NEON::BI__builtin_neon_vrax1q_u64:
    return "NEON::BI__builtin_neon_vrax1q_u64";
  case NEON::BI__builtin_neon_vrecpe_v:
    return "NEON::BI__builtin_neon_vrecpe_v";
  case NEON::BI__builtin_neon_vrecpeq_v:
    return "NEON::BI__builtin_neon_vrecpeq_v";
  case NEON::BI__builtin_neon_vrecps_v:
    return "NEON::BI__builtin_neon_vrecps_v";
  case NEON::BI__builtin_neon_vrecpsq_v:
    return "NEON::BI__builtin_neon_vrecpsq_v";
  case NEON::BI__builtin_neon_vrhadd_v:
    return "NEON::BI__builtin_neon_vrhadd_v";
  case NEON::BI__builtin_neon_vrhaddq_v:
    return "NEON::BI__builtin_neon_vrhaddq_v";
  case NEON::BI__builtin_neon_vrnd32x_f32:
    return "NEON::BI__builtin_neon_vrnd32x_f32";
  case NEON::BI__builtin_neon_vrnd32x_f64:
    return "NEON::BI__builtin_neon_vrnd32x_f64";
  case NEON::BI__builtin_neon_vrnd32xq_f32:
    return "NEON::BI__builtin_neon_vrnd32xq_f32";
  case NEON::BI__builtin_neon_vrnd32xq_f64:
    return "NEON::BI__builtin_neon_vrnd32xq_f64";
  case NEON::BI__builtin_neon_vrnd32z_f32:
    return "NEON::BI__builtin_neon_vrnd32z_f32";
  case NEON::BI__builtin_neon_vrnd32z_f64:
    return "NEON::BI__builtin_neon_vrnd32z_f64";
  case NEON::BI__builtin_neon_vrnd32zq_f32:
    return "NEON::BI__builtin_neon_vrnd32zq_f32";
  case NEON::BI__builtin_neon_vrnd32zq_f64:
    return "NEON::BI__builtin_neon_vrnd32zq_f64";
  case NEON::BI__builtin_neon_vrnd64x_f32:
    return "NEON::BI__builtin_neon_vrnd64x_f32";
  case NEON::BI__builtin_neon_vrnd64x_f64:
    return "NEON::BI__builtin_neon_vrnd64x_f64";
  case NEON::BI__builtin_neon_vrnd64xq_f32:
    return "NEON::BI__builtin_neon_vrnd64xq_f32";
  case NEON::BI__builtin_neon_vrnd64xq_f64:
    return "NEON::BI__builtin_neon_vrnd64xq_f64";
  case NEON::BI__builtin_neon_vrnd64z_f32:
    return "NEON::BI__builtin_neon_vrnd64z_f32";
  case NEON::BI__builtin_neon_vrnd64z_f64:
    return "NEON::BI__builtin_neon_vrnd64z_f64";
  case NEON::BI__builtin_neon_vrnd64zq_f32:
    return "NEON::BI__builtin_neon_vrnd64zq_f32";
  case NEON::BI__builtin_neon_vrnd64zq_f64:
    return "NEON::BI__builtin_neon_vrnd64zq_f64";
  case NEON::BI__builtin_neon_vrndi_v:
    return "NEON::BI__builtin_neon_vrndi_v";
  case NEON::BI__builtin_neon_vrndiq_v:
    return "NEON::BI__builtin_neon_vrndiq_v";
  case NEON::BI__builtin_neon_vrshl_v:
    return "NEON::BI__builtin_neon_vrshl_v";
  case NEON::BI__builtin_neon_vrshlq_v:
    return "NEON::BI__builtin_neon_vrshlq_v";
  case NEON::BI__builtin_neon_vrshr_n_v:
    return "NEON::BI__builtin_neon_vrshr_n_v";
  case NEON::BI__builtin_neon_vrshrq_n_v:
    return "NEON::BI__builtin_neon_vrshrq_n_v";
  case NEON::BI__builtin_neon_vrsqrte_v:
    return "NEON::BI__builtin_neon_vrsqrte_v";
  case NEON::BI__builtin_neon_vrsqrteq_v:
    return "NEON::BI__builtin_neon_vrsqrteq_v";
  case NEON::BI__builtin_neon_vrsqrts_v:
    return "NEON::BI__builtin_neon_vrsqrts_v";
  case NEON::BI__builtin_neon_vrsqrtsq_v:
    return "NEON::BI__builtin_neon_vrsqrtsq_v";
  case NEON::BI__builtin_neon_vrsubhn_v:
    return "NEON::BI__builtin_neon_vrsubhn_v";
  case NEON::BI__builtin_neon_vsha1su0q_u32:
    return "NEON::BI__builtin_neon_vsha1su0q_u32";
  case NEON::BI__builtin_neon_vsha1su1q_u32:
    return "NEON::BI__builtin_neon_vsha1su1q_u32";
  case NEON::BI__builtin_neon_vsha256h2q_u32:
    return "NEON::BI__builtin_neon_vsha256h2q_u32";
  case NEON::BI__builtin_neon_vsha256hq_u32:
    return "NEON::BI__builtin_neon_vsha256hq_u32";
  case NEON::BI__builtin_neon_vsha256su0q_u32:
    return "NEON::BI__builtin_neon_vsha256su0q_u32";
  case NEON::BI__builtin_neon_vsha256su1q_u32:
    return "NEON::BI__builtin_neon_vsha256su1q_u32";
  case NEON::BI__builtin_neon_vsha512h2q_u64:
    return "NEON::BI__builtin_neon_vsha512h2q_u64";
  case NEON::BI__builtin_neon_vsha512hq_u64:
    return "NEON::BI__builtin_neon_vsha512hq_u64";
  case NEON::BI__builtin_neon_vsha512su0q_u64:
    return "NEON::BI__builtin_neon_vsha512su0q_u64";
  case NEON::BI__builtin_neon_vsha512su1q_u64:
    return "NEON::BI__builtin_neon_vsha512su1q_u64";
  case NEON::BI__builtin_neon_vshl_n_v:
    return "NEON::BI__builtin_neon_vshl_n_v";
  case NEON::BI__builtin_neon_vshl_v:
    return "NEON::BI__builtin_neon_vshl_v";
  case NEON::BI__builtin_neon_vshll_n_v:
    return "NEON::BI__builtin_neon_vshll_n_v";
  case NEON::BI__builtin_neon_vshlq_n_v:
    return "NEON::BI__builtin_neon_vshlq_n_v";
  case NEON::BI__builtin_neon_vshlq_v:
    return "NEON::BI__builtin_neon_vshlq_v";
  case NEON::BI__builtin_neon_vshr_n_v:
    return "NEON::BI__builtin_neon_vshr_n_v";
  case NEON::BI__builtin_neon_vshrn_n_v:
    return "NEON::BI__builtin_neon_vshrn_n_v";
  case NEON::BI__builtin_neon_vshrq_n_v:
    return "NEON::BI__builtin_neon_vshrq_n_v";
  case NEON::BI__builtin_neon_vsm3partw1q_u32:
    return "NEON::BI__builtin_neon_vsm3partw1q_u32";
  case NEON::BI__builtin_neon_vsm3partw2q_u32:
    return "NEON::BI__builtin_neon_vsm3partw2q_u32";
  case NEON::BI__builtin_neon_vsm3ss1q_u32:
    return "NEON::BI__builtin_neon_vsm3ss1q_u32";
  case NEON::BI__builtin_neon_vsm3tt1aq_u32:
    return "NEON::BI__builtin_neon_vsm3tt1aq_u32";
  case NEON::BI__builtin_neon_vsm3tt1bq_u32:
    return "NEON::BI__builtin_neon_vsm3tt1bq_u32";
  case NEON::BI__builtin_neon_vsm3tt2aq_u32:
    return "NEON::BI__builtin_neon_vsm3tt2aq_u32";
  case NEON::BI__builtin_neon_vsm3tt2bq_u32:
    return "NEON::BI__builtin_neon_vsm3tt2bq_u32";
  case NEON::BI__builtin_neon_vsm4ekeyq_u32:
    return "NEON::BI__builtin_neon_vsm4ekeyq_u32";
  case NEON::BI__builtin_neon_vsm4eq_u32:
    return "NEON::BI__builtin_neon_vsm4eq_u32";
  case NEON::BI__builtin_neon_vst1_x2_v:
    return "NEON::BI__builtin_neon_vst1_x2_v";
  case NEON::BI__builtin_neon_vst1_x3_v:
    return "NEON::BI__builtin_neon_vst1_x3_v";
  case NEON::BI__builtin_neon_vst1_x4_v:
    return "NEON::BI__builtin_neon_vst1_x4_v";
  case NEON::BI__builtin_neon_vst1q_x2_v:
    return "NEON::BI__builtin_neon_vst1q_x2_v";
  case NEON::BI__builtin_neon_vst1q_x3_v:
    return "NEON::BI__builtin_neon_vst1q_x3_v";
  case NEON::BI__builtin_neon_vst1q_x4_v:
    return "NEON::BI__builtin_neon_vst1q_x4_v";
  case NEON::BI__builtin_neon_vsubhn_v:
    return "NEON::BI__builtin_neon_vsubhn_v";
  case NEON::BI__builtin_neon_vtst_v:
    return "NEON::BI__builtin_neon_vtst_v";
  case NEON::BI__builtin_neon_vtstq_v:
    return "NEON::BI__builtin_neon_vtstq_v";
  case NEON::BI__builtin_neon_vusdot_s32:
    return "NEON::BI__builtin_neon_vusdot_s32";
  case NEON::BI__builtin_neon_vusdotq_s32:
    return "NEON::BI__builtin_neon_vusdotq_s32";
  case NEON::BI__builtin_neon_vusmmlaq_s32:
    return "NEON::BI__builtin_neon_vusmmlaq_s32";
  case NEON::BI__builtin_neon_vxarq_u64:
    return "NEON::BI__builtin_neon_vxarq_u64";
  }
}

// Some intrinsics are equivalent for codegen.
static const std::pair<unsigned, unsigned> NEONEquivalentIntrinsicMap[] = {
    {
        NEON::BI__builtin_neon_splat_lane_bf16,
        NEON::BI__builtin_neon_splat_lane_v,
    },
    {
        NEON::BI__builtin_neon_splat_laneq_bf16,
        NEON::BI__builtin_neon_splat_laneq_v,
    },
    {
        NEON::BI__builtin_neon_splatq_lane_bf16,
        NEON::BI__builtin_neon_splatq_lane_v,
    },
    {
        NEON::BI__builtin_neon_splatq_laneq_bf16,
        NEON::BI__builtin_neon_splatq_laneq_v,
    },
    {
        NEON::BI__builtin_neon_vabd_f16,
        NEON::BI__builtin_neon_vabd_v,
    },
    {
        NEON::BI__builtin_neon_vabdq_f16,
        NEON::BI__builtin_neon_vabdq_v,
    },
    {
        NEON::BI__builtin_neon_vabs_f16,
        NEON::BI__builtin_neon_vabs_v,
    },
    {
        NEON::BI__builtin_neon_vabsq_f16,
        NEON::BI__builtin_neon_vabsq_v,
    },
    {
        NEON::BI__builtin_neon_vcage_f16,
        NEON::BI__builtin_neon_vcage_v,
    },
    {
        NEON::BI__builtin_neon_vcageq_f16,
        NEON::BI__builtin_neon_vcageq_v,
    },
    {
        NEON::BI__builtin_neon_vcagt_f16,
        NEON::BI__builtin_neon_vcagt_v,
    },
    {
        NEON::BI__builtin_neon_vcagtq_f16,
        NEON::BI__builtin_neon_vcagtq_v,
    },
    {
        NEON::BI__builtin_neon_vcale_f16,
        NEON::BI__builtin_neon_vcale_v,
    },
    {
        NEON::BI__builtin_neon_vcaleq_f16,
        NEON::BI__builtin_neon_vcaleq_v,
    },
    {
        NEON::BI__builtin_neon_vcalt_f16,
        NEON::BI__builtin_neon_vcalt_v,
    },
    {
        NEON::BI__builtin_neon_vcaltq_f16,
        NEON::BI__builtin_neon_vcaltq_v,
    },
    {
        NEON::BI__builtin_neon_vceqz_f16,
        NEON::BI__builtin_neon_vceqz_v,
    },
    {
        NEON::BI__builtin_neon_vceqzq_f16,
        NEON::BI__builtin_neon_vceqzq_v,
    },
    {
        NEON::BI__builtin_neon_vcgez_f16,
        NEON::BI__builtin_neon_vcgez_v,
    },
    {
        NEON::BI__builtin_neon_vcgezq_f16,
        NEON::BI__builtin_neon_vcgezq_v,
    },
    {
        NEON::BI__builtin_neon_vcgtz_f16,
        NEON::BI__builtin_neon_vcgtz_v,
    },
    {
        NEON::BI__builtin_neon_vcgtzq_f16,
        NEON::BI__builtin_neon_vcgtzq_v,
    },
    {
        NEON::BI__builtin_neon_vclez_f16,
        NEON::BI__builtin_neon_vclez_v,
    },
    {
        NEON::BI__builtin_neon_vclezq_f16,
        NEON::BI__builtin_neon_vclezq_v,
    },
    {
        NEON::BI__builtin_neon_vcltz_f16,
        NEON::BI__builtin_neon_vcltz_v,
    },
    {
        NEON::BI__builtin_neon_vcltzq_f16,
        NEON::BI__builtin_neon_vcltzq_v,
    },
    {
        NEON::BI__builtin_neon_vfma_f16,
        NEON::BI__builtin_neon_vfma_v,
    },
    {
        NEON::BI__builtin_neon_vfma_lane_f16,
        NEON::BI__builtin_neon_vfma_lane_v,
    },
    {
        NEON::BI__builtin_neon_vfma_laneq_f16,
        NEON::BI__builtin_neon_vfma_laneq_v,
    },
    {
        NEON::BI__builtin_neon_vfmaq_f16,
        NEON::BI__builtin_neon_vfmaq_v,
    },
    {
        NEON::BI__builtin_neon_vfmaq_lane_f16,
        NEON::BI__builtin_neon_vfmaq_lane_v,
    },
    {
        NEON::BI__builtin_neon_vfmaq_laneq_f16,
        NEON::BI__builtin_neon_vfmaq_laneq_v,
    },
    {NEON::BI__builtin_neon_vld1_bf16_x2, NEON::BI__builtin_neon_vld1_x2_v},
    {NEON::BI__builtin_neon_vld1_bf16_x3, NEON::BI__builtin_neon_vld1_x3_v},
    {NEON::BI__builtin_neon_vld1_bf16_x4, NEON::BI__builtin_neon_vld1_x4_v},
    {NEON::BI__builtin_neon_vld1_bf16, NEON::BI__builtin_neon_vld1_v},
    {NEON::BI__builtin_neon_vld1_dup_bf16, NEON::BI__builtin_neon_vld1_dup_v},
    {NEON::BI__builtin_neon_vld1_lane_bf16, NEON::BI__builtin_neon_vld1_lane_v},
    {NEON::BI__builtin_neon_vld1q_bf16_x2, NEON::BI__builtin_neon_vld1q_x2_v},
    {NEON::BI__builtin_neon_vld1q_bf16_x3, NEON::BI__builtin_neon_vld1q_x3_v},
    {NEON::BI__builtin_neon_vld1q_bf16_x4, NEON::BI__builtin_neon_vld1q_x4_v},
    {NEON::BI__builtin_neon_vld1q_bf16, NEON::BI__builtin_neon_vld1q_v},
    {NEON::BI__builtin_neon_vld1q_dup_bf16, NEON::BI__builtin_neon_vld1q_dup_v},
    {NEON::BI__builtin_neon_vld1q_lane_bf16,
     NEON::BI__builtin_neon_vld1q_lane_v},
    {NEON::BI__builtin_neon_vld2_bf16, NEON::BI__builtin_neon_vld2_v},
    {NEON::BI__builtin_neon_vld2_dup_bf16, NEON::BI__builtin_neon_vld2_dup_v},
    {NEON::BI__builtin_neon_vld2_lane_bf16, NEON::BI__builtin_neon_vld2_lane_v},
    {NEON::BI__builtin_neon_vld2q_bf16, NEON::BI__builtin_neon_vld2q_v},
    {NEON::BI__builtin_neon_vld2q_dup_bf16, NEON::BI__builtin_neon_vld2q_dup_v},
    {NEON::BI__builtin_neon_vld2q_lane_bf16,
     NEON::BI__builtin_neon_vld2q_lane_v},
    {NEON::BI__builtin_neon_vld3_bf16, NEON::BI__builtin_neon_vld3_v},
    {NEON::BI__builtin_neon_vld3_dup_bf16, NEON::BI__builtin_neon_vld3_dup_v},
    {NEON::BI__builtin_neon_vld3_lane_bf16, NEON::BI__builtin_neon_vld3_lane_v},
    {NEON::BI__builtin_neon_vld3q_bf16, NEON::BI__builtin_neon_vld3q_v},
    {NEON::BI__builtin_neon_vld3q_dup_bf16, NEON::BI__builtin_neon_vld3q_dup_v},
    {NEON::BI__builtin_neon_vld3q_lane_bf16,
     NEON::BI__builtin_neon_vld3q_lane_v},
    {NEON::BI__builtin_neon_vld4_bf16, NEON::BI__builtin_neon_vld4_v},
    {NEON::BI__builtin_neon_vld4_dup_bf16, NEON::BI__builtin_neon_vld4_dup_v},
    {NEON::BI__builtin_neon_vld4_lane_bf16, NEON::BI__builtin_neon_vld4_lane_v},
    {NEON::BI__builtin_neon_vld4q_bf16, NEON::BI__builtin_neon_vld4q_v},
    {NEON::BI__builtin_neon_vld4q_dup_bf16, NEON::BI__builtin_neon_vld4q_dup_v},
    {NEON::BI__builtin_neon_vld4q_lane_bf16,
     NEON::BI__builtin_neon_vld4q_lane_v},
    {
        NEON::BI__builtin_neon_vmax_f16,
        NEON::BI__builtin_neon_vmax_v,
    },
    {
        NEON::BI__builtin_neon_vmaxnm_f16,
        NEON::BI__builtin_neon_vmaxnm_v,
    },
    {
        NEON::BI__builtin_neon_vmaxnmq_f16,
        NEON::BI__builtin_neon_vmaxnmq_v,
    },
    {
        NEON::BI__builtin_neon_vmaxq_f16,
        NEON::BI__builtin_neon_vmaxq_v,
    },
    {
        NEON::BI__builtin_neon_vmin_f16,
        NEON::BI__builtin_neon_vmin_v,
    },
    {
        NEON::BI__builtin_neon_vminnm_f16,
        NEON::BI__builtin_neon_vminnm_v,
    },
    {
        NEON::BI__builtin_neon_vminnmq_f16,
        NEON::BI__builtin_neon_vminnmq_v,
    },
    {
        NEON::BI__builtin_neon_vminq_f16,
        NEON::BI__builtin_neon_vminq_v,
    },
    {
        NEON::BI__builtin_neon_vmulx_f16,
        NEON::BI__builtin_neon_vmulx_v,
    },
    {
        NEON::BI__builtin_neon_vmulxq_f16,
        NEON::BI__builtin_neon_vmulxq_v,
    },
    {
        NEON::BI__builtin_neon_vpadd_f16,
        NEON::BI__builtin_neon_vpadd_v,
    },
    {
        NEON::BI__builtin_neon_vpaddq_f16,
        NEON::BI__builtin_neon_vpaddq_v,
    },
    {
        NEON::BI__builtin_neon_vpmax_f16,
        NEON::BI__builtin_neon_vpmax_v,
    },
    {
        NEON::BI__builtin_neon_vpmaxnm_f16,
        NEON::BI__builtin_neon_vpmaxnm_v,
    },
    {
        NEON::BI__builtin_neon_vpmaxnmq_f16,
        NEON::BI__builtin_neon_vpmaxnmq_v,
    },
    {
        NEON::BI__builtin_neon_vpmaxq_f16,
        NEON::BI__builtin_neon_vpmaxq_v,
    },
    {
        NEON::BI__builtin_neon_vpmin_f16,
        NEON::BI__builtin_neon_vpmin_v,
    },
    {
        NEON::BI__builtin_neon_vpminnm_f16,
        NEON::BI__builtin_neon_vpminnm_v,
    },
    {
        NEON::BI__builtin_neon_vpminnmq_f16,
        NEON::BI__builtin_neon_vpminnmq_v,
    },
    {
        NEON::BI__builtin_neon_vpminq_f16,
        NEON::BI__builtin_neon_vpminq_v,
    },
    {
        NEON::BI__builtin_neon_vrecpe_f16,
        NEON::BI__builtin_neon_vrecpe_v,
    },
    {
        NEON::BI__builtin_neon_vrecpeq_f16,
        NEON::BI__builtin_neon_vrecpeq_v,
    },
    {
        NEON::BI__builtin_neon_vrecps_f16,
        NEON::BI__builtin_neon_vrecps_v,
    },
    {
        NEON::BI__builtin_neon_vrecpsq_f16,
        NEON::BI__builtin_neon_vrecpsq_v,
    },
    {
        NEON::BI__builtin_neon_vrnd_f16,
        NEON::BI__builtin_neon_vrnd_v,
    },
    {
        NEON::BI__builtin_neon_vrnda_f16,
        NEON::BI__builtin_neon_vrnda_v,
    },
    {
        NEON::BI__builtin_neon_vrndaq_f16,
        NEON::BI__builtin_neon_vrndaq_v,
    },
    {
        NEON::BI__builtin_neon_vrndi_f16,
        NEON::BI__builtin_neon_vrndi_v,
    },
    {
        NEON::BI__builtin_neon_vrndiq_f16,
        NEON::BI__builtin_neon_vrndiq_v,
    },
    {
        NEON::BI__builtin_neon_vrndm_f16,
        NEON::BI__builtin_neon_vrndm_v,
    },
    {
        NEON::BI__builtin_neon_vrndmq_f16,
        NEON::BI__builtin_neon_vrndmq_v,
    },
    {
        NEON::BI__builtin_neon_vrndn_f16,
        NEON::BI__builtin_neon_vrndn_v,
    },
    {
        NEON::BI__builtin_neon_vrndnq_f16,
        NEON::BI__builtin_neon_vrndnq_v,
    },
    {
        NEON::BI__builtin_neon_vrndp_f16,
        NEON::BI__builtin_neon_vrndp_v,
    },
    {
        NEON::BI__builtin_neon_vrndpq_f16,
        NEON::BI__builtin_neon_vrndpq_v,
    },
    {
        NEON::BI__builtin_neon_vrndq_f16,
        NEON::BI__builtin_neon_vrndq_v,
    },
    {
        NEON::BI__builtin_neon_vrndx_f16,
        NEON::BI__builtin_neon_vrndx_v,
    },
    {
        NEON::BI__builtin_neon_vrndxq_f16,
        NEON::BI__builtin_neon_vrndxq_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrte_f16,
        NEON::BI__builtin_neon_vrsqrte_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrteq_f16,
        NEON::BI__builtin_neon_vrsqrteq_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrts_f16,
        NEON::BI__builtin_neon_vrsqrts_v,
    },
    {
        NEON::BI__builtin_neon_vrsqrtsq_f16,
        NEON::BI__builtin_neon_vrsqrtsq_v,
    },
    {
        NEON::BI__builtin_neon_vsqrt_f16,
        NEON::BI__builtin_neon_vsqrt_v,
    },
    {
        NEON::BI__builtin_neon_vsqrtq_f16,
        NEON::BI__builtin_neon_vsqrtq_v,
    },
    {NEON::BI__builtin_neon_vst1_bf16_x2, NEON::BI__builtin_neon_vst1_x2_v},
    {NEON::BI__builtin_neon_vst1_bf16_x3, NEON::BI__builtin_neon_vst1_x3_v},
    {NEON::BI__builtin_neon_vst1_bf16_x4, NEON::BI__builtin_neon_vst1_x4_v},
    {NEON::BI__builtin_neon_vst1_bf16, NEON::BI__builtin_neon_vst1_v},
    {NEON::BI__builtin_neon_vst1_lane_bf16, NEON::BI__builtin_neon_vst1_lane_v},
    {NEON::BI__builtin_neon_vst1q_bf16_x2, NEON::BI__builtin_neon_vst1q_x2_v},
    {NEON::BI__builtin_neon_vst1q_bf16_x3, NEON::BI__builtin_neon_vst1q_x3_v},
    {NEON::BI__builtin_neon_vst1q_bf16_x4, NEON::BI__builtin_neon_vst1q_x4_v},
    {NEON::BI__builtin_neon_vst1q_bf16, NEON::BI__builtin_neon_vst1q_v},
    {NEON::BI__builtin_neon_vst1q_lane_bf16,
     NEON::BI__builtin_neon_vst1q_lane_v},
    {NEON::BI__builtin_neon_vst2_bf16, NEON::BI__builtin_neon_vst2_v},
    {NEON::BI__builtin_neon_vst2_lane_bf16, NEON::BI__builtin_neon_vst2_lane_v},
    {NEON::BI__builtin_neon_vst2q_bf16, NEON::BI__builtin_neon_vst2q_v},
    {NEON::BI__builtin_neon_vst2q_lane_bf16,
     NEON::BI__builtin_neon_vst2q_lane_v},
    {NEON::BI__builtin_neon_vst3_bf16, NEON::BI__builtin_neon_vst3_v},
    {NEON::BI__builtin_neon_vst3_lane_bf16, NEON::BI__builtin_neon_vst3_lane_v},
    {NEON::BI__builtin_neon_vst3q_bf16, NEON::BI__builtin_neon_vst3q_v},
    {NEON::BI__builtin_neon_vst3q_lane_bf16,
     NEON::BI__builtin_neon_vst3q_lane_v},
    {NEON::BI__builtin_neon_vst4_bf16, NEON::BI__builtin_neon_vst4_v},
    {NEON::BI__builtin_neon_vst4_lane_bf16, NEON::BI__builtin_neon_vst4_lane_v},
    {NEON::BI__builtin_neon_vst4q_bf16, NEON::BI__builtin_neon_vst4q_v},
    {NEON::BI__builtin_neon_vst4q_lane_bf16,
     NEON::BI__builtin_neon_vst4q_lane_v},
    // The mangling rules cause us to have one ID for each type for
    // vldap1(q)_lane and vstl1(q)_lane, but codegen is equivalent for all of
    // them. Choose an arbitrary one to be handled as tha canonical variation.
    {NEON::BI__builtin_neon_vldap1_lane_u64,
     NEON::BI__builtin_neon_vldap1_lane_s64},
    {NEON::BI__builtin_neon_vldap1_lane_f64,
     NEON::BI__builtin_neon_vldap1_lane_s64},
    {NEON::BI__builtin_neon_vldap1_lane_p64,
     NEON::BI__builtin_neon_vldap1_lane_s64},
    {NEON::BI__builtin_neon_vldap1q_lane_u64,
     NEON::BI__builtin_neon_vldap1q_lane_s64},
    {NEON::BI__builtin_neon_vldap1q_lane_f64,
     NEON::BI__builtin_neon_vldap1q_lane_s64},
    {NEON::BI__builtin_neon_vldap1q_lane_p64,
     NEON::BI__builtin_neon_vldap1q_lane_s64},
    {NEON::BI__builtin_neon_vstl1_lane_u64,
     NEON::BI__builtin_neon_vstl1_lane_s64},
    {NEON::BI__builtin_neon_vstl1_lane_f64,
     NEON::BI__builtin_neon_vstl1_lane_s64},
    {NEON::BI__builtin_neon_vstl1_lane_p64,
     NEON::BI__builtin_neon_vstl1_lane_s64},
    {NEON::BI__builtin_neon_vstl1q_lane_u64,
     NEON::BI__builtin_neon_vstl1q_lane_s64},
    {NEON::BI__builtin_neon_vstl1q_lane_f64,
     NEON::BI__builtin_neon_vstl1q_lane_s64},
    {NEON::BI__builtin_neon_vstl1q_lane_p64,
     NEON::BI__builtin_neon_vstl1q_lane_s64},
};

#undef NEONMAP0
#undef NEONMAP1
#undef NEONMAP2

#define SVEMAP1(NameBase, LLVMIntrinsic, TypeModifier)                         \
  {#NameBase, SVE::BI__builtin_sve_##NameBase, Intrinsic::LLVMIntrinsic, 0,    \
   TypeModifier}

#define SVEMAP2(NameBase, TypeModifier)                                        \
  {#NameBase, SVE::BI__builtin_sve_##NameBase, 0, 0, TypeModifier}
static const ARMVectorIntrinsicInfo AArch64SVEIntrinsicMap[] = {
#define GET_SVE_LLVM_INTRINSIC_MAP
#include "clang/Basic/BuiltinsAArch64NeonSVEBridge_cg.def"
#include "clang/Basic/arm_sve_builtin_cg.inc"
#undef GET_SVE_LLVM_INTRINSIC_MAP
};

#undef SVEMAP1
#undef SVEMAP2

#define SMEMAP1(NameBase, LLVMIntrinsic, TypeModifier)                         \
  {#NameBase, SME::BI__builtin_sme_##NameBase, Intrinsic::LLVMIntrinsic, 0,    \
   TypeModifier}

#define SMEMAP2(NameBase, TypeModifier)                                        \
  {#NameBase, SME::BI__builtin_sme_##NameBase, 0, 0, TypeModifier}
static const ARMVectorIntrinsicInfo AArch64SMEIntrinsicMap[] = {
#define GET_SME_LLVM_INTRINSIC_MAP
#include "clang/Basic/arm_sme_builtin_cg.inc"
#undef GET_SME_LLVM_INTRINSIC_MAP
};

#undef SMEMAP1
#undef SMEMAP2

static std::optional<CIRGenFunction::MSVCIntrin>
translateAarch64ToMsvcIntrin(unsigned BuiltinID) {
  using MSVCIntrin = CIRGenFunction::MSVCIntrin;
  switch (BuiltinID) {
  default:
    return std::nullopt;
  case clang::AArch64::BI_BitScanForward:
  case clang::AArch64::BI_BitScanForward64:
    return MSVCIntrin::_BitScanForward;
  case clang::AArch64::BI_BitScanReverse:
  case clang::AArch64::BI_BitScanReverse64:
    return MSVCIntrin::_BitScanReverse;
  case clang::AArch64::BI_InterlockedAnd64:
    return MSVCIntrin::_InterlockedAnd;
  case clang::AArch64::BI_InterlockedExchange64:
    return MSVCIntrin::_InterlockedExchange;
  case clang::AArch64::BI_InterlockedExchangeAdd64:
    return MSVCIntrin::_InterlockedExchangeAdd;
  case clang::AArch64::BI_InterlockedExchangeSub64:
    return MSVCIntrin::_InterlockedExchangeSub;
  case clang::AArch64::BI_InterlockedOr64:
    return MSVCIntrin::_InterlockedOr;
  case clang::AArch64::BI_InterlockedXor64:
    return MSVCIntrin::_InterlockedXor;
  case clang::AArch64::BI_InterlockedDecrement64:
    return MSVCIntrin::_InterlockedDecrement;
  case clang::AArch64::BI_InterlockedIncrement64:
    return MSVCIntrin::_InterlockedIncrement;
  case clang::AArch64::BI_InterlockedExchangeAdd8_acq:
  case clang::AArch64::BI_InterlockedExchangeAdd16_acq:
  case clang::AArch64::BI_InterlockedExchangeAdd_acq:
  case clang::AArch64::BI_InterlockedExchangeAdd64_acq:
    return MSVCIntrin::_InterlockedExchangeAdd_acq;
  case clang::AArch64::BI_InterlockedExchangeAdd8_rel:
  case clang::AArch64::BI_InterlockedExchangeAdd16_rel:
  case clang::AArch64::BI_InterlockedExchangeAdd_rel:
  case clang::AArch64::BI_InterlockedExchangeAdd64_rel:
    return MSVCIntrin::_InterlockedExchangeAdd_rel;
  case clang::AArch64::BI_InterlockedExchangeAdd8_nf:
  case clang::AArch64::BI_InterlockedExchangeAdd16_nf:
  case clang::AArch64::BI_InterlockedExchangeAdd_nf:
  case clang::AArch64::BI_InterlockedExchangeAdd64_nf:
    return MSVCIntrin::_InterlockedExchangeAdd_nf;
  case clang::AArch64::BI_InterlockedExchange8_acq:
  case clang::AArch64::BI_InterlockedExchange16_acq:
  case clang::AArch64::BI_InterlockedExchange_acq:
  case clang::AArch64::BI_InterlockedExchange64_acq:
    return MSVCIntrin::_InterlockedExchange_acq;
  case clang::AArch64::BI_InterlockedExchange8_rel:
  case clang::AArch64::BI_InterlockedExchange16_rel:
  case clang::AArch64::BI_InterlockedExchange_rel:
  case clang::AArch64::BI_InterlockedExchange64_rel:
    return MSVCIntrin::_InterlockedExchange_rel;
  case clang::AArch64::BI_InterlockedExchange8_nf:
  case clang::AArch64::BI_InterlockedExchange16_nf:
  case clang::AArch64::BI_InterlockedExchange_nf:
  case clang::AArch64::BI_InterlockedExchange64_nf:
    return MSVCIntrin::_InterlockedExchange_nf;
  case clang::AArch64::BI_InterlockedCompareExchange8_acq:
  case clang::AArch64::BI_InterlockedCompareExchange16_acq:
  case clang::AArch64::BI_InterlockedCompareExchange_acq:
  case clang::AArch64::BI_InterlockedCompareExchange64_acq:
    return MSVCIntrin::_InterlockedCompareExchange_acq;
  case clang::AArch64::BI_InterlockedCompareExchange8_rel:
  case clang::AArch64::BI_InterlockedCompareExchange16_rel:
  case clang::AArch64::BI_InterlockedCompareExchange_rel:
  case clang::AArch64::BI_InterlockedCompareExchange64_rel:
    return MSVCIntrin::_InterlockedCompareExchange_rel;
  case clang::AArch64::BI_InterlockedCompareExchange8_nf:
  case clang::AArch64::BI_InterlockedCompareExchange16_nf:
  case clang::AArch64::BI_InterlockedCompareExchange_nf:
  case clang::AArch64::BI_InterlockedCompareExchange64_nf:
    return MSVCIntrin::_InterlockedCompareExchange_nf;
  case clang::AArch64::BI_InterlockedCompareExchange128:
    return MSVCIntrin::_InterlockedCompareExchange128;
  case clang::AArch64::BI_InterlockedCompareExchange128_acq:
    return MSVCIntrin::_InterlockedCompareExchange128_acq;
  case clang::AArch64::BI_InterlockedCompareExchange128_nf:
    return MSVCIntrin::_InterlockedCompareExchange128_nf;
  case clang::AArch64::BI_InterlockedCompareExchange128_rel:
    return MSVCIntrin::_InterlockedCompareExchange128_rel;
  case clang::AArch64::BI_InterlockedOr8_acq:
  case clang::AArch64::BI_InterlockedOr16_acq:
  case clang::AArch64::BI_InterlockedOr_acq:
  case clang::AArch64::BI_InterlockedOr64_acq:
    return MSVCIntrin::_InterlockedOr_acq;
  case clang::AArch64::BI_InterlockedOr8_rel:
  case clang::AArch64::BI_InterlockedOr16_rel:
  case clang::AArch64::BI_InterlockedOr_rel:
  case clang::AArch64::BI_InterlockedOr64_rel:
    return MSVCIntrin::_InterlockedOr_rel;
  case clang::AArch64::BI_InterlockedOr8_nf:
  case clang::AArch64::BI_InterlockedOr16_nf:
  case clang::AArch64::BI_InterlockedOr_nf:
  case clang::AArch64::BI_InterlockedOr64_nf:
    return MSVCIntrin::_InterlockedOr_nf;
  case clang::AArch64::BI_InterlockedXor8_acq:
  case clang::AArch64::BI_InterlockedXor16_acq:
  case clang::AArch64::BI_InterlockedXor_acq:
  case clang::AArch64::BI_InterlockedXor64_acq:
    return MSVCIntrin::_InterlockedXor_acq;
  case clang::AArch64::BI_InterlockedXor8_rel:
  case clang::AArch64::BI_InterlockedXor16_rel:
  case clang::AArch64::BI_InterlockedXor_rel:
  case clang::AArch64::BI_InterlockedXor64_rel:
    return MSVCIntrin::_InterlockedXor_rel;
  case clang::AArch64::BI_InterlockedXor8_nf:
  case clang::AArch64::BI_InterlockedXor16_nf:
  case clang::AArch64::BI_InterlockedXor_nf:
  case clang::AArch64::BI_InterlockedXor64_nf:
    return MSVCIntrin::_InterlockedXor_nf;
  case clang::AArch64::BI_InterlockedAnd8_acq:
  case clang::AArch64::BI_InterlockedAnd16_acq:
  case clang::AArch64::BI_InterlockedAnd_acq:
  case clang::AArch64::BI_InterlockedAnd64_acq:
    return MSVCIntrin::_InterlockedAnd_acq;
  case clang::AArch64::BI_InterlockedAnd8_rel:
  case clang::AArch64::BI_InterlockedAnd16_rel:
  case clang::AArch64::BI_InterlockedAnd_rel:
  case clang::AArch64::BI_InterlockedAnd64_rel:
    return MSVCIntrin::_InterlockedAnd_rel;
  case clang::AArch64::BI_InterlockedAnd8_nf:
  case clang::AArch64::BI_InterlockedAnd16_nf:
  case clang::AArch64::BI_InterlockedAnd_nf:
  case clang::AArch64::BI_InterlockedAnd64_nf:
    return MSVCIntrin::_InterlockedAnd_nf;
  case clang::AArch64::BI_InterlockedIncrement16_acq:
  case clang::AArch64::BI_InterlockedIncrement_acq:
  case clang::AArch64::BI_InterlockedIncrement64_acq:
    return MSVCIntrin::_InterlockedIncrement_acq;
  case clang::AArch64::BI_InterlockedIncrement16_rel:
  case clang::AArch64::BI_InterlockedIncrement_rel:
  case clang::AArch64::BI_InterlockedIncrement64_rel:
    return MSVCIntrin::_InterlockedIncrement_rel;
  case clang::AArch64::BI_InterlockedIncrement16_nf:
  case clang::AArch64::BI_InterlockedIncrement_nf:
  case clang::AArch64::BI_InterlockedIncrement64_nf:
    return MSVCIntrin::_InterlockedIncrement_nf;
  case clang::AArch64::BI_InterlockedDecrement16_acq:
  case clang::AArch64::BI_InterlockedDecrement_acq:
  case clang::AArch64::BI_InterlockedDecrement64_acq:
    return MSVCIntrin::_InterlockedDecrement_acq;
  case clang::AArch64::BI_InterlockedDecrement16_rel:
  case clang::AArch64::BI_InterlockedDecrement_rel:
  case clang::AArch64::BI_InterlockedDecrement64_rel:
    return MSVCIntrin::_InterlockedDecrement_rel;
  case clang::AArch64::BI_InterlockedDecrement16_nf:
  case clang::AArch64::BI_InterlockedDecrement_nf:
  case clang::AArch64::BI_InterlockedDecrement64_nf:
    return MSVCIntrin::_InterlockedDecrement_nf;
  }
  llvm_unreachable("must return from switch");
}

static bool AArch64SIMDIntrinsicsProvenSorted = false;
static bool AArch64SISDIntrinsicsProvenSorted = false;
static bool AArch64SVEIntrinsicsProvenSorted = false;
static bool AArch64SMEIntrinsicsProvenSorted = false;

static const ARMVectorIntrinsicInfo *
findARMVectorIntrinsicInMap(ArrayRef<ARMVectorIntrinsicInfo> IntrinsicMap,
                            unsigned BuiltinID, bool &MapProvenSorted) {

#ifndef NDEBUG
  if (!MapProvenSorted) {
    assert(llvm::is_sorted(IntrinsicMap));
    MapProvenSorted = true;
  }
#endif

  const ARMVectorIntrinsicInfo *Builtin =
      llvm::lower_bound(IntrinsicMap, BuiltinID);

  if (Builtin != IntrinsicMap.end() && Builtin->BuiltinID == BuiltinID)
    return Builtin;

  return nullptr;
}

static cir::VectorType GetNeonType(CIRGenFunction *CGF, NeonTypeFlags TypeFlags,
                                   bool HasLegalHalfType = true,
                                   bool V1Ty = false,
                                   bool AllowBFloatArgsAndRet = true) {
  int IsQuad = TypeFlags.isQuad();
  switch (TypeFlags.getEltType()) {
  case NeonTypeFlags::Int8:
  case NeonTypeFlags::Poly8:
    return cir::VectorType::get(CGF->getBuilder().getContext(),
                                TypeFlags.isUnsigned() ? CGF->UInt8Ty
                                                       : CGF->SInt8Ty,
                                V1Ty ? 1 : (8 << IsQuad));
  case NeonTypeFlags::Int16:
  case NeonTypeFlags::Poly16:
    return cir::VectorType::get(CGF->getBuilder().getContext(),
                                TypeFlags.isUnsigned() ? CGF->UInt16Ty
                                                       : CGF->SInt16Ty,
                                V1Ty ? 1 : (4 << IsQuad));
  case NeonTypeFlags::BFloat16:
    if (AllowBFloatArgsAndRet)
      llvm_unreachable("NeonTypeFlags::BFloat16 NYI");
    else
      llvm_unreachable("NeonTypeFlags::BFloat16 NYI");
  case NeonTypeFlags::Float16:
    if (HasLegalHalfType)
      llvm_unreachable("NeonTypeFlags::Float16 NYI");
    else
      llvm_unreachable("NeonTypeFlags::Float16 NYI");
  case NeonTypeFlags::Int32:
    return cir::VectorType::get(CGF->getBuilder().getContext(),
                                TypeFlags.isUnsigned() ? CGF->UInt32Ty
                                                       : CGF->SInt32Ty,
                                V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Int64:
  case NeonTypeFlags::Poly64:
    return cir::VectorType::get(CGF->getBuilder().getContext(),
                                TypeFlags.isUnsigned() ? CGF->UInt64Ty
                                                       : CGF->SInt64Ty,
                                V1Ty ? 1 : (1 << IsQuad));
  case NeonTypeFlags::Poly128:
    // FIXME: i128 and f128 doesn't get fully support in Clang and llvm.
    // There is a lot of i128 and f128 API missing.
    // so we use v16i8 to represent poly128 and get pattern matched.
    llvm_unreachable("NeonTypeFlags::Poly128 NYI");
  case NeonTypeFlags::Float32:
    return cir::VectorType::get(CGF->getBuilder().getContext(),
                                CGF->getCIRGenModule().FloatTy,
                                V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Float64:
    return cir::VectorType::get(CGF->getBuilder().getContext(),
                                CGF->getCIRGenModule().DoubleTy,
                                V1Ty ? 1 : (1 << IsQuad));
  }
  llvm_unreachable("Unknown vector element type!");
}

static mlir::Value emitAArch64TblBuiltinExpr(CIRGenFunction &CGF,
                                             unsigned BuiltinID,
                                             const CallExpr *E,
                                             SmallVectorImpl<mlir::Value> &Ops,
                                             llvm::Triple::ArchType Arch) {
  unsigned int Int = 0;
  [[maybe_unused]] const char *s = nullptr;

  switch (BuiltinID) {
  default:
    return {};
  case NEON::BI__builtin_neon_vtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1q_v:
  case NEON::BI__builtin_neon_vtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2q_v:
  case NEON::BI__builtin_neon_vtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3q_v:
  case NEON::BI__builtin_neon_vtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4q_v:
    break;
  case NEON::BI__builtin_neon_vtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1q_v:
  case NEON::BI__builtin_neon_vtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2q_v:
  case NEON::BI__builtin_neon_vtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3q_v:
  case NEON::BI__builtin_neon_vtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4q_v:
    break;
  }

  assert(E->getNumArgs() >= 3);

  // Get the last argument, which specifies the vector type.
  const Expr *Arg = E->getArg(E->getNumArgs() - 1);
  std::optional<llvm::APSInt> Result =
      Arg->getIntegerConstantExpr(CGF.getContext());
  if (!Result)
    return nullptr;

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type = Result->getZExtValue();
  cir::VectorType Ty = GetNeonType(&CGF, Type);
  if (!Ty)
    return nullptr;

  // AArch64 scalar builtins are not overloaded, they do not have an extra
  // argument that specifies the vector type, need to handle each case.
  switch (BuiltinID) {
  case NEON::BI__builtin_neon_vtbl1_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vtbl1_v NYI");
  }
  case NEON::BI__builtin_neon_vtbl2_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vtbl2_v NYI");
  }
  case NEON::BI__builtin_neon_vtbl3_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vtbl3_v NYI");
  }
  case NEON::BI__builtin_neon_vtbl4_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vtbl4_v NYI");
  }
  case NEON::BI__builtin_neon_vtbx1_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vtbx1_v NYI");
  }
  case NEON::BI__builtin_neon_vtbx2_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vtbx2_v NYI");
  }
  case NEON::BI__builtin_neon_vtbx3_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vtbx3_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1q_v:
    Int = Intrinsic::aarch64_neon_tbl1;
    s = "vtbl1";
    break;
  case NEON::BI__builtin_neon_vqtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2q_v: {
    Int = Intrinsic::aarch64_neon_tbl2;
    s = "vtbl2";
    break;
  case NEON::BI__builtin_neon_vqtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3q_v:
    Int = Intrinsic::aarch64_neon_tbl3;
    s = "vtbl3";
    break;
  case NEON::BI__builtin_neon_vqtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4q_v:
    Int = Intrinsic::aarch64_neon_tbl4;
    s = "vtbl4";
    break;
  case NEON::BI__builtin_neon_vqtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1q_v:
    Int = Intrinsic::aarch64_neon_tbx1;
    s = "vtbx1";
    break;
  case NEON::BI__builtin_neon_vqtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2q_v:
    Int = Intrinsic::aarch64_neon_tbx2;
    s = "vtbx2";
    break;
  case NEON::BI__builtin_neon_vqtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3q_v:
    Int = Intrinsic::aarch64_neon_tbx3;
    s = "vtbx3";
    break;
  case NEON::BI__builtin_neon_vqtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4q_v:
    Int = Intrinsic::aarch64_neon_tbx4;
    s = "vtbx4";
    break;
  }
  }

  if (!Int)
    return nullptr;

  llvm_unreachable("NYI");
}

mlir::Value CIRGenFunction::emitAArch64SMEBuiltinExpr(unsigned BuiltinID,
                                                      const CallExpr *E) {
  auto *Builtin = findARMVectorIntrinsicInMap(AArch64SMEIntrinsicMap, BuiltinID,
                                              AArch64SMEIntrinsicsProvenSorted);
  (void)Builtin;
  llvm_unreachable("NYI");
}

mlir::Value CIRGenFunction::emitAArch64SVEBuiltinExpr(unsigned BuiltinID,
                                                      const CallExpr *E) {
  if (BuiltinID >= SVE::BI__builtin_sve_reinterpret_s8_s8 &&
      BuiltinID <= SVE::BI__builtin_sve_reinterpret_f64_f64_x4) {
    llvm_unreachable("NYI");
  }
  auto *Builtin = findARMVectorIntrinsicInMap(AArch64SVEIntrinsicMap, BuiltinID,
                                              AArch64SVEIntrinsicsProvenSorted);
  (void)Builtin;
  llvm_unreachable("NYI");
}

static mlir::Value emitArmLdrexNon128Intrinsic(unsigned int builtinID,
                                               const CallExpr *clangCallExpr,
                                               CIRGenFunction &cgf) {
  StringRef intrinsicName = builtinID == clang::AArch64::BI__builtin_arm_ldrex
                                ? "aarch64.ldxr"
                                : "aarch64.ldaxr";

  // Argument
  mlir::Value loadAddr = cgf.emitScalarExpr(clangCallExpr->getArg(0));
  // Get Instrinc call
  CIRGenBuilderTy &builder = cgf.getBuilder();
  QualType clangResTy = clangCallExpr->getType();
  mlir::Type realResTy = cgf.convertType(clangResTy);
  // Return type of LLVM intrinsic is defined in Intrinsic<arch_type>.td,
  // which can be found under LLVM IR directory.
  mlir::Type funcResTy = builder.getSInt64Ty();
  mlir::Location loc = cgf.getLoc(clangCallExpr->getExprLoc());
  cir::LLVMIntrinsicCallOp op = builder.create<cir::LLVMIntrinsicCallOp>(
      loc, builder.getStringAttr(intrinsicName), funcResTy, loadAddr);
  mlir::Value res = op.getResult();

  // Convert result type to the expected type.
  if (mlir::isa<cir::PointerType>(realResTy)) {
    return builder.createIntToPtr(res, realResTy);
  }
  cir::IntType intResTy =
      builder.getSIntNTy(cgf.CGM.getDataLayout().getTypeSizeInBits(realResTy));
  mlir::Value intCastRes = builder.createIntCast(res, intResTy);
  if (mlir::isa<cir::IntType>(realResTy)) {
    return builder.createIntCast(intCastRes, realResTy);
  } else {
    // Above cases should cover most situations and we have test coverage.
    llvm_unreachable("Unsupported return type for now");
  }
}

/// Given a vector of int type `vecTy`, return a vector type of
/// int type with the same element type width, different signedness,
/// and the same vector size.
static cir::VectorType getSignChangedVectorType(CIRGenBuilderTy &builder,
                                                cir::VectorType vecTy) {
  auto elemTy = mlir::cast<cir::IntType>(vecTy.getEltType());
  elemTy = elemTy.isSigned() ? builder.getUIntNTy(elemTy.getWidth())
                             : builder.getSIntNTy(elemTy.getWidth());
  return cir::VectorType::get(builder.getContext(), elemTy, vecTy.getSize());
}

static cir::VectorType
getHalfEltSizeTwiceNumElemsVecType(CIRGenBuilderTy &builder,
                                   cir::VectorType vecTy) {
  auto elemTy = mlir::cast<cir::IntType>(vecTy.getEltType());
  elemTy = elemTy.isSigned() ? builder.getSIntNTy(elemTy.getWidth() / 2)
                             : builder.getUIntNTy(elemTy.getWidth() / 2);
  return cir::VectorType::get(builder.getContext(), elemTy,
                              vecTy.getSize() * 2);
}

static cir::VectorType
castVecOfFPTypeToVecOfIntWithSameWidth(CIRGenBuilderTy &builder,
                                       cir::VectorType vecTy) {
  if (mlir::isa<cir::SingleType>(vecTy.getEltType()))
    return cir::VectorType::get(builder.getContext(), builder.getSInt32Ty(),
                                vecTy.getSize());
  if (mlir::isa<cir::DoubleType>(vecTy.getEltType()))
    return cir::VectorType::get(builder.getContext(), builder.getSInt64Ty(),
                                vecTy.getSize());
  llvm_unreachable(
      "Unsupported element type in getVecOfIntTypeWithSameEltWidth");
}

/// Get integer from a mlir::Value that is an int constant or a constant op.
static int64_t getIntValueFromConstOp(mlir::Value val) {
  auto constOp = mlir::cast<cir::ConstantOp>(val.getDefiningOp());
  return (mlir::cast<cir::IntAttr>(constOp.getValue()))
      .getValue()
      .getSExtValue();
}

static mlir::Value emitNeonSplat(CIRGenBuilderTy &builder, mlir::Location loc,
                                 mlir::Value splatVec, mlir::Value splatLane,
                                 unsigned int splatCnt) {
  int64_t splatValInt = getIntValueFromConstOp(splatLane);
  llvm::SmallVector<int64_t, 4> splatMask(splatCnt, splatValInt);
  return builder.createVecShuffle(loc, splatVec, splatMask);
}

/// Build a constant shift amount vector of `vecTy` to shift a vector
/// Here `shitfVal` is a constant integer that will be splated into a
/// a const vector of `vecTy` which is the return of this function
static mlir::Value emitNeonShiftVector(CIRGenBuilderTy &builder,
                                       mlir::Value shiftVal,
                                       cir::VectorType vecTy,
                                       mlir::Location loc, bool neg) {
  int shiftAmt = getIntValueFromConstOp(shiftVal);
  if (neg)
    shiftAmt = -shiftAmt;
  llvm::SmallVector<mlir::Attribute> vecAttr{
      vecTy.getSize(),
      // ConstVectorAttr requires cir::IntAttr
      cir::IntAttr::get(vecTy.getEltType(), shiftAmt)};
  cir::ConstVectorAttr constVecAttr = cir::ConstVectorAttr::get(
      vecTy, mlir::ArrayAttr::get(builder.getContext(), vecAttr));
  return builder.create<cir::ConstantOp>(loc, vecTy, constVecAttr);
}

/// Build ShiftOp of vector type whose shift amount is a vector built
/// from a constant integer using `emitNeonShiftVector` function
static mlir::Value
emitCommonNeonShift(CIRGenBuilderTy &builder, mlir::Location loc,
                    cir::VectorType resTy, mlir::Value shifTgt,
                    mlir::Value shiftAmt, bool shiftLeft, bool negAmt = false) {
  shiftAmt = emitNeonShiftVector(builder, shiftAmt, resTy, loc, negAmt);
  return builder.create<cir::ShiftOp>(
      loc, resTy, builder.createBitcast(shifTgt, resTy), shiftAmt, shiftLeft);
}

/// Right-shift a vector by a constant.
static mlir::Value emitNeonRShiftImm(CIRGenFunction &cgf, mlir::Value shiftVec,
                                     mlir::Value shiftVal,
                                     cir::VectorType vecTy, bool usgn,
                                     mlir::Location loc) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  int64_t shiftAmt = getIntValueFromConstOp(shiftVal);
  int eltSize = cgf.CGM.getDataLayout().getTypeSizeInBits(vecTy.getEltType());

  shiftVec = builder.createBitcast(shiftVec, vecTy);
  // lshr/ashr are undefined when the shift amount is equal to the vector
  // element size.
  if (shiftAmt == eltSize) {
    if (usgn) {
      // Right-shifting an unsigned value by its size yields 0.
      return builder.getZero(loc, vecTy);
    }
    // Right-shifting a signed value by its size is equivalent
    // to a shift of size-1.
    --shiftAmt;
    shiftVal = builder.getConstInt(loc, vecTy.getEltType(), shiftAmt);
  }
  return emitCommonNeonShift(builder, loc, vecTy, shiftVec, shiftVal,
                             false /* right shift */);
}

/// Vectorize value, usually for argument of a neon SISD intrinsic call.
static void vecExtendIntValue(CIRGenFunction &cgf, cir::VectorType argVTy,
                              mlir::Value &arg, mlir::Location loc) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  cir::IntType eltTy = mlir::dyn_cast<cir::IntType>(argVTy.getEltType());
  assert(mlir::isa<cir::IntType>(arg.getType()) && eltTy);
  // The constant argument to an _n_ intrinsic always has Int32Ty, so truncate
  // it before inserting.
  arg = builder.createIntCast(arg, eltTy);
  mlir::Value zero = builder.getConstInt(loc, cgf.SizeTy, 0);
  mlir::Value poison = builder.create<cir::ConstantOp>(
      loc, eltTy, builder.getAttr<cir::PoisonAttr>(eltTy));
  arg = builder.create<cir::VecInsertOp>(
      loc, builder.create<cir::VecSplatOp>(loc, argVTy, poison), arg, zero);
}

/// Reduce vector type value to scalar, usually for result of a
/// neon SISD intrinsic call
static mlir::Value vecReduceIntValue(CIRGenFunction &cgf, mlir::Value val,
                                     mlir::Location loc) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  assert(mlir::isa<cir::VectorType>(val.getType()));
  return builder.create<cir::VecExtractOp>(
      loc, val, builder.getConstInt(loc, cgf.SizeTy, 0));
}

mlir::Value emitNeonCall(CIRGenBuilderTy &builder,
                         llvm::SmallVector<mlir::Type> argTypes,
                         llvm::SmallVectorImpl<mlir::Value> &args,
                         llvm::StringRef intrinsicName, mlir::Type funcResTy,
                         mlir::Location loc,
                         bool isConstrainedFPIntrinsic = false,
                         unsigned shift = 0, bool rightshift = false) {
  // TODO: Consider removing the following unreachable when we have
  // emitConstrainedFPCall feature implemented
  assert(!cir::MissingFeatures::emitConstrainedFPCall());
  if (isConstrainedFPIntrinsic)
    llvm_unreachable("isConstrainedFPIntrinsic NYI");

  for (unsigned j = 0; j < argTypes.size(); ++j) {
    if (isConstrainedFPIntrinsic) {
      assert(!cir::MissingFeatures::emitConstrainedFPCall());
    }
    if (shift > 0 && shift == j) {
      args[j] = emitNeonShiftVector(builder, args[j],
                                    mlir::cast<cir::VectorType>(argTypes[j]),
                                    loc, rightshift);
    } else {
      args[j] = builder.createBitcast(args[j], argTypes[j]);
    }
  }
  if (isConstrainedFPIntrinsic) {
    assert(!cir::MissingFeatures::emitConstrainedFPCall());
    return nullptr;
  }
  return builder
      .create<cir::LLVMIntrinsicCallOp>(
          loc, builder.getStringAttr(intrinsicName), funcResTy, args)
      .getResult();
}

/// This function `emitCommonNeonCallPattern0` implements a common way
///  to generate neon intrinsic call that has following pattern:
///  1. There is a need to cast result of the intrinsic call back to
///     expression type.
///  2. Function arg types are given, not deduced from actual arg types.
static mlir::Value
emitCommonNeonCallPattern0(CIRGenFunction &cgf, llvm::StringRef intrincsName,
                           llvm::SmallVector<mlir::Type> argTypes,
                           llvm::SmallVectorImpl<mlir::Value> &ops,
                           mlir::Type funcResTy, const clang::CallExpr *e) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  if (argTypes.empty()) {
    // The most common arg types is {funcResTy, funcResTy} for neon intrinsic
    // functions. Thus, it is as default so call site does not need to
    // provide it. Every neon intrinsic function has at least one argument,
    // Thus empty argTypes really just means {funcResTy, funcResTy}.
    argTypes = {funcResTy, funcResTy};
  }
  mlir::Value res =
      emitNeonCall(builder, std::move(argTypes), ops, intrincsName, funcResTy,
                   cgf.getLoc(e->getExprLoc()));
  mlir::Type resultType = cgf.convertType(e->getType());
  return builder.createBitcast(res, resultType);
}

/// The function `emitCommonNeonVecAcrossCall` implements a common way
/// to implement neon intrinsic which has the following pattern:
///  1. There is only one argument which is of vector type
///  2. The result of the neon intrinsic is the element type of the input.
/// This type of intrinsic usually is for across operations of the input vector.

static mlir::Value emitCommonNeonVecAcrossCall(CIRGenFunction &cgf,
                                               llvm::StringRef intrincsName,
                                               mlir::Type eltTy,
                                               unsigned vecLen,
                                               const clang::CallExpr *e) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Value op = cgf.emitScalarExpr(e->getArg(0));
  cir::VectorType vTy =
      cir::VectorType::get(&cgf.getMLIRContext(), eltTy, vecLen);
  llvm::SmallVector<mlir::Value, 1> args{op};
  return emitNeonCall(builder, {vTy}, args, intrincsName, eltTy,
                      cgf.getLoc(e->getExprLoc()));
}

mlir::Value CIRGenFunction::emitCommonNeonBuiltinExpr(
    unsigned builtinID, unsigned llvmIntrinsic, unsigned altLLVMIntrinsic,
    const char *nameHint, unsigned modifier, const CallExpr *e,
    llvm::SmallVectorImpl<mlir::Value> &ops, Address ptrOp0, Address ptrOp1,
    llvm::Triple::ArchType arch) {
  // Get the last argument, which specifies the vector type.
  const clang::Expr *arg = e->getArg(e->getNumArgs() - 1);
  std::optional<llvm::APSInt> neonTypeConst =
      arg->getIntegerConstantExpr(getContext());
  if (!neonTypeConst)
    return nullptr;

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags neonType(neonTypeConst->getZExtValue());
  bool isUnsigned = neonType.isUnsigned();
  bool isQuad = neonType.isQuad();
  const bool hasLegalHalfType = getTarget().hasLegalHalfType();
  // The value of allowBFloatArgsAndRet is true for AArch64, but it should
  // come from ABI info.
  const bool allowBFloatArgsAndRet =
      getTargetHooks().getABIInfo().allowBFloatArgsAndRet();

  cir::VectorType vTy = GetNeonType(this, neonType, hasLegalHalfType, false,
                                    allowBFloatArgsAndRet);
  mlir::Type ty = vTy;
  if (!ty)
    return nullptr;

  unsigned intrinicId = llvmIntrinsic;
  if ((modifier & UnsignedAlts) && !isUnsigned)
    intrinicId = altLLVMIntrinsic;

  // This first switch is for the intrinsics that cannot have a more generic
  // codegen solution.
  switch (builtinID) {
  default:
    break;
  case NEON::BI__builtin_neon_splat_lane_v:
  case NEON::BI__builtin_neon_splat_laneq_v:
  case NEON::BI__builtin_neon_splatq_lane_v:
  case NEON::BI__builtin_neon_splatq_laneq_v: {
    uint64_t numElements = vTy.getSize();
    if (builtinID == NEON::BI__builtin_neon_splatq_lane_v)
      numElements = numElements << 1;
    if (builtinID == NEON::BI__builtin_neon_splat_laneq_v)
      numElements = numElements >> 1;
    ops[0] = builder.createBitcast(ops[0], vTy);
    return emitNeonSplat(builder, getLoc(e->getExprLoc()), ops[0], ops[1],
                         numElements);
  }
  case NEON::BI__builtin_neon_vabs_v:
  case NEON::BI__builtin_neon_vabsq_v: {
    mlir::Location loc = getLoc(e->getExprLoc());
    ops[0] = builder.createBitcast(ops[0], vTy);
    if (mlir::isa<cir::SingleType, cir::DoubleType>(vTy.getEltType())) {
      return builder.create<cir::FAbsOp>(loc, ops[0]);
    }
    return builder.create<cir::AbsOp>(loc, ops[0]);
  }
  case NEON::BI__builtin_neon_vmovl_v: {
    cir::VectorType dTy = builder.getExtendedOrTruncatedElementVectorType(
        vTy, false /* truncate */,
        mlir::cast<cir::IntType>(vTy.getEltType()).isSigned());
    // This cast makes sure arg type conforms intrinsic expected arg type.
    ops[0] = builder.createBitcast(ops[0], dTy);
    return builder.createIntCast(ops[0], ty);
  }
  case NEON::BI__builtin_neon_vmovn_v: {
    cir::VectorType qTy = builder.getExtendedOrTruncatedElementVectorType(
        vTy, true, mlir::cast<cir::IntType>(vTy.getEltType()).isSigned());
    ops[0] = builder.createBitcast(ops[0], qTy);
    // It really is truncation in this context.
    // In CIR, integral cast op supports vector of int type truncating.
    return builder.createIntCast(ops[0], ty);
  }
  case NEON::BI__builtin_neon_vpaddl_v:
  case NEON::BI__builtin_neon_vpaddlq_v: {
    // The source operand type has twice as many elements of half the size.
    cir::VectorType narrowTy = getHalfEltSizeTwiceNumElemsVecType(builder, vTy);
    return emitNeonCall(builder, {narrowTy}, ops,
                        isUnsigned ? "aarch64.neon.uaddlp"
                                   : "aarch64.neon.saddlp",
                        vTy, getLoc(e->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vqdmlal_v:
  case NEON::BI__builtin_neon_vqdmlsl_v: {
    llvm::SmallVector<mlir::Value, 2> mulOps(ops.begin() + 1, ops.end());
    cir::VectorType srcVty = builder.getExtendedOrTruncatedElementVectorType(
        vTy, false, /* truncate */
        mlir::cast<cir::IntType>(vTy.getEltType()).isSigned());
    ops[1] = emitNeonCall(builder, {srcVty, srcVty}, mulOps,
                          "aarch64.neon.sqdmull", vTy, getLoc(e->getExprLoc()));
    ops.resize(2);
    return emitNeonCall(builder, {vTy, vTy}, ops,
                        builtinID == NEON::BI__builtin_neon_vqdmlal_v
                            ? "aarch64.neon.sqadd"
                            : "aarch64.neon.sqsub",
                        vTy, getLoc(e->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vcvt_f32_v:
  case NEON::BI__builtin_neon_vcvtq_f32_v: {
    ops[0] = builder.createBitcast(ops[0], ty);
    ty = GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, isQuad),
                     hasLegalHalfType);
    return builder.createCast(cir::CastKind::int_to_float, ops[0], ty);
  }
  case NEON::BI__builtin_neon_vext_v:
  case NEON::BI__builtin_neon_vextq_v: {
    int cv = getIntValueFromConstOp(ops[2]);
    llvm::SmallVector<int64_t, 16> indices;
    for (unsigned i = 0, e = vTy.getSize(); i != e; ++i)
      indices.push_back(i + cv);

    ops[0] = builder.createBitcast(ops[0], ty);
    ops[1] = builder.createBitcast(ops[1], ty);
    return builder.createVecShuffle(getLoc(e->getExprLoc()), ops[0], ops[1],
                                    indices);
  }
  case NEON::BI__builtin_neon_vqdmulhq_lane_v:
  case NEON::BI__builtin_neon_vqdmulh_lane_v:
  case NEON::BI__builtin_neon_vqrdmulhq_lane_v:
  case NEON::BI__builtin_neon_vqrdmulh_lane_v: {
    cir::VectorType resTy =
        (builtinID == NEON::BI__builtin_neon_vqdmulhq_lane_v ||
         builtinID == NEON::BI__builtin_neon_vqrdmulhq_lane_v)
            ? cir::VectorType::get(&getMLIRContext(), vTy.getEltType(),
                                   vTy.getSize() * 2)
            : vTy;
    cir::VectorType mulVecT =
        GetNeonType(this, NeonTypeFlags(neonType.getEltType(), false,
                                        /*isQuad*/ false));
    return emitNeonCall(builder, {resTy, mulVecT, SInt32Ty}, ops,
                        (builtinID == NEON::BI__builtin_neon_vqdmulhq_lane_v ||
                         builtinID == NEON::BI__builtin_neon_vqdmulh_lane_v)
                            ? "aarch64.neon.sqdmulh.lane"
                            : "aarch64.neon.sqrdmulh.lane",
                        resTy, getLoc(e->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vqshlu_n_v:
  case NEON::BI__builtin_neon_vqshluq_n_v: {
    // These intrinsics expect signed vector type as input, but
    // return unsigned vector type.
    cir::VectorType srcTy = getSignChangedVectorType(builder, vTy);
    return emitNeonCall(builder, {srcTy, srcTy}, ops, "aarch64.neon.sqshlu",
                        vTy, getLoc(e->getExprLoc()),
                        false, /* not fp constrained op */
                        1,     /* second arg is shift amount */
                        false /* leftshift */);
  }
  case NEON::BI__builtin_neon_vrshr_n_v:
  case NEON::BI__builtin_neon_vrshrq_n_v: {
    return emitNeonCall(
        builder,
        {vTy, isUnsigned ? getSignChangedVectorType(builder, vTy) : vTy}, ops,
        isUnsigned ? "aarch64.neon.urshl" : "aarch64.neon.srshl", vTy,
        getLoc(e->getExprLoc()), false, /* not fp constrained op*/
        1,                              /* second arg is shift amount */
        true /* rightshift */);
  }
  case NEON::BI__builtin_neon_vshl_n_v:
  case NEON::BI__builtin_neon_vshlq_n_v: {
    mlir::Location loc = getLoc(e->getExprLoc());
    return emitCommonNeonShift(builder, loc, vTy, ops[0], ops[1], true);
  }
  case NEON::BI__builtin_neon_vshll_n_v: {
    mlir::Location loc = getLoc(e->getExprLoc());
    cir::VectorType srcTy = builder.getExtendedOrTruncatedElementVectorType(
        vTy, false /* truncate */,
        mlir::cast<cir::IntType>(vTy.getEltType()).isSigned());
    ops[0] = builder.createBitcast(ops[0], srcTy);
    // The following cast will be lowered to SExt or ZExt in LLVM.
    ops[0] = builder.createIntCast(ops[0], vTy);
    return emitCommonNeonShift(builder, loc, vTy, ops[0], ops[1], true);
  }
  case NEON::BI__builtin_neon_vshrn_n_v: {
    mlir::Location loc = getLoc(e->getExprLoc());
    cir::VectorType srcTy = builder.getExtendedOrTruncatedElementVectorType(
        vTy, true /* extended */,
        mlir::cast<cir::IntType>(vTy.getEltType()).isSigned());
    ops[0] = builder.createBitcast(ops[0], srcTy);
    ops[0] = emitCommonNeonShift(builder, loc, srcTy, ops[0], ops[1], false);
    return builder.createIntCast(ops[0], vTy);
  }
  case NEON::BI__builtin_neon_vshr_n_v:
  case NEON::BI__builtin_neon_vshrq_n_v:
    return emitNeonRShiftImm(*this, ops[0], ops[1], vTy, isUnsigned,
                             getLoc(e->getExprLoc()));
  case NEON::BI__builtin_neon_vtst_v:
  case NEON::BI__builtin_neon_vtstq_v: {
    mlir::Location loc = getLoc(e->getExprLoc());
    ops[0] = builder.createBitcast(ops[0], ty);
    ops[1] = builder.createBitcast(ops[1], ty);
    ops[0] = builder.createAnd(ops[0], ops[1]);
    // Note that during vmVM Lowering, result of `VecCmpOp` is sign extended,
    // matching traditional codegen behavior.
    return builder.create<cir::VecCmpOp>(loc, ty, cir::CmpOpKind::ne, ops[0],
                                         builder.getZero(loc, ty));
  }
  }

  // This second switch is for the intrinsics that might have a more generic
  // codegen solution so we can use the common codegen in future.
  llvm::StringRef intrincsName;
  llvm::SmallVector<mlir::Type> argTypes;
  switch (builtinID) {
  default:
    llvm::errs() << getAArch64SIMDIntrinsicString(builtinID) << " ";
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vaesmcq_u8: {
    intrincsName = "aarch64.crypto.aesmc";
    argTypes.push_back(vTy);
    break;
  }
  case NEON::BI__builtin_neon_vaeseq_u8: {
    intrincsName = "aarch64.crypto.aese";
    break;
  }
  case NEON::BI__builtin_neon_vpadd_v:
  case NEON::BI__builtin_neon_vpaddq_v: {
    intrincsName = mlir::isa<mlir::FloatType>(vTy.getEltType())
                       ? "aarch64.neon.faddp"
                       : "aarch64.neon.addp";
    break;
  }
  case NEON::BI__builtin_neon_vqadd_v:
  case NEON::BI__builtin_neon_vqaddq_v: {
    intrincsName = (intrinicId != altLLVMIntrinsic) ? "aarch64.neon.uqadd"
                                                    : "aarch64.neon.sqadd";
    break;
  }
  case NEON::BI__builtin_neon_vqdmulh_v:
  case NEON::BI__builtin_neon_vqdmulhq_v: {
    intrincsName = "aarch64.neon.sqdmulh";
    break;
  }
  case NEON::BI__builtin_neon_vqrdmulh_v:
  case NEON::BI__builtin_neon_vqrdmulhq_v: {
    intrincsName = "aarch64.neon.sqrdmulh";
    break;
  }
  case NEON::BI__builtin_neon_vqsub_v:
  case NEON::BI__builtin_neon_vqsubq_v: {
    intrincsName = (intrinicId != altLLVMIntrinsic) ? "aarch64.neon.uqsub"
                                                    : "aarch64.neon.sqsub";
    break;
  }
  case NEON::BI__builtin_neon_vrhadd_v:
  case NEON::BI__builtin_neon_vrhaddq_v: {
    intrincsName = (intrinicId != altLLVMIntrinsic) ? "aarch64.neon.urhadd"
                                                    : "aarch64.neon.srhadd";
    break;
  }
  case NEON::BI__builtin_neon_vrnd32x_f32:
  case NEON::BI__builtin_neon_vrnd32xq_f32:
  case NEON::BI__builtin_neon_vrnd32x_f64:
  case NEON::BI__builtin_neon_vrnd32xq_f64: {
    intrincsName = "aarch64.neon.frint32x";
    argTypes.push_back(vTy);
    break;
  }
  case NEON::BI__builtin_neon_vrnd64x_f32:
  case NEON::BI__builtin_neon_vrnd64xq_f32:
  case NEON::BI__builtin_neon_vrnd64x_f64:
  case NEON::BI__builtin_neon_vrnd64xq_f64: {
    intrincsName = "aarch64.neon.frint64x";
    argTypes.push_back(vTy);
    break;
  }
  case NEON::BI__builtin_neon_vrnd32z_f32:
  case NEON::BI__builtin_neon_vrnd32zq_f32:
  case NEON::BI__builtin_neon_vrnd32z_f64:
  case NEON::BI__builtin_neon_vrnd32zq_f64: {
    intrincsName = "aarch64.neon.frint32z";
    argTypes.push_back(vTy);
    break;
  }
  case NEON::BI__builtin_neon_vrnd64z_f32:
  case NEON::BI__builtin_neon_vrnd64zq_f32:
  case NEON::BI__builtin_neon_vrnd64z_f64:
  case NEON::BI__builtin_neon_vrnd64zq_f64: {
    intrincsName = "aarch64.neon.frint64z";
    argTypes.push_back(vTy);
    break;
  }
  case NEON::BI__builtin_neon_vshl_v:
  case NEON::BI__builtin_neon_vshlq_v: {
    return builder.create<cir::ShiftOp>(
        getLoc(e->getExprLoc()), vTy, builder.createBitcast(ops[0], vTy),
        builder.createBitcast(ops[1], vTy), true /* left */);
    break;
  }
  case NEON::BI__builtin_neon_vhadd_v:
  case NEON::BI__builtin_neon_vhaddq_v: {
    intrincsName = (intrinicId != altLLVMIntrinsic) ? "aarch64.neon.uhadd"
                                                    : "aarch64.neon.shadd";
    break;
  }
  case NEON::BI__builtin_neon_vhsub_v:
  case NEON::BI__builtin_neon_vhsubq_v: {
    intrincsName = (intrinicId != altLLVMIntrinsic) ? "aarch64.neon.uhsub"
                                                    : "aarch64.neon.shsub";
    break;
  }
  case NEON::BI__builtin_neon_vqmovn_v: {
    intrincsName = (intrinicId != altLLVMIntrinsic) ? "aarch64.neon.uqxtn"
                                                    : "aarch64.neon.sqxtn";
    argTypes.push_back(builder.getExtendedOrTruncatedElementVectorType(
        vTy, true /* extended */,
        mlir::cast<cir::IntType>(vTy.getEltType()).isSigned()));
    break;
  }

  case NEON::BI__builtin_neon_vqmovun_v: {
    intrincsName = "aarch64.neon.sqxtun";
    argTypes.push_back(builder.getExtendedOrTruncatedElementVectorType(
        vTy, true /* extended */, true /* signed */));
    break;
  }
  case NEON::BI__builtin_neon_vrshl_v:
  case NEON::BI__builtin_neon_vrshlq_v: {
    intrincsName = (intrinicId != altLLVMIntrinsic) ? "aarch64.neon.urshl"
                                                    : "aarch64.neon.srshl";
    break;
  }
  }

  if (intrincsName.empty())
    return nullptr;
  return emitCommonNeonCallPattern0(*this, intrincsName, argTypes, ops, vTy, e);
}

static mlir::Value emitCommonNeonSISDBuiltinExpr(
    CIRGenFunction &cgf, const ARMVectorIntrinsicInfo &info,
    llvm::SmallVectorImpl<mlir::Value> &ops, const CallExpr *expr) {
  unsigned builtinID = info.BuiltinID;
  clang::CIRGen::CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Type resultTy = cgf.convertType(expr->getType());
  mlir::Type argTy = cgf.convertType(expr->getArg(0)->getType());
  mlir::Location loc = cgf.getLoc(expr->getExprLoc());

  switch (builtinID) {
  default:
    llvm::errs() << getAArch64SIMDIntrinsicString(builtinID) << " ";
    llvm_unreachable("in emitCommonNeonSISDBuiltinExpr NYI");
  case NEON::BI__builtin_neon_vabdd_f64:
    llvm_unreachable(" neon_vabdd_f64 NYI ");
  case NEON::BI__builtin_neon_vabds_f32:
    llvm_unreachable(" neon_vabds_f32 NYI ");
  case NEON::BI__builtin_neon_vabsd_s64:
    llvm_unreachable(" neon_vabsd_s64 NYI ");
  case NEON::BI__builtin_neon_vaddlv_s32:
    llvm_unreachable(" neon_vaddlv_s32 NYI ");
  case NEON::BI__builtin_neon_vaddlv_u32:
    llvm_unreachable(" neon_vaddlv_u32 NYI ");
  case NEON::BI__builtin_neon_vaddlvq_s32:
    llvm_unreachable(" neon_vaddlvq_s32 NYI ");
  case NEON::BI__builtin_neon_vaddlvq_u32:
    return emitNeonCall(builder, {argTy}, ops, "aarch64.neon.uaddlv", resultTy,
                        loc);
  case NEON::BI__builtin_neon_vaddv_f32:
  case NEON::BI__builtin_neon_vaddvq_f32:
  case NEON::BI__builtin_neon_vaddvq_f64:
    return emitNeonCall(builder, {argTy}, ops, "aarch64.neon.faddv", resultTy,
                        loc);
  case NEON::BI__builtin_neon_vaddv_s32:
  case NEON::BI__builtin_neon_vaddvq_s32:
  case NEON::BI__builtin_neon_vaddvq_s64:
    return emitNeonCall(builder, {argTy}, ops, "aarch64.neon.saddv", resultTy,
                        loc);
  case NEON::BI__builtin_neon_vaddv_u32:
  case NEON::BI__builtin_neon_vaddvq_u32:
  case NEON::BI__builtin_neon_vaddvq_u64:
    return emitNeonCall(builder, {argTy}, ops, "aarch64.neon.uaddv", resultTy,
                        loc);
  case NEON::BI__builtin_neon_vcaged_f64:
    llvm_unreachable(" neon_vcaged_f64 NYI ");
  case NEON::BI__builtin_neon_vcages_f32:
    llvm_unreachable(" neon_vcages_f32 NYI ");
  case NEON::BI__builtin_neon_vcagtd_f64:
    llvm_unreachable(" neon_vcagtd_f64 NYI ");
  case NEON::BI__builtin_neon_vcagts_f32:
    llvm_unreachable(" neon_vcagts_f32 NYI ");
  case NEON::BI__builtin_neon_vcaled_f64:
    llvm_unreachable(" neon_vcaled_f64 NYI ");
  case NEON::BI__builtin_neon_vcales_f32:
    llvm_unreachable(" neon_vcales_f32 NYI ");
  case NEON::BI__builtin_neon_vcaltd_f64:
    llvm_unreachable(" neon_vcaltd_f64 NYI ");
  case NEON::BI__builtin_neon_vcalts_f32:
    llvm_unreachable(" neon_vcalts_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtad_s64_f64:
    llvm_unreachable(" neon_vcvtad_s64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtad_u64_f64:
    llvm_unreachable(" neon_vcvtad_u64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtas_s32_f32:
    llvm_unreachable(" neon_vcvtas_s32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtas_u32_f32:
    llvm_unreachable(" neon_vcvtas_u32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtd_n_f64_s64:
    llvm_unreachable(" neon_vcvtd_n_f64_s64 NYI ");
  case NEON::BI__builtin_neon_vcvtd_n_f64_u64:
    llvm_unreachable(" neon_vcvtd_n_f64_u64 NYI ");
  case NEON::BI__builtin_neon_vcvtd_n_s64_f64:
    llvm_unreachable(" neon_vcvtd_n_s64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtd_n_u64_f64:
    llvm_unreachable(" neon_vcvtd_n_u64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtd_s64_f64:
    llvm_unreachable(" neon_vcvtd_s64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtd_u64_f64:
    llvm_unreachable(" neon_vcvtd_u64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvth_bf16_f32:
    llvm_unreachable(" neon_vcvth_bf16_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtmd_s64_f64:
    llvm_unreachable(" neon_vcvtmd_s64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtmd_u64_f64:
    llvm_unreachable(" neon_vcvtmd_u64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtms_s32_f32:
    llvm_unreachable(" neon_vcvtms_s32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtms_u32_f32:
    llvm_unreachable(" neon_vcvtms_u32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtnd_s64_f64:
    llvm_unreachable(" neon_vcvtnd_s64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtnd_u64_f64:
    llvm_unreachable(" neon_vcvtnd_u64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtns_s32_f32:
    llvm_unreachable(" neon_vcvtns_s32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtns_u32_f32:
    llvm_unreachable(" neon_vcvtns_u32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtpd_s64_f64:
    llvm_unreachable(" neon_vcvtpd_s64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtpd_u64_f64:
    llvm_unreachable(" neon_vcvtpd_u64_f64 NYI ");
  case NEON::BI__builtin_neon_vcvtps_s32_f32:
    llvm_unreachable(" neon_vcvtps_s32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtps_u32_f32:
    llvm_unreachable(" neon_vcvtps_u32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvts_n_f32_s32:
    llvm_unreachable(" neon_vcvts_n_f32_s32 NYI ");
  case NEON::BI__builtin_neon_vcvts_n_f32_u32:
    llvm_unreachable(" neon_vcvts_n_f32_u32 NYI ");
  case NEON::BI__builtin_neon_vcvts_n_s32_f32:
    llvm_unreachable(" neon_vcvts_n_s32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvts_n_u32_f32:
    llvm_unreachable(" neon_vcvts_n_u32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvts_s32_f32:
    llvm_unreachable(" neon_vcvts_s32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvts_u32_f32:
    llvm_unreachable(" neon_vcvts_u32_f32 NYI ");
  case NEON::BI__builtin_neon_vcvtxd_f32_f64:
    llvm_unreachable(" neon_vcvtxd_f32_f64 NYI ");
  case NEON::BI__builtin_neon_vmaxnmv_f32:
    llvm_unreachable(" neon_vmaxnmv_f32 NYI ");
  case NEON::BI__builtin_neon_vmaxnmvq_f32:
    llvm_unreachable(" neon_vmaxnmvq_f32 NYI ");
  case NEON::BI__builtin_neon_vmaxnmvq_f64:
    llvm_unreachable(" neon_vmaxnmvq_f64 NYI ");
  case NEON::BI__builtin_neon_vmaxv_f32:
    llvm_unreachable(" neon_vmaxv_f32 NYI ");
  case NEON::BI__builtin_neon_vmaxv_s32:
    llvm_unreachable(" neon_vmaxv_s32 NYI ");
  case NEON::BI__builtin_neon_vmaxv_u32:
    llvm_unreachable(" neon_vmaxv_u32 NYI ");
  case NEON::BI__builtin_neon_vmaxvq_f32:
    llvm_unreachable(" neon_vmaxvq_f32 NYI ");
  case NEON::BI__builtin_neon_vmaxvq_f64:
    llvm_unreachable(" neon_vmaxvq_f64 NYI ");
  case NEON::BI__builtin_neon_vmaxvq_s32:
    llvm_unreachable(" neon_vmaxvq_s32 NYI ");
  case NEON::BI__builtin_neon_vmaxvq_u32:
    llvm_unreachable(" neon_vmaxvq_u32 NYI ");
  case NEON::BI__builtin_neon_vminnmv_f32:
    llvm_unreachable(" neon_vminnmv_f32 NYI ");
  case NEON::BI__builtin_neon_vminnmvq_f32:
    llvm_unreachable(" neon_vminnmvq_f32 NYI ");
  case NEON::BI__builtin_neon_vminnmvq_f64:
    llvm_unreachable(" neon_vminnmvq_f64 NYI ");
  case NEON::BI__builtin_neon_vminv_f32:
    llvm_unreachable(" neon_vminv_f32 NYI ");
  case NEON::BI__builtin_neon_vminv_s32:
    llvm_unreachable(" neon_vminv_s32 NYI ");
  case NEON::BI__builtin_neon_vminv_u32:
    llvm_unreachable(" neon_vminv_u32 NYI ");
  case NEON::BI__builtin_neon_vminvq_f32:
    llvm_unreachable(" neon_vminvq_f32 NYI ");
  case NEON::BI__builtin_neon_vminvq_f64:
    llvm_unreachable(" neon_vminvq_f64 NYI ");
  case NEON::BI__builtin_neon_vminvq_s32:
    llvm_unreachable(" neon_vminvq_s32 NYI ");
  case NEON::BI__builtin_neon_vminvq_u32:
    llvm_unreachable(" neon_vminvq_u32 NYI ");
  case NEON::BI__builtin_neon_vmull_p64:
    llvm_unreachable(" neon_vmull_p64 NYI ");
  case NEON::BI__builtin_neon_vmulxd_f64:
    llvm_unreachable(" neon_vmulxd_f64 NYI ");
  case NEON::BI__builtin_neon_vmulxs_f32:
    llvm_unreachable(" neon_vmulxs_f32 NYI ");
  case NEON::BI__builtin_neon_vpaddd_s64:
    llvm_unreachable(" neon_vpaddd_s64 NYI ");
  case NEON::BI__builtin_neon_vpaddd_u64:
    llvm_unreachable(" neon_vpaddd_u64 NYI ");
  case NEON::BI__builtin_neon_vpmaxnmqd_f64:
    llvm_unreachable(" neon_vpmaxnmqd_f64 NYI ");
  case NEON::BI__builtin_neon_vpmaxnms_f32:
    llvm_unreachable(" neon_vpmaxnms_f32 NYI ");
  case NEON::BI__builtin_neon_vpmaxqd_f64:
    llvm_unreachable(" neon_vpmaxqd_f64 NYI ");
  case NEON::BI__builtin_neon_vpmaxs_f32:
    llvm_unreachable(" neon_vpmaxs_f32 NYI ");
  case NEON::BI__builtin_neon_vpminnmqd_f64:
    llvm_unreachable(" neon_vpminnmqd_f64 NYI ");
  case NEON::BI__builtin_neon_vpminnms_f32:
    llvm_unreachable(" neon_vpminnms_f32 NYI ");
  case NEON::BI__builtin_neon_vpminqd_f64:
    llvm_unreachable(" neon_vpminqd_f64 NYI ");
  case NEON::BI__builtin_neon_vpmins_f32:
    llvm_unreachable(" neon_vpmins_f32 NYI ");
  case NEON::BI__builtin_neon_vqabsb_s8:
    llvm_unreachable(" neon_vqabsb_s8 NYI ");
  case NEON::BI__builtin_neon_vqabsd_s64:
    llvm_unreachable(" neon_vqabsd_s64 NYI ");
  case NEON::BI__builtin_neon_vqabsh_s16:
    llvm_unreachable(" neon_vqabsh_s16 NYI ");
  case NEON::BI__builtin_neon_vqabss_s32:
    llvm_unreachable(" neon_vqabss_s32 NYI ");
  case NEON::BI__builtin_neon_vqaddb_s8:
    llvm_unreachable(" neon_vqaddb_s8 NYI ");
  case NEON::BI__builtin_neon_vqaddb_u8:
    llvm_unreachable(" neon_vqaddb_u8 NYI ");
  case NEON::BI__builtin_neon_vqaddd_s64:
    llvm_unreachable(" neon_vqaddd_s64 NYI ");
  case NEON::BI__builtin_neon_vqaddd_u64:
    llvm_unreachable(" neon_vqaddd_u64 NYI ");
  case NEON::BI__builtin_neon_vqaddh_s16:
    llvm_unreachable(" neon_vqaddh_s16 NYI ");
  case NEON::BI__builtin_neon_vqaddh_u16:
    llvm_unreachable(" neon_vqaddh_u16 NYI ");
  case NEON::BI__builtin_neon_vqadds_s32:
    return builder.createAdd(ops[0], ops[1], false, false, true);
  case NEON::BI__builtin_neon_vqadds_u32:
    llvm_unreachable(" neon_vqadds_u32 NYI ");
  case NEON::BI__builtin_neon_vqdmulhh_s16:
    llvm_unreachable(" neon_vqdmulhh_s16 NYI ");
  case NEON::BI__builtin_neon_vqdmulhs_s32:
    llvm_unreachable(" neon_vqdmulhs_s32 NYI ");
  case NEON::BI__builtin_neon_vqdmullh_s16:
    llvm_unreachable(" neon_vqdmullh_s16 NYI ");
  case NEON::BI__builtin_neon_vqdmulls_s32:
    llvm_unreachable(" neon_vqdmulls_s32 NYI ");
  case NEON::BI__builtin_neon_vqmovnd_s64:
    llvm_unreachable(" neon_vqmovnd_s64 NYI ");
  case NEON::BI__builtin_neon_vqmovnd_u64:
    llvm_unreachable(" neon_vqmovnd_u64 NYI ");
  case NEON::BI__builtin_neon_vqmovnh_s16:
    llvm_unreachable(" neon_vqmovnh_s16 NYI ");
  case NEON::BI__builtin_neon_vqmovnh_u16:
    llvm_unreachable(" neon_vqmovnh_u16 NYI ");
  case NEON::BI__builtin_neon_vqmovns_s32: {
    mlir::Location loc = cgf.getLoc(expr->getExprLoc());
    cir::VectorType argVecTy =
        cir::VectorType::get(&(cgf.getMLIRContext()), cgf.SInt32Ty, 4);
    cir::VectorType resVecTy =
        cir::VectorType::get(&(cgf.getMLIRContext()), cgf.SInt16Ty, 4);
    vecExtendIntValue(cgf, argVecTy, ops[0], loc);
    mlir::Value result = emitNeonCall(builder, {argVecTy}, ops,
                                      "aarch64.neon.sqxtn", resVecTy, loc);
    return vecReduceIntValue(cgf, result, loc);
  }
  case NEON::BI__builtin_neon_vqmovns_u32:
    llvm_unreachable(" neon_vqmovns_u32 NYI ");
  case NEON::BI__builtin_neon_vqmovund_s64:
    llvm_unreachable(" neon_vqmovund_s64 NYI ");
  case NEON::BI__builtin_neon_vqmovunh_s16:
    llvm_unreachable(" neon_vqmovunh_s16 NYI ");
  case NEON::BI__builtin_neon_vqmovuns_s32:
    llvm_unreachable(" neon_vqmovuns_s32 NYI ");
  case NEON::BI__builtin_neon_vqnegb_s8:
    llvm_unreachable(" neon_vqnegb_s8 NYI ");
  case NEON::BI__builtin_neon_vqnegd_s64:
    llvm_unreachable(" neon_vqnegd_s64 NYI ");
  case NEON::BI__builtin_neon_vqnegh_s16:
    llvm_unreachable(" neon_vqnegh_s16 NYI ");
  case NEON::BI__builtin_neon_vqnegs_s32:
    llvm_unreachable(" neon_vqnegs_s32 NYI ");
  case NEON::BI__builtin_neon_vqrdmlahh_s16:
    llvm_unreachable(" neon_vqrdmlahh_s16 NYI ");
  case NEON::BI__builtin_neon_vqrdmlahs_s32:
    llvm_unreachable(" neon_vqrdmlahs_s32 NYI ");
  case NEON::BI__builtin_neon_vqrdmlshh_s16:
    llvm_unreachable(" neon_vqrdmlshh_s16 NYI ");
  case NEON::BI__builtin_neon_vqrdmlshs_s32:
    llvm_unreachable(" neon_vqrdmlshs_s32 NYI ");
  case NEON::BI__builtin_neon_vqrdmulhh_s16:
    llvm_unreachable(" neon_vqrdmulhh_s16 NYI ");
  case NEON::BI__builtin_neon_vqrdmulhs_s32:
    return emitNeonCall(builder, {resultTy, resultTy}, ops,
                        "aarch64.neon.sqrdmulh", resultTy, loc);
  case NEON::BI__builtin_neon_vqrshlb_s8:
    llvm_unreachable(" neon_vqrshlb_s8 NYI ");
  case NEON::BI__builtin_neon_vqrshlb_u8:
    llvm_unreachable(" neon_vqrshlb_u8 NYI ");
  case NEON::BI__builtin_neon_vqrshld_s64:
    llvm_unreachable(" neon_vqrshld_s64 NYI ");
  case NEON::BI__builtin_neon_vqrshld_u64:
    llvm_unreachable(" neon_vqrshld_u64 NYI ");
  case NEON::BI__builtin_neon_vqrshlh_s16:
    llvm_unreachable(" neon_vqrshlh_s16 NYI ");
  case NEON::BI__builtin_neon_vqrshlh_u16:
    llvm_unreachable(" neon_vqrshlh_u16 NYI ");
  case NEON::BI__builtin_neon_vqrshls_s32:
    llvm_unreachable(" neon_vqrshls_s32 NYI ");
  case NEON::BI__builtin_neon_vqrshls_u32:
    llvm_unreachable(" neon_vqrshls_u32 NYI ");
  case NEON::BI__builtin_neon_vqrshrnd_n_s64:
    llvm_unreachable(" neon_vqrshrnd_n_s64 NYI ");
  case NEON::BI__builtin_neon_vqrshrnd_n_u64:
    llvm_unreachable(" neon_vqrshrnd_n_u64 NYI ");
  case NEON::BI__builtin_neon_vqrshrnh_n_s16:
    llvm_unreachable(" neon_vqrshrnh_n_s16 NYI ");
  case NEON::BI__builtin_neon_vqrshrnh_n_u16:
    llvm_unreachable(" neon_vqrshrnh_n_u16 NYI ");
  case NEON::BI__builtin_neon_vqrshrns_n_s32:
    llvm_unreachable(" neon_vqrshrns_n_s32 NYI ");
  case NEON::BI__builtin_neon_vqrshrns_n_u32:
    llvm_unreachable(" neon_vqrshrns_n_u32 NYI ");
  case NEON::BI__builtin_neon_vqrshrund_n_s64:
    llvm_unreachable(" neon_vqrshrund_n_s64 NYI ");
  case NEON::BI__builtin_neon_vqrshrunh_n_s16:
    llvm_unreachable(" neon_vqrshrunh_n_s16 NYI ");
  case NEON::BI__builtin_neon_vqrshruns_n_s32:
    llvm_unreachable(" neon_vqrshruns_n_s32 NYI ");
  case NEON::BI__builtin_neon_vqshlb_n_s8:
    llvm_unreachable(" neon_vqshlb_n_s8 NYI ");
  case NEON::BI__builtin_neon_vqshlb_n_u8:
    llvm_unreachable(" neon_vqshlb_n_u8 NYI ");
  case NEON::BI__builtin_neon_vqshlb_s8:
    llvm_unreachable(" neon_vqshlb_s8 NYI ");
  case NEON::BI__builtin_neon_vqshlb_u8:
    llvm_unreachable(" neon_vqshlb_u8 NYI ");
  case NEON::BI__builtin_neon_vqshld_s64:
    llvm_unreachable(" neon_vqshld_s64 NYI ");
  case NEON::BI__builtin_neon_vqshld_u64:
    llvm_unreachable(" neon_vqshld_u64 NYI ");
  case NEON::BI__builtin_neon_vqshlh_n_s16:
    llvm_unreachable(" neon_vqshlh_n_s16 NYI ");
  case NEON::BI__builtin_neon_vqshlh_n_u16:
    llvm_unreachable(" neon_vqshlh_n_u16 NYI ");
  case NEON::BI__builtin_neon_vqshlh_s16:
    llvm_unreachable(" neon_vqshlh_s16 NYI ");
  case NEON::BI__builtin_neon_vqshlh_u16:
    llvm_unreachable(" neon_vqshlh_u16 NYI ");
  case NEON::BI__builtin_neon_vqshls_n_s32:
    llvm_unreachable(" neon_vqshls_n_s32 NYI ");
  case NEON::BI__builtin_neon_vqshls_n_u32:
    llvm_unreachable(" neon_vqshls_n_u32 NYI ");
  case NEON::BI__builtin_neon_vqshls_s32:
    llvm_unreachable(" neon_vqshls_s32 NYI ");
  case NEON::BI__builtin_neon_vqshls_u32:
    llvm_unreachable(" neon_vqshls_u32 NYI ");
  case NEON::BI__builtin_neon_vqshlub_n_s8:
    llvm_unreachable(" neon_vqshlub_n_s8 NYI ");
  case NEON::BI__builtin_neon_vqshluh_n_s16:
    llvm_unreachable(" neon_vqshluh_n_s16 NYI ");
  case NEON::BI__builtin_neon_vqshlus_n_s32:
    llvm_unreachable(" neon_vqshlus_n_s32 NYI ");
  case NEON::BI__builtin_neon_vqshrnd_n_s64:
    llvm_unreachable(" neon_vqshrnd_n_s64 NYI ");
  case NEON::BI__builtin_neon_vqshrnd_n_u64:
    llvm_unreachable(" neon_vqshrnd_n_u64 NYI ");
  case NEON::BI__builtin_neon_vqshrnh_n_s16:
    llvm_unreachable(" neon_vqshrnh_n_s16 NYI ");
  case NEON::BI__builtin_neon_vqshrnh_n_u16:
    llvm_unreachable(" neon_vqshrnh_n_u16 NYI ");
  case NEON::BI__builtin_neon_vqshrns_n_s32:
    llvm_unreachable(" neon_vqshrns_n_s32 NYI ");
  case NEON::BI__builtin_neon_vqshrns_n_u32:
    llvm_unreachable(" neon_vqshrns_n_u32 NYI ");
  case NEON::BI__builtin_neon_vqshrund_n_s64:
    llvm_unreachable(" neon_vqshrund_n_s64 NYI ");
  case NEON::BI__builtin_neon_vqshrunh_n_s16:
    llvm_unreachable(" neon_vqshrunh_n_s16 NYI ");
  case NEON::BI__builtin_neon_vqshruns_n_s32:
    llvm_unreachable(" neon_vqshruns_n_s32 NYI ");
  case NEON::BI__builtin_neon_vqsubb_s8:
    llvm_unreachable(" neon_vqsubb_s8 NYI ");
  case NEON::BI__builtin_neon_vqsubb_u8:
    llvm_unreachable(" neon_vqsubb_u8 NYI ");
  case NEON::BI__builtin_neon_vqsubd_s64:
    llvm_unreachable(" neon_vqsubd_s64 NYI ");
  case NEON::BI__builtin_neon_vqsubd_u64:
    llvm_unreachable(" neon_vqsubd_u64 NYI ");
  case NEON::BI__builtin_neon_vqsubh_s16:
    llvm_unreachable(" neon_vqsubh_s16 NYI ");
  case NEON::BI__builtin_neon_vqsubh_u16:
    llvm_unreachable(" neon_vqsubh_u16 NYI ");
  case NEON::BI__builtin_neon_vqsubs_s32:
    return builder.createSub(ops[0], ops[1], false, false, true);
  case NEON::BI__builtin_neon_vqsubs_u32:
    llvm_unreachable(" neon_vqsubs_u32 NYI ");
  case NEON::BI__builtin_neon_vrecped_f64:
    llvm_unreachable(" neon_vrecped_f64 NYI ");
  case NEON::BI__builtin_neon_vrecpes_f32:
    llvm_unreachable(" neon_vrecpes_f32 NYI ");
  case NEON::BI__builtin_neon_vrecpxd_f64:
    llvm_unreachable(" neon_vrecpxd_f64 NYI ");
  case NEON::BI__builtin_neon_vrecpxs_f32:
    llvm_unreachable(" neon_vrecpxs_f32 NYI ");
  case NEON::BI__builtin_neon_vrshld_s64:
    llvm_unreachable(" neon_vrshld_s64 NYI ");
  case NEON::BI__builtin_neon_vrshld_u64:
    llvm_unreachable(" neon_vrshld_u64 NYI ");
  case NEON::BI__builtin_neon_vrsqrted_f64:
    llvm_unreachable(" neon_vrsqrted_f64 NYI ");
  case NEON::BI__builtin_neon_vrsqrtes_f32:
    llvm_unreachable(" neon_vrsqrtes_f32 NYI ");
  case NEON::BI__builtin_neon_vrsqrtsd_f64:
    llvm_unreachable(" neon_vrsqrtsd_f64 NYI ");
  case NEON::BI__builtin_neon_vrsqrtss_f32:
    llvm_unreachable(" neon_vrsqrtss_f32 NYI ");
  case NEON::BI__builtin_neon_vsha1cq_u32:
    llvm_unreachable(" neon_vsha1cq_u32 NYI ");
  case NEON::BI__builtin_neon_vsha1h_u32:
    llvm_unreachable(" neon_vsha1h_u32 NYI ");
  case NEON::BI__builtin_neon_vsha1mq_u32:
    llvm_unreachable(" neon_vsha1mq_u32 NYI ");
  case NEON::BI__builtin_neon_vsha1pq_u32:
    llvm_unreachable(" neon_vsha1pq_u32 NYI ");
  case NEON::BI__builtin_neon_vshld_s64:
    llvm_unreachable(" neon_vshld_s64 NYI ");
  case NEON::BI__builtin_neon_vshld_u64:
    llvm_unreachable(" neon_vshld_u64 NYI ");
  case NEON::BI__builtin_neon_vslid_n_s64:
    llvm_unreachable(" neon_vslid_n_s64 NYI ");
  case NEON::BI__builtin_neon_vslid_n_u64:
    llvm_unreachable(" neon_vslid_n_u64 NYI ");
  case NEON::BI__builtin_neon_vsqaddb_u8:
    llvm_unreachable(" neon_vsqaddb_u8 NYI ");
  case NEON::BI__builtin_neon_vsqaddd_u64:
    llvm_unreachable(" neon_vsqaddd_u64 NYI ");
  case NEON::BI__builtin_neon_vsqaddh_u16:
    llvm_unreachable(" neon_vsqaddh_u16 NYI ");
  case NEON::BI__builtin_neon_vsqadds_u32:
    llvm_unreachable(" neon_vsqadds_u32 NYI ");
  case NEON::BI__builtin_neon_vsrid_n_s64:
    llvm_unreachable(" neon_vsrid_n_s64 NYI ");
  case NEON::BI__builtin_neon_vsrid_n_u64:
    llvm_unreachable(" neon_vsrid_n_u64 NYI ");
  case NEON::BI__builtin_neon_vuqaddb_s8:
    llvm_unreachable(" neon_vuqaddb_s8 NYI ");
  case NEON::BI__builtin_neon_vuqaddd_s64:
    llvm_unreachable(" neon_vuqaddd_s64 NYI ");
  case NEON::BI__builtin_neon_vuqaddh_s16:
    llvm_unreachable(" neon_vuqaddh_s16 NYI ");
  case NEON::BI__builtin_neon_vuqadds_s32:
    llvm_unreachable(" neon_vuqadds_s32 NYI ");
  // FP16 scalar intrinisics go here.
  case NEON::BI__builtin_neon_vabdh_f16:
    llvm_unreachable(" neon_vabdh_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtah_s32_f16:
    llvm_unreachable(" neon_vcvtah_s32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtah_s64_f16:
    llvm_unreachable(" neon_vcvtah_s64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtah_u32_f16:
    llvm_unreachable(" neon_vcvtah_u32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtah_u64_f16:
    llvm_unreachable(" neon_vcvtah_u64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_f16_s32:
    llvm_unreachable(" neon_vcvth_n_f16_s32 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_f16_s64:
    llvm_unreachable(" neon_vcvth_n_f16_s64 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_f16_u32:
    llvm_unreachable(" neon_vcvth_n_f16_u32 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_f16_u64:
    llvm_unreachable(" neon_vcvth_n_f16_u64 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_s32_f16:
    llvm_unreachable(" neon_vcvth_n_s32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_s64_f16:
    llvm_unreachable(" neon_vcvth_n_s64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_u32_f16:
    llvm_unreachable(" neon_vcvth_n_u32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_n_u64_f16:
    llvm_unreachable(" neon_vcvth_n_u64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_s32_f16:
    llvm_unreachable(" neon_vcvth_s32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_s64_f16:
    llvm_unreachable(" neon_vcvth_s64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_u32_f16:
    llvm_unreachable(" neon_vcvth_u32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvth_u64_f16:
    llvm_unreachable(" neon_vcvth_u64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtmh_s32_f16:
    llvm_unreachable(" neon_vcvtmh_s32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtmh_s64_f16:
    llvm_unreachable(" neon_vcvtmh_s64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtmh_u32_f16:
    llvm_unreachable(" neon_vcvtmh_u32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtmh_u64_f16:
    llvm_unreachable(" neon_vcvtmh_u64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtnh_s32_f16:
    llvm_unreachable(" neon_vcvtnh_s32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtnh_s64_f16:
    llvm_unreachable(" neon_vcvtnh_s64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtnh_u32_f16:
    llvm_unreachable(" neon_vcvtnh_u32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtnh_u64_f16:
    llvm_unreachable(" neon_vcvtnh_u64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtph_s32_f16:
    llvm_unreachable(" neon_vcvtph_s32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtph_s64_f16:
    llvm_unreachable(" neon_vcvtph_s64_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtph_u32_f16:
    llvm_unreachable(" neon_vcvtph_u32_f16 NYI ");
  case NEON::BI__builtin_neon_vcvtph_u64_f16:
    llvm_unreachable(" neon_vcvtph_u64_f16 NYI ");
  case NEON::BI__builtin_neon_vmulxh_f16:
    llvm_unreachable(" neon_vmulxh_f16 NYI ");
  case NEON::BI__builtin_neon_vrecpeh_f16:
    llvm_unreachable(" neon_vrecpeh_f16 NYI ");
  case NEON::BI__builtin_neon_vrecpxh_f16:
    llvm_unreachable(" neon_vrecpxh_f16 NYI ");
  case NEON::BI__builtin_neon_vrsqrteh_f16:
    llvm_unreachable(" neon_vrsqrteh_f16 NYI ");
  case NEON::BI__builtin_neon_vrsqrtsh_f16:
    llvm_unreachable(" neon_vrsqrtsh_f16 NYI ");
  }
  return nullptr;
}

mlir::Value
CIRGenFunction::emitAArch64BuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                       ReturnValueSlot ReturnValue,
                                       llvm::Triple::ArchType Arch) {
  if (BuiltinID >= clang::AArch64::FirstSVEBuiltin &&
      BuiltinID <= clang::AArch64::LastSVEBuiltin)
    return emitAArch64SVEBuiltinExpr(BuiltinID, E);

  if (BuiltinID >= clang::AArch64::FirstSMEBuiltin &&
      BuiltinID <= clang::AArch64::LastSMEBuiltin)
    return emitAArch64SMEBuiltinExpr(BuiltinID, E);

  if (BuiltinID == Builtin::BI__builtin_cpu_supports)
    llvm_unreachable("Builtin::BI__builtin_cpu_supports NYI");

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
    llvm_unreachable("clang::AArch64::BI__builtin_arm_trap NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_get_sme_state) {
    // Create call to __arm_sme_state and store the results to the two pointers.
    llvm_unreachable("clang::AArch64::BI__builtin_arm_get_sme_state NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rbit) {
    assert((getContext().getTypeSize(E->getType()) == 32) &&
           "rbit of unusual size!");
    llvm_unreachable("clang::AArch64::BI__builtin_arm_rbit NYI");
  }
  if (BuiltinID == clang::AArch64::BI__builtin_arm_rbit64) {
    assert((getContext().getTypeSize(E->getType()) == 64) &&
           "rbit of unusual size!");
    llvm_unreachable("clang::AArch64::BI__builtin_arm_rbit64 NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_clz ||
      BuiltinID == clang::AArch64::BI__builtin_arm_clz64) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_clz64 NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_cls) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_cls NYI");
  }
  if (BuiltinID == clang::AArch64::BI__builtin_arm_cls64) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_cls64 NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint32zf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint32z) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_rint32z NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint64zf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint64z) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_rint64z NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint32xf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint32x) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_rint32x NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rint64xf ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rint64x) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_rint64x NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_jcvt) {
    assert((getContext().getTypeSize(E->getType()) == 32) &&
           "__jcvt of unusual size!");
    llvm_unreachable("clang::AArch64::BI__builtin_arm_jcvt NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_ld64b ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64b ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64bv ||
      BuiltinID == clang::AArch64::BI__builtin_arm_st64bv0) {
    llvm_unreachable(" clang::AArch64::BI__builtin_arm_st64bv0 like NYI");

    if (BuiltinID == clang::AArch64::BI__builtin_arm_ld64b) {
      // Load from the address via an LLVM intrinsic, receiving a
      // tuple of 8 i64 words, and store each one to ValPtr.
      llvm_unreachable("clang::AArch64::BI__builtin_arm_ld64b NYI");
    } else {
      // Load 8 i64 words from ValPtr, and store them to the address
      // via an LLVM intrinsic.
      llvm_unreachable("clang::AArch64::BI__builtin_arm_st64b NYI");
    }
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rndr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rndrrs) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_rndrrs NYI");
  }

  if (BuiltinID == clang::AArch64::BI__clear_cache) {
    assert(E->getNumArgs() == 2 && "__clear_cache takes 2 arguments");
    llvm_unreachable("clang::AArch64::BI__clear_cache NYI");
  }

  if ((BuiltinID == clang::AArch64::BI__builtin_arm_ldrex ||
       BuiltinID == clang::AArch64::BI__builtin_arm_ldaex) &&
      getContext().getTypeSize(E->getType()) == 128) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_ldaex NYI");
  } else if (BuiltinID == clang::AArch64::BI__builtin_arm_ldrex ||
             BuiltinID == clang::AArch64::BI__builtin_arm_ldaex) {
    return emitArmLdrexNon128Intrinsic(BuiltinID, E, *this);
  }

  if ((BuiltinID == clang::AArch64::BI__builtin_arm_strex ||
       BuiltinID == clang::AArch64::BI__builtin_arm_stlex) &&
      getContext().getTypeSize(E->getArg(0)->getType()) == 128) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_stlex NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_strex ||
      BuiltinID == clang::AArch64::BI__builtin_arm_stlex) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_stlex NYI");
  }

  if (BuiltinID == clang::AArch64::BI__getReg) {
    llvm_unreachable("clang::AArch64::BI__getReg NYI");
  }

  if (BuiltinID == clang::AArch64::BI__break) {
    llvm_unreachable("clang::AArch64::BI__break NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_clrex) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_clrex NYI");
  }

  if (BuiltinID == clang::AArch64::BI_ReadWriteBarrier)
    llvm_unreachable("clang::AArch64::BI_ReadWriteBarrier");

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
    llvm_unreachable("clang::AArch64::BI__builtin_arm_crc32cd NYI");
  }

  if (CRCIntrinsicID != llvm::Intrinsic::not_intrinsic) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_crc32cd NYI");
  }

  // Memory Operations (MOPS)
  if (BuiltinID == AArch64::BI__builtin_arm_mops_memset_tag) {
    llvm_unreachable("clang::AArch64::BI__builtin_arm_crc32cd NYI");
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
    llvm_unreachable("clang::AArch64::BI__builtin_arm_subp NYI");
  }

  if (MTEIntrinsicID != llvm::Intrinsic::not_intrinsic) {
    llvm_unreachable("llvm::Intrinsic::not_intrinsic NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsr64 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_rsrp ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr64 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsr128 ||
      BuiltinID == clang::AArch64::BI__builtin_arm_wsrp) {

    llvm_unreachable("clang::AArch64::BI__builtin_arm_wsrp NYI");
    if (BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsr64 ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
        BuiltinID == clang::AArch64::BI__builtin_arm_rsrp)
      llvm_unreachable("clang::AArch64::BI__builtin_arm_rsrp NYI");

    bool IsPointerBuiltin = BuiltinID == clang::AArch64::BI__builtin_arm_rsrp ||
                            BuiltinID == clang::AArch64::BI__builtin_arm_wsrp;

    bool Is32Bit = BuiltinID == clang::AArch64::BI__builtin_arm_rsr ||
                   BuiltinID == clang::AArch64::BI__builtin_arm_wsr;

    bool Is128Bit = BuiltinID == clang::AArch64::BI__builtin_arm_rsr128 ||
                    BuiltinID == clang::AArch64::BI__builtin_arm_wsr128;

    if (Is32Bit) {
      llvm_unreachable("clang::AArch64::BI__builtin_arm_wsr NYI");
    } else if (Is128Bit) {
      llvm_unreachable("clang::AArch64::BI__builtin_arm_wsr128 NYI");
    } else if (IsPointerBuiltin) {
      llvm_unreachable("clang::AArch64::BI__builtin_arm_wsrp NYI");
    } else {
      llvm_unreachable("NYI");
    };

    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI__builtin_sponentry) {
    llvm_unreachable("clang::AArch64::BI__builtin_sponentry NYI");
  }

  if (BuiltinID == clang::AArch64::BI_ReadStatusReg ||
      BuiltinID == clang::AArch64::BI_WriteStatusReg) {
    llvm_unreachable("clang::AArch64::BI_WriteStatusReg NYI");
  }

  if (BuiltinID == clang::AArch64::BI_AddressOfReturnAddress) {
    llvm_unreachable("clang::AArch64::BI_AddressOfReturnAddress NYI");
  }

  if (BuiltinID == clang::AArch64::BI__mulh ||
      BuiltinID == clang::AArch64::BI__umulh) {
    llvm_unreachable("clang::AArch64::BI__umulh NYI");
  }

  if (BuiltinID == AArch64::BI__writex18byte ||
      BuiltinID == AArch64::BI__writex18word ||
      BuiltinID == AArch64::BI__writex18dword ||
      BuiltinID == AArch64::BI__writex18qword) {
    // Read x18 as i8*
    llvm_unreachable("AArch64::BI__writex18qword NYI");
  }

  if (BuiltinID == AArch64::BI__readx18byte ||
      BuiltinID == AArch64::BI__readx18word ||
      BuiltinID == AArch64::BI__readx18dword ||
      BuiltinID == AArch64::BI__readx18qword) {
    llvm_unreachable("AArch64::BI__readx18qword NYI");
  }

  if (BuiltinID == AArch64::BI_CopyDoubleFromInt64 ||
      BuiltinID == AArch64::BI_CopyFloatFromInt32 ||
      BuiltinID == AArch64::BI_CopyInt32FromFloat ||
      BuiltinID == AArch64::BI_CopyInt64FromDouble) {
    llvm_unreachable("AArch64::BI_CopyInt64FromDouble NYI");
  }

  if (BuiltinID == AArch64::BI_CountLeadingOnes ||
      BuiltinID == AArch64::BI_CountLeadingOnes64 ||
      BuiltinID == AArch64::BI_CountLeadingZeros ||
      BuiltinID == AArch64::BI_CountLeadingZeros64) {
    llvm_unreachable("AArch64::BI_CountLeadingZeros64 NYI");

    if (BuiltinID == AArch64::BI_CountLeadingOnes ||
        BuiltinID == AArch64::BI_CountLeadingOnes64)
      llvm_unreachable("AArch64::BI_CountLeadingOnes64 NYI");

    llvm_unreachable("BI_CountLeadingZeros64 NYI");
  }

  if (BuiltinID == AArch64::BI_CountLeadingSigns ||
      BuiltinID == AArch64::BI_CountLeadingSigns64) {
    llvm_unreachable("BI_CountLeadingSigns64 NYI");
  }

  if (BuiltinID == AArch64::BI_CountOneBits ||
      BuiltinID == AArch64::BI_CountOneBits64) {
    llvm_unreachable("AArch64::BI_CountOneBits64 NYI");
  }

  if (BuiltinID == AArch64::BI__prefetch) {
    llvm_unreachable("AArch64::BI__prefetch NYI");
  }

  if (BuiltinID == NEON::BI__builtin_neon_vcvth_bf16_f32)
    llvm_unreachable("NYI");

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  if (std::optional<CIRGenFunction::MSVCIntrin> MsvcIntId =
          translateAarch64ToMsvcIntrin(BuiltinID))
    llvm_unreachable("translateAarch64ToMsvcIntrin NYI");

  // Some intrinsics are equivalent - if they are use the base intrinsic ID.
  auto It = llvm::find_if(NEONEquivalentIntrinsicMap, [BuiltinID](auto &P) {
    return P.first == BuiltinID;
  });
  if (It != end(NEONEquivalentIntrinsicMap))
    BuiltinID = It->second;

  // Find out if any arguments are required to be integer constant
  // expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  llvm::SmallVector<mlir::Value, 4> Ops;
  Address PtrOp0 = Address::invalid();
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
        PtrOp0 = emitPointerWithAlignment(E->getArg(0));
        Ops.push_back(PtrOp0.emitRawPointer());
        continue;
      }
    }
    Ops.push_back(emitScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  auto theSISDMap = ArrayRef(AArch64SISDIntrinsicMap);
  const ARMVectorIntrinsicInfo *builtinInfo = findARMVectorIntrinsicInMap(
      theSISDMap, BuiltinID, AArch64SISDIntrinsicsProvenSorted);

  if (builtinInfo) {
    Ops.push_back(emitScalarExpr(E->getArg(E->getNumArgs() - 1)));
    mlir::Value result =
        emitCommonNeonSISDBuiltinExpr(*this, *builtinInfo, Ops, E);
    assert(result && "SISD intrinsic should have been handled");
    return result;
  }

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
  case NEON::BI__builtin_neon_vabsh_f16: {
    Ops.push_back(emitScalarExpr(E->getArg(0)));
    return builder.create<cir::FAbsOp>(getLoc(E->getExprLoc()), Ops);
  }
  case NEON::BI__builtin_neon_vaddq_p128: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddq_p128 NYI");
  }
  case NEON::BI__builtin_neon_vldrq_p128: {
    llvm_unreachable("NEON::BI__builtin_neon_vldrq_p128 NYI");
  }
  case NEON::BI__builtin_neon_vstrq_p128: {
    llvm_unreachable("NEON::BI__builtin_neon_vstrq_p128 NYI");
  }
  case NEON::BI__builtin_neon_vcvts_f32_u32:
  case NEON::BI__builtin_neon_vcvtd_f64_u64:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vcvts_f32_s32:
  case NEON::BI__builtin_neon_vcvtd_f64_s64: {
    if (usgn)
      llvm_unreachable("NEON::BI__builtin_neon_vcvtd_f64_s64 NYI");
    llvm_unreachable("NEON::BI__builtin_neon_vcvtd_f64_s64 NYI");
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
      llvm_unreachable("NEON::BI__builtin_neon_vcvth_f16_s64 NYI");
    llvm_unreachable("NEON::BI__builtin_neon_vcvth_f16_s64 NYI");
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
    llvm_unreachable("NEON::BI__builtin_neon_vcvth_s16_f16 NYI");
  }
  case NEON::BI__builtin_neon_vcaleh_f16:
  case NEON::BI__builtin_neon_vcalth_f16:
  case NEON::BI__builtin_neon_vcageh_f16:
  case NEON::BI__builtin_neon_vcagth_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vcagth_f16 NYI");
  }
  case NEON::BI__builtin_neon_vcvth_n_s16_f16:
  case NEON::BI__builtin_neon_vcvth_n_u16_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvth_n_u16_f16 NYI");
  }
  case NEON::BI__builtin_neon_vcvth_n_f16_s16:
  case NEON::BI__builtin_neon_vcvth_n_f16_u16: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvth_n_f16_u16 NYI");
  }
  case NEON::BI__builtin_neon_vpaddd_s64: {
    llvm_unreachable("NEON::BI__builtin_neon_vpaddd_s64 NYI");
  }
  case NEON::BI__builtin_neon_vpaddd_f64: {
    llvm_unreachable("NEON::BI__builtin_neon_vpaddd_f64 NYI");
  }
  case NEON::BI__builtin_neon_vpadds_f32: {
    llvm_unreachable("NEON::BI__builtin_neon_vpadds_f32 NYI");
  }
  case NEON::BI__builtin_neon_vceqzd_s64:
  case NEON::BI__builtin_neon_vceqzd_f64:
  case NEON::BI__builtin_neon_vceqzs_f32:
  case NEON::BI__builtin_neon_vceqzh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vceqzh_f16 NYI");
  case NEON::BI__builtin_neon_vcgezd_s64:
  case NEON::BI__builtin_neon_vcgezd_f64:
  case NEON::BI__builtin_neon_vcgezs_f32:
  case NEON::BI__builtin_neon_vcgezh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vcgezh_f16 NYI");
  case NEON::BI__builtin_neon_vclezd_s64:
  case NEON::BI__builtin_neon_vclezd_f64:
  case NEON::BI__builtin_neon_vclezs_f32:
  case NEON::BI__builtin_neon_vclezh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vclezh_f16 NYI");
  case NEON::BI__builtin_neon_vcgtzd_s64:
  case NEON::BI__builtin_neon_vcgtzd_f64:
  case NEON::BI__builtin_neon_vcgtzs_f32:
  case NEON::BI__builtin_neon_vcgtzh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vcgtzh_f16 NYI");
  case NEON::BI__builtin_neon_vcltzd_s64:
  case NEON::BI__builtin_neon_vcltzd_f64:
  case NEON::BI__builtin_neon_vcltzs_f32:
  case NEON::BI__builtin_neon_vcltzh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vcltzh_f16 NYI");

  case NEON::BI__builtin_neon_vceqzd_u64: {
    llvm_unreachable("NEON::BI__builtin_neon_vceqzd_u64 NYI");
  }
  case NEON::BI__builtin_neon_vceqd_f64:
  case NEON::BI__builtin_neon_vcled_f64:
  case NEON::BI__builtin_neon_vcltd_f64:
  case NEON::BI__builtin_neon_vcged_f64:
  case NEON::BI__builtin_neon_vcgtd_f64: {
    llvm_unreachable("NEON::BI__builtin_neon_vcgtd_f64 NYI");
  }
  case NEON::BI__builtin_neon_vceqs_f32:
  case NEON::BI__builtin_neon_vcles_f32:
  case NEON::BI__builtin_neon_vclts_f32:
  case NEON::BI__builtin_neon_vcges_f32:
  case NEON::BI__builtin_neon_vcgts_f32: {
    llvm_unreachable("NEON::BI__builtin_neon_vcgts_f32 NYI");
  }
  case NEON::BI__builtin_neon_vceqh_f16:
  case NEON::BI__builtin_neon_vcleh_f16:
  case NEON::BI__builtin_neon_vclth_f16:
  case NEON::BI__builtin_neon_vcgeh_f16:
  case NEON::BI__builtin_neon_vcgth_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vcgth_f16 NYI");
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
    llvm_unreachable("NEON::BI__builtin_neon_vcled_s64 NYI");
  }
  case NEON::BI__builtin_neon_vtstd_s64:
  case NEON::BI__builtin_neon_vtstd_u64: {
    llvm_unreachable("NEON::BI__builtin_neon_vtstd_u64 NYI");
  }
  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
    Ops.push_back(emitScalarExpr(E->getArg(2)));
    return builder.create<cir::VecInsertOp>(getLoc(E->getExprLoc()), Ops[1],
                                            Ops[0], Ops[2]);
  case NEON::BI__builtin_neon_vset_lane_bf16:
  case NEON::BI__builtin_neon_vsetq_lane_bf16:
    // No support for now as no real/test case for them
    // at the moment, the implementation should be the same as above
    // vset_lane or vsetq_lane intrinsics
    llvm_unreachable("NEON::BI__builtin_neon_vsetq_lane_bf16 NYI");

  case NEON::BI__builtin_neon_vset_lane_f64: {
    Ops.push_back(emitScalarExpr(E->getArg(2)));
    Ops[1] = builder.createBitcast(
        Ops[1], cir::VectorType::get(&getMLIRContext(), DoubleTy, 1));
    return builder.create<cir::VecInsertOp>(getLoc(E->getExprLoc()), Ops[1],
                                            Ops[0], Ops[2]);
  }
  case NEON::BI__builtin_neon_vsetq_lane_f64: {
    Ops.push_back(emitScalarExpr(E->getArg(2)));
    Ops[1] = builder.createBitcast(
        Ops[1], cir::VectorType::get(&getMLIRContext(), DoubleTy, 2));
    return builder.create<cir::VecInsertOp>(getLoc(E->getExprLoc()), Ops[1],
                                            Ops[0], Ops[2]);
  }
  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vdupb_lane_i8:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt8Ty, 8));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vdupb_laneq_i8:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt8Ty, 16));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vduph_lane_i16:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt16Ty, 4));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vduph_laneq_i16:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt16Ty, 8));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vdups_lane_i32:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt32Ty, 2));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vget_lane_f32:
  case NEON::BI__builtin_neon_vdups_lane_f32:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), FloatTy, 2));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vdups_laneq_i32:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt32Ty, 4));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vdupd_lane_i64:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt64Ty, 1));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vdupd_lane_f64:
  case NEON::BI__builtin_neon_vget_lane_f64:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), DoubleTy, 1));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vdupd_laneq_i64:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), UInt64Ty, 2));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vgetq_lane_f32:
  case NEON::BI__builtin_neon_vdups_laneq_f32:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), FloatTy, 4));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vgetq_lane_f64:
  case NEON::BI__builtin_neon_vdupd_laneq_f64:
    Ops[0] = builder.createBitcast(
        Ops[0], cir::VectorType::get(&getMLIRContext(), DoubleTy, 2));
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vaddh_f16: {
    Ops.push_back(emitScalarExpr(E->getArg(1)));
    return builder.createFAdd(Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vsubh_f16: {
    Ops.push_back(emitScalarExpr(E->getArg(1)));
    return builder.createFSub(Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vmulh_f16: {
    Ops.push_back(emitScalarExpr(E->getArg(1)));
    return builder.createFMul(Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vdivh_f16: {
    Ops.push_back(emitScalarExpr(E->getArg(1)));
    return builder.createFDiv(Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vfmah_f16:
    // NEON intrinsic puts accumulator first, unlike the LLVM fma.
    llvm_unreachable("NEON::BI__builtin_neon_vfmah_f16 NYI");
  case NEON::BI__builtin_neon_vfmsh_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vfmsh_f16 NYI");
  }
  case NEON::BI__builtin_neon_vaddd_s64:
  case NEON::BI__builtin_neon_vaddd_u64:
    return builder.createAdd(Ops[0], emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vsubd_s64:
  case NEON::BI__builtin_neon_vsubd_u64:
    return builder.createSub(Ops[0], emitScalarExpr(E->getArg(1)));
  case NEON::BI__builtin_neon_vqdmlalh_s16:
  case NEON::BI__builtin_neon_vqdmlslh_s16: {
    llvm_unreachable("NEON::BI__builtin_neon_vqdmlslh_s16 NYI");
  }
  case NEON::BI__builtin_neon_vqshlud_n_s64: {
    const cir::IntType IntType = builder.getSInt64Ty();
    Ops.push_back(emitScalarExpr(E->getArg(1)));
    std::optional<llvm::APSInt> APSInt =
        E->getArg(1)->getIntegerConstantExpr(getContext());
    assert(APSInt && "Expected argument to be a constant");
    Ops[1] = builder.getSInt64(APSInt->getZExtValue(), getLoc(E->getExprLoc()));
    const StringRef Intrinsic = "aarch64.neon.sqshlu";
    return emitNeonCall(builder, {IntType, IntType}, Ops, Intrinsic, IntType,
                        getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vqshld_n_u64:
  case NEON::BI__builtin_neon_vqshld_n_s64: {
    cir::IntType IntType = BuiltinID == NEON::BI__builtin_neon_vqshld_n_u64
                               ? builder.getUInt64Ty()
                               : builder.getSInt64Ty();

    const StringRef Intrinsic = BuiltinID == NEON::BI__builtin_neon_vqshld_n_u64
                                    ? "aarch64.neon.uqshl"
                                    : "aarch64.neon.sqshl";
    Ops.push_back(emitScalarExpr(E->getArg(1)));
    Ops[1] = builder.createIntCast(Ops[1], IntType);
    return emitNeonCall(builder, {IntType, IntType}, Ops, Intrinsic, IntType,
                        getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vrshrd_n_u64:
  case NEON::BI__builtin_neon_vrshrd_n_s64: {
    cir::IntType IntType = BuiltinID == NEON::BI__builtin_neon_vrshrd_n_u64
                               ? builder.getUInt64Ty()
                               : builder.getSInt64Ty();

    const StringRef Intrinsic = BuiltinID == NEON::BI__builtin_neon_vrshrd_n_u64
                                    ? "aarch64.neon.urshl"
                                    : "aarch64.neon.srshl";
    Ops.push_back(emitScalarExpr(E->getArg(1)));
    std::optional<llvm::APSInt> APSInt =
        E->getArg(1)->getIntegerConstantExpr(getContext());
    assert(APSInt && "Expected argument to be a constant");
    int64_t SV = -APSInt->getSExtValue();
    Ops[1] = builder.getSInt64(SV, getLoc(E->getExprLoc()));
    return emitNeonCall(builder, {IntType, builder.getSInt64Ty()}, Ops,
                        Intrinsic, IntType, getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vrsrad_n_u64:
  case NEON::BI__builtin_neon_vrsrad_n_s64: {
    cir::IntType IntType = BuiltinID == NEON::BI__builtin_neon_vrsrad_n_u64
                               ? builder.getUInt64Ty()
                               : builder.getSInt64Ty();
    Ops[1] = builder.createBitcast(Ops[1], IntType);
    Ops.push_back(builder.createNeg(emitScalarExpr(E->getArg(2))));

    const StringRef Intrinsic = BuiltinID == NEON::BI__builtin_neon_vrsrad_n_u64
                                    ? "aarch64.neon.urshl"
                                    : "aarch64.neon.srshl";

    llvm::SmallVector<mlir::Value, 2> args = {
        Ops[1], builder.createIntCast(Ops[2], IntType)};
    Ops[1] = emitNeonCall(builder, {IntType, IntType}, args, Intrinsic, IntType,
                          getLoc(E->getExprLoc()));
    return builder.createAdd(Ops[0], builder.createBitcast(Ops[1], IntType));
  }
  case NEON::BI__builtin_neon_vshld_n_s64:
  case NEON::BI__builtin_neon_vshld_n_u64: {
    std::optional<llvm::APSInt> amt =
        E->getArg(1)->getIntegerConstantExpr(getContext());
    assert(amt && "Expected argument to be a constant");
    return builder.createShiftLeft(Ops[0], amt->getZExtValue());
  }
  case NEON::BI__builtin_neon_vshrd_n_s64: {
    std::optional<llvm::APSInt> amt =
        E->getArg(1)->getIntegerConstantExpr(getContext());
    assert(amt && "Expected argument to be a constant");
    uint64_t bits = std::min(static_cast<uint64_t>(63), amt->getZExtValue());
    return builder.createShiftRight(Ops[0], bits);
  }
  case NEON::BI__builtin_neon_vshrd_n_u64: {
    std::optional<llvm::APSInt> amt =
        E->getArg(1)->getIntegerConstantExpr(getContext());
    assert(amt && "Expected argument to be a constant");
    uint64_t shiftAmt = amt->getZExtValue();
    if (shiftAmt == 64)
      return builder.getConstInt(getLoc(E->getExprLoc()), builder.getUInt64Ty(),
                                 0);

    return builder.createShiftRight(Ops[0], shiftAmt);
  }
  case NEON::BI__builtin_neon_vsrad_n_s64: {
    std::optional<llvm::APSInt> amt =
        E->getArg(2)->getIntegerConstantExpr(getContext());
    uint64_t shiftAmt =
        std::min(static_cast<uint64_t>(63), amt->getZExtValue());
    return builder.createAdd(Ops[0],
                             builder.createShift(Ops[1], shiftAmt, false));
  }
  case NEON::BI__builtin_neon_vsrad_n_u64: {
    std::optional<llvm::APSInt> amt =
        E->getArg(2)->getIntegerConstantExpr(getContext());
    uint64_t shiftAmt = amt->getZExtValue();
    if (shiftAmt == 64)
      return Ops[0];

    return builder.createAdd(Ops[0], builder.createShiftLeft(Ops[1], shiftAmt));
  }
  case NEON::BI__builtin_neon_vqdmlalh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlalh_laneq_s16:
  case NEON::BI__builtin_neon_vqdmlslh_lane_s16:
  case NEON::BI__builtin_neon_vqdmlslh_laneq_s16: {
    llvm_unreachable("NEON::BI__builtin_neon_vqdmlslh_laneq_s16 NYI");
  }
  case NEON::BI__builtin_neon_vqdmlals_s32:
  case NEON::BI__builtin_neon_vqdmlsls_s32: {
    llvm_unreachable("NEON::BI__builtin_neon_vqdmlsls_s32 NYI");
  }
  case NEON::BI__builtin_neon_vqdmlals_lane_s32:
  case NEON::BI__builtin_neon_vqdmlals_laneq_s32:
  case NEON::BI__builtin_neon_vqdmlsls_lane_s32:
  case NEON::BI__builtin_neon_vqdmlsls_laneq_s32: {
    llvm_unreachable("NEON::BI__builtin_neon_vqdmlsls_laneq_s32 NYI");
  }
  case NEON::BI__builtin_neon_vget_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_bf16:
  case NEON::BI__builtin_neon_vduph_lane_f16: {
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  }
  case NEON::BI__builtin_neon_vgetq_lane_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_bf16:
  case NEON::BI__builtin_neon_vduph_laneq_f16: {
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             emitScalarExpr(E->getArg(1)));
  }
  case NEON::BI__builtin_neon_vcvt_bf16_f32:
  case NEON::BI__builtin_neon_vcvtq_low_bf16_f32:
  case NEON::BI__builtin_neon_vcvtq_high_bf16_f32:
    llvm_unreachable("NYI");

  case clang::AArch64::BI_InterlockedAdd:
  case clang::AArch64::BI_InterlockedAdd64: {
    llvm_unreachable("clang::AArch64::BI_InterlockedAdd64 NYI");
  }
  }

  cir::VectorType ty = GetNeonType(this, Type);
  if (!ty)
    return nullptr;

  // Not all intrinsics handled by the common case work for AArch64 yet, so only
  // defer to common code if it's been added to our special map.
  builtinInfo = findARMVectorIntrinsicInMap(AArch64SIMDIntrinsicMap, BuiltinID,
                                            AArch64SIMDIntrinsicsProvenSorted);
  if (builtinInfo)
    return emitCommonNeonBuiltinExpr(
        builtinInfo->BuiltinID, builtinInfo->LLVMIntrinsic,
        builtinInfo->AltLLVMIntrinsic, builtinInfo->NameHint,
        builtinInfo->TypeModifier, E, Ops,
        /*never use addresses*/ Address::invalid(), Address::invalid(), Arch);

  if (mlir::Value V = emitAArch64TblBuiltinExpr(*this, BuiltinID, E, Ops, Arch))
    return V;

  cir::VectorType vTy = ty;
  llvm::SmallVector<mlir::Value, 4> args;
  switch (BuiltinID) {
  default:
    return nullptr;
  case NEON::BI__builtin_neon_vbsl_v:
  case NEON::BI__builtin_neon_vbslq_v: {
    cir::VectorType bitTy = vTy;
    if (cir::isAnyFloatingPointType(bitTy.getEltType()))
      bitTy = castVecOfFPTypeToVecOfIntWithSameWidth(builder, vTy);
    Ops[0] = builder.createBitcast(Ops[0], bitTy);
    Ops[1] = builder.createBitcast(Ops[1], bitTy);
    Ops[2] = builder.createBitcast(Ops[2], bitTy);

    Ops[1] = builder.createAnd(Ops[0], Ops[1]);
    Ops[2] = builder.createAnd(builder.createNot(Ops[0]), Ops[2]);
    Ops[0] = builder.createOr(Ops[1], Ops[2]);
    return builder.createBitcast(Ops[0], ty);
  }
  case NEON::BI__builtin_neon_vfma_lane_v:
  case NEON::BI__builtin_neon_vfmaq_lane_v: { // Only used for FP types
    // The ARM builtins (and instructions) have the addend as the first
    // operand, but the 'fma' intrinsics have it last. Swap it around here.
    llvm_unreachable("NEON::BI__builtin_neon_vfmaq_lane_v NYI");
  }
  case NEON::BI__builtin_neon_vfma_laneq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vfma_laneq_v NYI");
  }
  case NEON::BI__builtin_neon_vfmaq_laneq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vfmaq_laneq_v NYI");
  }
  case NEON::BI__builtin_neon_vfmah_lane_f16:
  case NEON::BI__builtin_neon_vfmas_lane_f32:
  case NEON::BI__builtin_neon_vfmah_laneq_f16:
  case NEON::BI__builtin_neon_vfmas_laneq_f32:
  case NEON::BI__builtin_neon_vfmad_lane_f64:
  case NEON::BI__builtin_neon_vfmad_laneq_f64: {
    llvm_unreachable("NEON::BI__builtin_neon_vfmad_laneq_f64 NYI");
  }
  case NEON::BI__builtin_neon_vmull_v: {
    llvm::StringRef name = usgn ? "aarch64.neon.umull" : "aarch64.neon.smull";
    if (Type.isPoly())
      name = "aarch64.neon.pmull";
    cir::VectorType argTy = builder.getExtendedOrTruncatedElementVectorType(
        ty, false /* truncated */, !usgn);
    return emitNeonCall(builder, {argTy, argTy}, Ops, name, ty,
                        getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vmax_v:
  case NEON::BI__builtin_neon_vmaxq_v: {
    mlir::Location loc = getLoc(E->getExprLoc());
    Ops[0] = builder.createBitcast(Ops[0], ty);
    Ops[1] = builder.createBitcast(Ops[1], ty);
    if (cir::isFPOrFPVectorTy(ty)) {
      return builder.create<cir::FMaximumOp>(loc, Ops[0], Ops[1]);
    }
    return builder.create<cir::BinOp>(loc, cir::BinOpKind::Max, Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vmaxh_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxh_f16 NYI");
  }
  case NEON::BI__builtin_neon_vmin_v:
  case NEON::BI__builtin_neon_vminq_v: {
    llvm::StringRef name = usgn ? "aarch64.neon.umin" : "aarch64.neon.smin";
    if (cir::isFPOrFPVectorTy(ty))
      name = "aarch64.neon.fmin";
    return emitNeonCall(builder, {ty, ty}, Ops, name, ty,
                        getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vminh_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminh_f16 NYI");
  }
  case NEON::BI__builtin_neon_vabd_v:
  case NEON::BI__builtin_neon_vabdq_v: {
    llvm::StringRef name = usgn ? "aarch64.neon.uabd" : "aarch64.neon.sabd";
    if (cir::isFPOrFPVectorTy(ty))
      name = "aarch64.neon.fabd";
    return emitNeonCall(builder, {ty, ty}, Ops, name, ty,
                        getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vpadal_v:
  case NEON::BI__builtin_neon_vpadalq_v: {
    cir::VectorType argTy = getHalfEltSizeTwiceNumElemsVecType(builder, vTy);
    mlir::Location loc = getLoc(E->getExprLoc());
    llvm::SmallVector<mlir::Value, 1> args = {Ops[1]};
    mlir::Value tmp = emitNeonCall(
        builder, {argTy}, args,
        usgn ? "aarch64.neon.uaddlp" : "aarch64.neon.saddlp", vTy, loc);
    mlir::Value addEnd = builder.createBitcast(Ops[0], vTy);
    return builder.createAdd(tmp, addEnd);
  }
  case NEON::BI__builtin_neon_vpmin_v:
  case NEON::BI__builtin_neon_vpminq_v:
    llvm_unreachable("NEON::BI__builtin_neon_vpminq_v NYI");
  case NEON::BI__builtin_neon_vpmax_v:
  case NEON::BI__builtin_neon_vpmaxq_v:
    llvm_unreachable("NEON::BI__builtin_neon_vpmaxq_v NYI");
  case NEON::BI__builtin_neon_vminnm_v:
  case NEON::BI__builtin_neon_vminnmq_v:
    llvm_unreachable("NEON::BI__builtin_neon_vminnmq_v NYI");
  case NEON::BI__builtin_neon_vminnmh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vminnmh_f16 NYI");
  case NEON::BI__builtin_neon_vmaxnm_v:
  case NEON::BI__builtin_neon_vmaxnmq_v:
    llvm_unreachable("NEON::BI__builtin_neon_vmaxnmq_v NYI");
  case NEON::BI__builtin_neon_vmaxnmh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vmaxnmh_f16 NYI");
  case NEON::BI__builtin_neon_vrecpss_f32: {
    llvm_unreachable("NEON::BI__builtin_neon_vrecpss_f32 NYI");
  }
  case NEON::BI__builtin_neon_vrecpsd_f64:
    llvm_unreachable("NEON::BI__builtin_neon_vrecpsd_f64 NYI");
  case NEON::BI__builtin_neon_vrecpsh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vrecpsh_f16 NYI");
  case NEON::BI__builtin_neon_vqshrun_n_v:
    llvm_unreachable("NEON::BI__builtin_neon_vqshrun_n_v NYI");
  case NEON::BI__builtin_neon_vqrshrun_n_v:
    // The prototype of builtin_neon_vqrshrun_n can be found at
    // https://developer.arm.com/architectures/instruction-sets/intrinsics/
    return emitNeonCall(
        builder,
        {builder.getExtendedOrTruncatedElementVectorType(ty, true, true),
         SInt32Ty},
        Ops, "aarch64.neon.sqrshrun", ty, getLoc(E->getExprLoc()));
  case NEON::BI__builtin_neon_vqshrn_n_v:
    return emitNeonCall(
        builder,
        {builder.getExtendedOrTruncatedElementVectorType(
             vTy, true /* extend */,
             mlir::cast<cir::IntType>(vTy.getEltType()).isSigned()),
         SInt32Ty},
        Ops, usgn ? "aarch64.neon.uqshrn" : "aarch64.neon.sqshrn", ty,
        getLoc(E->getExprLoc()));
  case NEON::BI__builtin_neon_vrshrn_n_v:
    return emitNeonCall(
        builder,
        {builder.getExtendedOrTruncatedElementVectorType(
             vTy, true /* extend */,
             mlir::cast<cir::IntType>(vTy.getEltType()).isSigned()),
         SInt32Ty},
        Ops, "aarch64.neon.rshrn", ty, getLoc(E->getExprLoc()));
  case NEON::BI__builtin_neon_vqrshrn_n_v:
    return emitNeonCall(
        builder,
        {builder.getExtendedOrTruncatedElementVectorType(
             vTy, true /* extend */,
             mlir::cast<cir::IntType>(vTy.getEltType()).isSigned()),
         SInt32Ty},
        Ops, usgn ? "aarch64.neon.uqrshrn" : "aarch64.neon.sqrshrn", ty,
        getLoc(E->getExprLoc()));
  case NEON::BI__builtin_neon_vrndah_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndah_f16 NYI");
  }
  case NEON::BI__builtin_neon_vrnda_v:
  case NEON::BI__builtin_neon_vrndaq_v: {
    assert(!cir::MissingFeatures::emitConstrainedFPCall());
    return emitNeonCall(builder, {ty}, Ops, "round", ty,
                        getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vrndih_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndih_f16 NYI");
  }
  case NEON::BI__builtin_neon_vrndmh_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndmh_f16 NYI");
  }
  case NEON::BI__builtin_neon_vrndm_v:
  case NEON::BI__builtin_neon_vrndmq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndmq_v NYI");
  }
  case NEON::BI__builtin_neon_vrndnh_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndnh_f16 NYI");
  }
  case NEON::BI__builtin_neon_vrndn_v:
  case NEON::BI__builtin_neon_vrndnq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndnq_v NYI");
  }
  case NEON::BI__builtin_neon_vrndns_f32: {
    mlir::Value arg0 = emitScalarExpr(E->getArg(0));
    args.push_back(arg0);
    return emitNeonCall(builder, {arg0.getType()}, args, "roundeven.f32",
                        getCIRGenModule().FloatTy, getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vrndph_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndph_f16 NYI");
  }
  case NEON::BI__builtin_neon_vrndp_v:
  case NEON::BI__builtin_neon_vrndpq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndpq_v NYI");
  }
  case NEON::BI__builtin_neon_vrndxh_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndxh_f16 NYI");
  }
  case NEON::BI__builtin_neon_vrndx_v:
  case NEON::BI__builtin_neon_vrndxq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndxq_v NYI");
  }
  case NEON::BI__builtin_neon_vrndh_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndh_f16 NYI");
  }
  case NEON::BI__builtin_neon_vrnd64z_f32:
  case NEON::BI__builtin_neon_vrnd64zq_f32:
  case NEON::BI__builtin_neon_vrnd64z_f64:
  case NEON::BI__builtin_neon_vrnd64zq_f64: {
    llvm_unreachable("NEON::BI__builtin_neon_vrnd64zq_f64 NYI");
  }
  case NEON::BI__builtin_neon_vrnd_v:
  case NEON::BI__builtin_neon_vrndq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vrndq_v NYI");
  }
  case NEON::BI__builtin_neon_vcvt_f64_v:
  case NEON::BI__builtin_neon_vcvtq_f64_v:
    llvm_unreachable("NEON::BI__builtin_neon_vcvtq_f64_v NYI");
  case NEON::BI__builtin_neon_vcvt_f64_f32: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvt_f64_f32 NYI");
  }
  case NEON::BI__builtin_neon_vcvt_f32_f64: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvt_f32_f64 NYI");
  }
  case NEON::BI__builtin_neon_vcvt_s32_v:
  case NEON::BI__builtin_neon_vcvt_u32_v:
  case NEON::BI__builtin_neon_vcvt_s64_v:
  case NEON::BI__builtin_neon_vcvt_u64_v:
  case NEON::BI__builtin_neon_vcvt_s16_f16:
  case NEON::BI__builtin_neon_vcvt_u16_f16:
  case NEON::BI__builtin_neon_vcvtq_s32_v:
  case NEON::BI__builtin_neon_vcvtq_u32_v:
  case NEON::BI__builtin_neon_vcvtq_s64_v:
  case NEON::BI__builtin_neon_vcvtq_u64_v:
  case NEON::BI__builtin_neon_vcvtq_s16_f16:
  case NEON::BI__builtin_neon_vcvtq_u16_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvtq_u16_f16 NYI");
  }
  case NEON::BI__builtin_neon_vcvta_s16_f16:
  case NEON::BI__builtin_neon_vcvta_u16_f16:
  case NEON::BI__builtin_neon_vcvta_s32_v:
  case NEON::BI__builtin_neon_vcvtaq_s16_f16:
  case NEON::BI__builtin_neon_vcvtaq_s32_v:
  case NEON::BI__builtin_neon_vcvta_u32_v:
  case NEON::BI__builtin_neon_vcvtaq_u16_f16:
  case NEON::BI__builtin_neon_vcvtaq_u32_v:
  case NEON::BI__builtin_neon_vcvta_s64_v:
  case NEON::BI__builtin_neon_vcvtaq_s64_v:
  case NEON::BI__builtin_neon_vcvta_u64_v:
  case NEON::BI__builtin_neon_vcvtaq_u64_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvtaq_u64_v NYI");
  }
  case NEON::BI__builtin_neon_vcvtm_s16_f16:
  case NEON::BI__builtin_neon_vcvtm_s32_v:
  case NEON::BI__builtin_neon_vcvtmq_s16_f16:
  case NEON::BI__builtin_neon_vcvtmq_s32_v:
  case NEON::BI__builtin_neon_vcvtm_u16_f16:
  case NEON::BI__builtin_neon_vcvtm_u32_v:
  case NEON::BI__builtin_neon_vcvtmq_u16_f16:
  case NEON::BI__builtin_neon_vcvtmq_u32_v:
  case NEON::BI__builtin_neon_vcvtm_s64_v:
  case NEON::BI__builtin_neon_vcvtmq_s64_v:
  case NEON::BI__builtin_neon_vcvtm_u64_v:
  case NEON::BI__builtin_neon_vcvtmq_u64_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvtmq_u64_v NYI");
  }
  case NEON::BI__builtin_neon_vcvtn_s16_f16:
  case NEON::BI__builtin_neon_vcvtn_s32_v:
  case NEON::BI__builtin_neon_vcvtnq_s16_f16:
  case NEON::BI__builtin_neon_vcvtnq_s32_v:
  case NEON::BI__builtin_neon_vcvtn_u16_f16:
  case NEON::BI__builtin_neon_vcvtn_u32_v:
  case NEON::BI__builtin_neon_vcvtnq_u16_f16:
  case NEON::BI__builtin_neon_vcvtnq_u32_v:
  case NEON::BI__builtin_neon_vcvtn_s64_v:
  case NEON::BI__builtin_neon_vcvtnq_s64_v:
  case NEON::BI__builtin_neon_vcvtn_u64_v:
  case NEON::BI__builtin_neon_vcvtnq_u64_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvtnq_u64_v NYI");
  }
  case NEON::BI__builtin_neon_vcvtp_s16_f16:
  case NEON::BI__builtin_neon_vcvtp_s32_v:
  case NEON::BI__builtin_neon_vcvtpq_s16_f16:
  case NEON::BI__builtin_neon_vcvtpq_s32_v:
  case NEON::BI__builtin_neon_vcvtp_u16_f16:
  case NEON::BI__builtin_neon_vcvtp_u32_v:
  case NEON::BI__builtin_neon_vcvtpq_u16_f16:
  case NEON::BI__builtin_neon_vcvtpq_u32_v:
  case NEON::BI__builtin_neon_vcvtp_s64_v:
  case NEON::BI__builtin_neon_vcvtpq_s64_v:
  case NEON::BI__builtin_neon_vcvtp_u64_v:
  case NEON::BI__builtin_neon_vcvtpq_u64_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vcvtpq_u64_v NYI");
  }
  case NEON::BI__builtin_neon_vmulx_v:
  case NEON::BI__builtin_neon_vmulxq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vmulxq_v NYI");
  }
  case NEON::BI__builtin_neon_vmulxh_lane_f16:
  case NEON::BI__builtin_neon_vmulxh_laneq_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmulxh_laneq_f16 NYI");
  }
  case NEON::BI__builtin_neon_vmul_lane_v:
  case NEON::BI__builtin_neon_vmul_laneq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vmul_laneq_v NYI");
  }
  case NEON::BI__builtin_neon_vnegd_s64:
    llvm_unreachable("NEON::BI__builtin_neon_vnegd_s64 NYI");
  case NEON::BI__builtin_neon_vnegh_f16:
    llvm_unreachable("NEON::BI__builtin_neon_vnegh_f16 NYI");
  case NEON::BI__builtin_neon_vpmaxnm_v:
  case NEON::BI__builtin_neon_vpmaxnmq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vpmaxnmq_v NYI");
  }
  case NEON::BI__builtin_neon_vpminnm_v:
  case NEON::BI__builtin_neon_vpminnmq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vpminnmq_v NYI");
  }
  case NEON::BI__builtin_neon_vsqrth_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vsqrth_f16 NYI");
  }
  case NEON::BI__builtin_neon_vsqrt_v:
  case NEON::BI__builtin_neon_vsqrtq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vsqrtq_v NYI");
  }
  case NEON::BI__builtin_neon_vrbit_v:
  case NEON::BI__builtin_neon_vrbitq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vrbitq_v NYI");
  }
  case NEON::BI__builtin_neon_vaddv_u8:
    // FIXME: These are handled by the AArch64 scalar code.
    llvm_unreachable("NEON::BI__builtin_neon_vaddv_u8 NYI");
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddv_s8: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddv_s8 NYI");
  }
  case NEON::BI__builtin_neon_vaddv_u16:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddv_s16: {
    cir::IntType eltTy = usgn ? UInt16Ty : SInt16Ty;
    cir::VectorType vTy = cir::VectorType::get(builder.getContext(), eltTy, 4);
    Ops.push_back(emitScalarExpr(E->getArg(0)));
    // This is to add across the vector elements, so wider result type needed.
    Ops[0] = emitNeonCall(builder, {vTy}, Ops,
                          usgn ? "aarch64.neon.uaddv" : "aarch64.neon.saddv",
                          SInt32Ty, getLoc(E->getExprLoc()));
    return builder.createIntCast(Ops[0], eltTy);
  }
  case NEON::BI__builtin_neon_vaddvq_u8:
    llvm_unreachable("NEON::BI__builtin_neon_vaddvq_u8 NYI");
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddvq_s8: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddvq_s8 NYI");
  }
  case NEON::BI__builtin_neon_vaddvq_u16:
    llvm_unreachable("NEON::BI__builtin_neon_vaddvq_u16 NYI");
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddvq_s16: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddvq_s16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_u8: {
    return emitCommonNeonVecAcrossCall(*this, "aarch64.neon.umaxv", UInt8Ty, 8,
                                       E);
  }
  case NEON::BI__builtin_neon_vmaxv_u16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxv_u16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_u8: {
    return emitCommonNeonVecAcrossCall(*this, "aarch64.neon.umaxv", UInt8Ty, 16,
                                       E);
  }
  case NEON::BI__builtin_neon_vmaxvq_u16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxvq_u16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_s8: {
    return emitCommonNeonVecAcrossCall(*this, "aarch64.neon.smaxv", SInt8Ty, 8,
                                       E);
  }
  case NEON::BI__builtin_neon_vmaxv_s16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxv_s16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_s8: {
    return emitCommonNeonVecAcrossCall(*this, "aarch64.neon.smaxv", SInt8Ty, 16,
                                       E);
  }
  case NEON::BI__builtin_neon_vmaxvq_s16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxvq_s16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxv_f16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxvq_f16 NYI");
  }
  case NEON::BI__builtin_neon_vminv_u8: {
    llvm_unreachable("NEON::BI__builtin_neon_vminv_u8 NYI");
  }
  case NEON::BI__builtin_neon_vminv_u16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminv_u16 NYI");
  }
  case NEON::BI__builtin_neon_vminvq_u8: {
    llvm_unreachable("NEON::BI__builtin_neon_vminvq_u8 NYI");
  }
  case NEON::BI__builtin_neon_vminvq_u16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminvq_u16 NYI");
  }
  case NEON::BI__builtin_neon_vminv_s8: {
    llvm_unreachable("NEON::BI__builtin_neon_vminv_s8 NYI");
  }
  case NEON::BI__builtin_neon_vminv_s16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminv_s16 NYI");
  }
  case NEON::BI__builtin_neon_vminvq_s8: {
    llvm_unreachable("NEON::BI__builtin_neon_vminvq_s8 NYI");
  }
  case NEON::BI__builtin_neon_vminvq_s16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminvq_s16 NYI");
  }
  case NEON::BI__builtin_neon_vminv_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminv_f16 NYI");
  }
  case NEON::BI__builtin_neon_vminvq_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminvq_f16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxnmv_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxnmv_f16 NYI");
  }
  case NEON::BI__builtin_neon_vmaxnmvq_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vmaxnmvq_f16 NYI");
  }
  case NEON::BI__builtin_neon_vminnmv_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminnmv_f16 NYI");
  }
  case NEON::BI__builtin_neon_vminnmvq_f16: {
    llvm_unreachable("NEON::BI__builtin_neon_vminnmvq_f16 NYI");
  }
  case NEON::BI__builtin_neon_vmul_n_f64: {
    llvm_unreachable("NEON::BI__builtin_neon_vmul_n_f64 NYI");
  }
  case NEON::BI__builtin_neon_vaddlv_u8: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddlv_u8 NYI");
  }
  case NEON::BI__builtin_neon_vaddlvq_u8: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddlvq_u8 NYI");
  }
  case NEON::BI__builtin_neon_vaddlvq_u16:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddlvq_s16: {
    mlir::Type argTy = cir::VectorType::get(builder.getContext(),
                                            usgn ? UInt16Ty : SInt16Ty, 8);
    llvm::SmallVector<mlir::Value, 1> argOps = {emitScalarExpr(E->getArg(0))};
    return emitNeonCall(builder, {argTy}, argOps,
                        usgn ? "aarch64.neon.uaddlv" : "aarch64.neon.saddlv",
                        usgn ? UInt32Ty : SInt32Ty, getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vaddlv_s8: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddlv_s8 NYI");
  }
  case NEON::BI__builtin_neon_vaddlv_u16:
    usgn = true;
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddlv_s16: {
    mlir::Type argTy = cir::VectorType::get(builder.getContext(),
                                            usgn ? UInt16Ty : SInt16Ty, 4);
    llvm::SmallVector<mlir::Value, 1> argOps = {emitScalarExpr(E->getArg(0))};
    return emitNeonCall(builder, {argTy}, argOps,
                        usgn ? "aarch64.neon.uaddlv" : "aarch64.neon.saddlv",
                        usgn ? UInt32Ty : SInt32Ty, getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vaddlvq_s8: {
    llvm_unreachable("NEON::BI__builtin_neon_vaddlvq_s8 NYI");
  }
  case NEON::BI__builtin_neon_vsri_n_v:
  case NEON::BI__builtin_neon_vsriq_n_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vsriq_n_v NYI");
  }
  case NEON::BI__builtin_neon_vsli_n_v:
  case NEON::BI__builtin_neon_vsliq_n_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vsliq_n_v NYI");
  }
  case NEON::BI__builtin_neon_vsra_n_v:
  case NEON::BI__builtin_neon_vsraq_n_v: {
    Ops[0] = builder.createBitcast(Ops[0], vTy);
    Ops[1] = emitNeonRShiftImm(*this, Ops[1], Ops[2], vTy, usgn,
                               getLoc(E->getExprLoc()));
    return builder.createAdd(Ops[0], Ops[1]);
  }
  case NEON::BI__builtin_neon_vrsra_n_v:
  case NEON::BI__builtin_neon_vrsraq_n_v: {
    llvm::SmallVector<mlir::Value> tmpOps = {Ops[1], Ops[2]};
    // The llvm intrinsic is expecting negative shift amount for right shift.
    // Thus we have to make shift amount vec type to be signed.
    cir::VectorType shitAmtVecTy =
        usgn ? getSignChangedVectorType(builder, vTy) : vTy;
    mlir::Value tmp =
        emitNeonCall(builder, {vTy, shitAmtVecTy}, tmpOps,
                     usgn ? "aarch64.neon.urshl" : "aarch64.neon.srshl", vTy,
                     getLoc(E->getExprLoc()), false,
                     1 /* shift amount is args[1]*/, true /* right shift */);
    Ops[0] = builder.createBitcast(Ops[0], vTy);
    return builder.createBinop(Ops[0], cir::BinOpKind::Add, tmp);
  }
  case NEON::BI__builtin_neon_vld1_v:
  case NEON::BI__builtin_neon_vld1q_v: {
    return builder.createAlignedLoad(Ops[0].getLoc(), vTy, Ops[0],
                                     PtrOp0.getAlignment());
  }
  case NEON::BI__builtin_neon_vst1_v:
  case NEON::BI__builtin_neon_vst1q_v: {
    Ops[1] = builder.createBitcast(Ops[1], vTy);
    (void)builder.createAlignedStore(Ops[1].getLoc(), Ops[1], Ops[0],
                                     PtrOp0.getAlignment());
    return Ops[1];
  }
  case NEON::BI__builtin_neon_vld1_lane_v:
  case NEON::BI__builtin_neon_vld1q_lane_v: {
    Ops[1] = builder.createBitcast(Ops[1], vTy);
    Ops[0] = builder.createAlignedLoad(Ops[0].getLoc(), vTy.getEltType(),
                                       Ops[0], PtrOp0.getAlignment());
    return builder.create<cir::VecInsertOp>(getLoc(E->getExprLoc()), Ops[1],
                                            Ops[0], Ops[2]);
  }
  case NEON::BI__builtin_neon_vldap1_lane_s64:
  case NEON::BI__builtin_neon_vldap1q_lane_s64: {
    cir::LoadOp Load = builder.createAlignedLoad(
        Ops[0].getLoc(), vTy.getEltType(), Ops[0], PtrOp0.getAlignment());
    Load.setAtomic(cir::MemOrder::Acquire);
    return builder.create<cir::VecInsertOp>(getLoc(E->getExprLoc()),
                                            builder.createBitcast(Ops[1], vTy),
                                            Load, Ops[2]);
  }
  case NEON::BI__builtin_neon_vld1_dup_v:
  case NEON::BI__builtin_neon_vld1q_dup_v: {
    Address ptrAddr = PtrOp0.withElementType(builder, vTy.getEltType());
    mlir::Value val = builder.createLoad(getLoc(E->getExprLoc()), ptrAddr);
    cir::VecSplatOp vecSplat =
        builder.create<cir::VecSplatOp>(getLoc(E->getExprLoc()), vTy, val);
    return vecSplat;
  }
  case NEON::BI__builtin_neon_vst1_lane_v:
  case NEON::BI__builtin_neon_vst1q_lane_v: {
    Ops[1] = builder.createBitcast(Ops[1], ty);
    Ops[1] = builder.create<cir::VecExtractOp>(Ops[1].getLoc(), Ops[1], Ops[2]);
    (void)builder.createAlignedStore(getLoc(E->getExprLoc()), Ops[1], Ops[0],
                                     PtrOp0.getAlignment());
    return Ops[1];
  }
  case NEON::BI__builtin_neon_vstl1_lane_s64:
  case NEON::BI__builtin_neon_vstl1q_lane_s64: {
    Ops[1] = builder.createBitcast(Ops[1], ty);
    Ops[1] = builder.create<cir::VecExtractOp>(Ops[1].getLoc(), Ops[1], Ops[2]);
    cir::StoreOp Store = builder.createAlignedStore(
        getLoc(E->getExprLoc()), Ops[1], Ops[0], PtrOp0.getAlignment());
    Store.setAtomic(cir::MemOrder::Release);
    return Ops[1];
  }
  case NEON::BI__builtin_neon_vld2_v:
  case NEON::BI__builtin_neon_vld2q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld2q_v NYI");
  }
  case NEON::BI__builtin_neon_vld3_v:
  case NEON::BI__builtin_neon_vld3q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld3q_v NYI");
  }
  case NEON::BI__builtin_neon_vld4_v:
  case NEON::BI__builtin_neon_vld4q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld4q_v NYI");
  }
  case NEON::BI__builtin_neon_vld2_dup_v:
  case NEON::BI__builtin_neon_vld2q_dup_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld2q_dup_v NYI");
  }
  case NEON::BI__builtin_neon_vld3_dup_v:
  case NEON::BI__builtin_neon_vld3q_dup_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld3q_dup_v NYI");
  }
  case NEON::BI__builtin_neon_vld4_dup_v:
  case NEON::BI__builtin_neon_vld4q_dup_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld4q_dup_v NYI");
  }
  case NEON::BI__builtin_neon_vld2_lane_v:
  case NEON::BI__builtin_neon_vld2q_lane_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld2q_lane_v NYI");
  }
  case NEON::BI__builtin_neon_vld3_lane_v:
  case NEON::BI__builtin_neon_vld3q_lane_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld3q_lane_v NYI");
  }
  case NEON::BI__builtin_neon_vld4_lane_v:
  case NEON::BI__builtin_neon_vld4q_lane_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vld4q_lane_v NYI");
  }
  case NEON::BI__builtin_neon_vst2_v:
  case NEON::BI__builtin_neon_vst2q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vst2q_v NYI");
  }
  case NEON::BI__builtin_neon_vst2_lane_v:
  case NEON::BI__builtin_neon_vst2q_lane_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vst2q_lane_v NYI");
  }
  case NEON::BI__builtin_neon_vst3_v:
  case NEON::BI__builtin_neon_vst3q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vst3q_v NYI");
  }
  case NEON::BI__builtin_neon_vst3_lane_v:
  case NEON::BI__builtin_neon_vst3q_lane_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vst3q_lane_v NYI");
  }
  case NEON::BI__builtin_neon_vst4_v:
  case NEON::BI__builtin_neon_vst4q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vst4q_v NYI");
  }
  case NEON::BI__builtin_neon_vst4_lane_v:
  case NEON::BI__builtin_neon_vst4q_lane_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vst4q_lane_v NYI");
  }
  case NEON::BI__builtin_neon_vtrn_v:
  case NEON::BI__builtin_neon_vtrnq_v: {
    // This set of neon intrinsics implement SIMD matrix transpose.
    // The matrix transposed is always 2x2, and these intrincis transpose
    // multiple 2x2 matrices in parallel, that is why result type is
    // always 2-D matrix whose last dimension is 2.
    // For example `vtrn_s16` would have:
    //  input 1: {0, 1, 2, 3}
    //  input 2; {4, 5, 6, 7}
    //  This basically represents two 2x2 matrices:
    //  [ 0, 1 ]  and  [ 2, 3]
    //  [ 4, 5 ]       [ 6, 7]
    //  They should be simultaneously and independently transposed.
    //  Thus, result is :
    //   { {0, 4, 2, 6},
    //     {1, 5, 3, 7 } }
    Ops[1] = builder.createBitcast(Ops[1], ty);
    Ops[2] = builder.createBitcast(Ops[2], ty);
    // Adding a bitcast here as Ops[0] might be a void pointer.
    mlir::Value baseAddr =
        builder.createBitcast(Ops[0], builder.getPointerTo(ty));
    mlir::Value sv;
    mlir::Location loc = getLoc(E->getExprLoc());

    for (unsigned vi = 0; vi != 2; ++vi) {
      llvm::SmallVector<int64_t, 16> indices;
      for (unsigned i = 0, e = vTy.getSize(); i != e; i += 2) {
        indices.push_back(i + vi);
        indices.push_back(i + e + vi);
      }
      cir::ConstantOp idx = builder.getConstInt(loc, SInt32Ty, vi);
      mlir::Value addr = builder.create<cir::PtrStrideOp>(
          loc, baseAddr.getType(), baseAddr, idx);
      sv = builder.createVecShuffle(loc, Ops[1], Ops[2], indices);
      (void)builder.CIRBaseBuilderTy::createStore(loc, sv, addr);
    }
    return sv;
  }
  case NEON::BI__builtin_neon_vuzp_v:
  case NEON::BI__builtin_neon_vuzpq_v: {
    Ops[1] = builder.createBitcast(Ops[1], ty);
    Ops[2] = builder.createBitcast(Ops[2], ty);
    // Adding a bitcast here as Ops[0] might be a void pointer.
    mlir::Value baseAddr =
        builder.createBitcast(Ops[0], builder.getPointerTo(ty));
    mlir::Value sv;
    mlir::Location loc = getLoc(E->getExprLoc());

    for (unsigned vi = 0; vi != 2; ++vi) {
      llvm::SmallVector<int64_t, 16> indices;
      for (unsigned i = 0, e = vTy.getSize(); i != e; ++i) {
        indices.push_back(2 * i + vi);
      }
      cir::ConstantOp idx = builder.getConstInt(loc, SInt32Ty, vi);
      mlir::Value addr = builder.create<cir::PtrStrideOp>(
          loc, baseAddr.getType(), baseAddr, idx);
      sv = builder.createVecShuffle(loc, Ops[1], Ops[2], indices);
      (void)builder.CIRBaseBuilderTy::createStore(loc, sv, addr);
    }
    return sv;
  }
  case NEON::BI__builtin_neon_vzip_v:
  case NEON::BI__builtin_neon_vzipq_v: {
    Ops[1] = builder.createBitcast(Ops[1], ty);
    Ops[2] = builder.createBitcast(Ops[2], ty);
    // Adding a bitcast here as Ops[0] might be a void pointer.
    mlir::Value baseAddr =
        builder.createBitcast(Ops[0], builder.getPointerTo(ty));
    mlir::Value sv;
    mlir::Location loc = getLoc(E->getExprLoc());

    for (unsigned vi = 0; vi != 2; ++vi) {
      llvm::SmallVector<int64_t, 16> indices;
      for (unsigned i = 0, e = vTy.getSize(); i != e; i += 2) {
        indices.push_back((i + vi * e) >> 1);
        indices.push_back(((i + vi * e) >> 1) + e);
      }
      cir::ConstantOp idx = builder.getConstInt(loc, SInt32Ty, vi);
      mlir::Value addr = builder.create<cir::PtrStrideOp>(
          loc, baseAddr.getType(), baseAddr, idx);
      sv = builder.createVecShuffle(loc, Ops[1], Ops[2], indices);
      (void)builder.CIRBaseBuilderTy::createStore(loc, sv, addr);
    }
    return sv;
  }
  case NEON::BI__builtin_neon_vqtbl1q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbl1q_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbl2q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbl2q_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbl3q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbl3q_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbl4q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbl4q_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbx1q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbx1q_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbx2q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbx2q_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbx3q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbx3q_v NYI");
  }
  case NEON::BI__builtin_neon_vqtbx4q_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vqtbx4q_v NYI");
  }
  case NEON::BI__builtin_neon_vsqadd_v:
  case NEON::BI__builtin_neon_vsqaddq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vsqaddq_v NYI");
  }
  case NEON::BI__builtin_neon_vuqadd_v:
  case NEON::BI__builtin_neon_vuqaddq_v: {
    llvm_unreachable("NEON::BI__builtin_neon_vuqaddq_v NYI");
  }
  }
}
