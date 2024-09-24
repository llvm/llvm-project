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

using namespace cir;
using namespace clang;
using namespace mlir::cir;
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
    NEONMAP1(__a64_vcvtq_low_bf16_f32, aarch64_neon_bfcvtn, 0),
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
    NEONMAP1(vcvtq_high_bf16_f32, aarch64_neon_bfcvtn2, 0),
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
    NEONMAP1(vcvth_bf16_f32, aarch64_neon_bfcvt, 0),
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

// Many of MSVC builtins are on x64, ARM and AArch64; to avoid repeating code,
// we handle them here.
enum class CIRGenFunction::MSVCIntrin {
  _BitScanForward,
  _BitScanReverse,
  _InterlockedAnd,
  _InterlockedDecrement,
  _InterlockedExchange,
  _InterlockedExchangeAdd,
  _InterlockedExchangeSub,
  _InterlockedIncrement,
  _InterlockedOr,
  _InterlockedXor,
  _InterlockedExchangeAdd_acq,
  _InterlockedExchangeAdd_rel,
  _InterlockedExchangeAdd_nf,
  _InterlockedExchange_acq,
  _InterlockedExchange_rel,
  _InterlockedExchange_nf,
  _InterlockedCompareExchange_acq,
  _InterlockedCompareExchange_rel,
  _InterlockedCompareExchange_nf,
  _InterlockedCompareExchange128,
  _InterlockedCompareExchange128_acq,
  _InterlockedCompareExchange128_rel,
  _InterlockedCompareExchange128_nf,
  _InterlockedOr_acq,
  _InterlockedOr_rel,
  _InterlockedOr_nf,
  _InterlockedXor_acq,
  _InterlockedXor_rel,
  _InterlockedXor_nf,
  _InterlockedAnd_acq,
  _InterlockedAnd_rel,
  _InterlockedAnd_nf,
  _InterlockedIncrement_acq,
  _InterlockedIncrement_rel,
  _InterlockedIncrement_nf,
  _InterlockedDecrement_acq,
  _InterlockedDecrement_rel,
  _InterlockedDecrement_nf,
  __fastfail,
};

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

static mlir::Type GetNeonType(CIRGenFunction *CGF, NeonTypeFlags TypeFlags,
                              bool HasLegalHalfType = true, bool V1Ty = false,
                              bool AllowBFloatArgsAndRet = true) {
  int IsQuad = TypeFlags.isQuad();
  switch (TypeFlags.getEltType()) {
  case NeonTypeFlags::Int8:
  case NeonTypeFlags::Poly8:
    return mlir::cir::VectorType::get(CGF->getBuilder().getContext(),
                                      TypeFlags.isUnsigned() ? CGF->UInt8Ty
                                                             : CGF->SInt8Ty,
                                      V1Ty ? 1 : (8 << IsQuad));
  case NeonTypeFlags::Int16:
  case NeonTypeFlags::Poly16:
    return mlir::cir::VectorType::get(CGF->getBuilder().getContext(),
                                      TypeFlags.isUnsigned() ? CGF->UInt16Ty
                                                             : CGF->SInt16Ty,
                                      V1Ty ? 1 : (4 << IsQuad));
  case NeonTypeFlags::BFloat16:
    if (AllowBFloatArgsAndRet)
      llvm_unreachable("NYI");
    else
      llvm_unreachable("NYI");
  case NeonTypeFlags::Float16:
    if (HasLegalHalfType)
      llvm_unreachable("NYI");
    else
      llvm_unreachable("NYI");
  case NeonTypeFlags::Int32:
    return mlir::cir::VectorType::get(CGF->getBuilder().getContext(),
                                      TypeFlags.isUnsigned() ? CGF->UInt32Ty
                                                             : CGF->SInt32Ty,
                                      V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Int64:
  case NeonTypeFlags::Poly64:
    return mlir::cir::VectorType::get(CGF->getBuilder().getContext(),
                                      TypeFlags.isUnsigned() ? CGF->UInt64Ty
                                                             : CGF->SInt64Ty,
                                      V1Ty ? 1 : (1 << IsQuad));
  case NeonTypeFlags::Poly128:
    // FIXME: i128 and f128 doesn't get fully support in Clang and llvm.
    // There is a lot of i128 and f128 API missing.
    // so we use v16i8 to represent poly128 and get pattern matched.
    llvm_unreachable("NYI");
  case NeonTypeFlags::Float32:
    return mlir::cir::VectorType::get(CGF->getBuilder().getContext(),
                                      CGF->getCIRGenModule().FloatTy,
                                      V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Float64:
    llvm_unreachable("NYI");
  }
  llvm_unreachable("Unknown vector element type!");
}

static mlir::Value buildAArch64TblBuiltinExpr(CIRGenFunction &CGF,
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
  auto Ty = GetNeonType(&CGF, Type);
  if (!Ty)
    return nullptr;

  // AArch64 scalar builtins are not overloaded, they do not have an extra
  // argument that specifies the vector type, need to handle each case.
  switch (BuiltinID) {
  case NEON::BI__builtin_neon_vtbl1_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtbl2_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtbl3_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtbl4_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtbx1_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtbx2_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtbx3_v: {
    llvm_unreachable("NYI");
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

mlir::Value CIRGenFunction::buildAArch64SMEBuiltinExpr(unsigned BuiltinID,
                                                       const CallExpr *E) {
  auto *Builtin = findARMVectorIntrinsicInMap(AArch64SMEIntrinsicMap, BuiltinID,
                                              AArch64SMEIntrinsicsProvenSorted);
  (void)Builtin;
  llvm_unreachable("NYI");
}

mlir::Value CIRGenFunction::buildAArch64SVEBuiltinExpr(unsigned BuiltinID,
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

mlir::Value CIRGenFunction::buildScalarOrConstFoldImmArg(unsigned ICEArguments,
                                                         unsigned Idx,
                                                         const CallExpr *E) {
  mlir::Value Arg = {};
  if ((ICEArguments & (1 << Idx)) == 0) {
    Arg = buildScalarExpr(E->getArg(Idx));
  } else {
    // If this is required to be a constant, constant fold it so that we
    // know that the generated intrinsic gets a ConstantInt.
    std::optional<llvm::APSInt> Result =
        E->getArg(Idx)->getIntegerConstantExpr(getContext());
    assert(Result && "Expected argument to be a constant");
    Arg = builder.getConstInt(getLoc(E->getSourceRange()), *Result);
  }
  return Arg;
}

static mlir::Value buildArmLdrexNon128Intrinsic(unsigned int builtinID,
                                                const CallExpr *clangCallExpr,
                                                CIRGenFunction &cgf) {
  StringRef intrinsicName;
  if (builtinID == clang::AArch64::BI__builtin_arm_ldrex) {
    intrinsicName = "llvm.aarch64.ldxr";
  } else {
    llvm_unreachable("Unknown builtinID");
  }
  // Argument
  mlir::Value loadAddr = cgf.buildScalarExpr(clangCallExpr->getArg(0));
  // Get Instrinc call
  CIRGenBuilderTy &builder = cgf.getBuilder();
  QualType clangResTy = clangCallExpr->getType();
  mlir::Type realResTy = cgf.ConvertType(clangResTy);
  // Return type of LLVM intrinsic is defined in Intrinsic<arch_type>.td,
  // which can be found under LLVM IR directory.
  mlir::Type funcResTy = builder.getSInt64Ty();
  mlir::Location loc = cgf.getLoc(clangCallExpr->getExprLoc());
  mlir::cir::IntrinsicCallOp op = builder.create<mlir::cir::IntrinsicCallOp>(
      loc, builder.getStringAttr(intrinsicName), funcResTy, loadAddr);
  mlir::Value res = op.getResult();

  // Convert result type to the expected type.
  if (mlir::isa<mlir::cir::PointerType>(realResTy)) {
    return builder.createIntToPtr(res, realResTy);
  }
  mlir::cir::IntType intResTy =
      builder.getSIntNTy(cgf.CGM.getDataLayout().getTypeSizeInBits(realResTy));
  mlir::Value intCastRes = builder.createIntCast(res, intResTy);
  if (mlir::isa<mlir::cir::IntType>(realResTy)) {
    return builder.createIntCast(intCastRes, realResTy);
  } else {
    // Above cases should cover most situations and we have test coverage.
    llvm_unreachable("Unsupported return type for now");
  }
}

mlir::Value buildNeonCall(unsigned int builtinID, CIRGenFunction &cgf,
                          llvm::SmallVector<mlir::Type> argTypes,
                          llvm::SmallVector<mlir::Value, 4> args,
                          llvm::StringRef intrinsicName, mlir::Type funcResTy,
                          mlir::Location loc,
                          bool isConstrainedFPIntrinsic = false,
                          unsigned shift = 0, bool rightshift = false) {
  // TODO: Consider removing the following unreachable when we have
  // buildConstrainedFPCall feature implemented
  assert(!MissingFeatures::buildConstrainedFPCall());
  if (isConstrainedFPIntrinsic)
    llvm_unreachable("isConstrainedFPIntrinsic NYI");
  // TODO: Remove the following unreachable and call it in the loop once
  // there is an implementation of buildNeonShiftVector
  if (shift > 0)
    llvm_unreachable("Argument shift NYI");

  CIRGenBuilderTy &builder = cgf.getBuilder();
  for (unsigned j = 0; j < argTypes.size(); ++j) {
    if (isConstrainedFPIntrinsic) {
      assert(!MissingFeatures::buildConstrainedFPCall());
    }
    if (shift > 0 && shift == j) {
      assert(!MissingFeatures::buildNeonShiftVector());
    } else {
      args[j] = builder.createBitcast(args[j], argTypes[j]);
    }
  }
  if (isConstrainedFPIntrinsic) {
    assert(!MissingFeatures::buildConstrainedFPCall());
    return nullptr;
  } else {
    return builder
        .create<mlir::cir::IntrinsicCallOp>(
            loc, builder.getStringAttr(intrinsicName), funcResTy, args)
        .getResult();
  }
}

mlir::Value
CIRGenFunction::buildAArch64BuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                        ReturnValueSlot ReturnValue,
                                        llvm::Triple::ArchType Arch) {
  if (BuiltinID >= clang::AArch64::FirstSVEBuiltin &&
      BuiltinID <= clang::AArch64::LastSVEBuiltin)
    return buildAArch64SVEBuiltinExpr(BuiltinID, E);

  if (BuiltinID >= clang::AArch64::FirstSMEBuiltin &&
      BuiltinID <= clang::AArch64::LastSMEBuiltin)
    return buildAArch64SMEBuiltinExpr(BuiltinID, E);

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
    return buildArmLdrexNon128Intrinsic(BuiltinID, E, *this);
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

  if (BuiltinID == clang::AArch64::BI__builtin_sponentry) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI_ReadStatusReg ||
      BuiltinID == clang::AArch64::BI_WriteStatusReg) {
    llvm_unreachable("NYI");
  }

  if (BuiltinID == clang::AArch64::BI_AddressOfReturnAddress) {
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
  if (std::optional<CIRGenFunction::MSVCIntrin> MsvcIntId =
          translateAarch64ToMsvcIntrin(BuiltinID))
    llvm_unreachable("NYI");

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
        PtrOp0 = buildPointerWithAlignment(E->getArg(0));
        Ops.push_back(PtrOp0.emitRawPointer());
        continue;
      }
    }
    Ops.push_back(buildScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  auto SISDMap = ArrayRef(AArch64SISDIntrinsicMap);
  const ARMVectorIntrinsicInfo *Builtin = findARMVectorIntrinsicInMap(
      SISDMap, BuiltinID, AArch64SISDIntrinsicsProvenSorted);

  if (Builtin) {
    llvm_unreachable("NYI");
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
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
    Ops.push_back(buildScalarExpr(E->getArg(2)));
    return builder.create<mlir::cir::VecInsertOp>(getLoc(E->getExprLoc()),
                                                  Ops[1], Ops[0], Ops[2]);
  case NEON::BI__builtin_neon_vset_lane_bf16:
  case NEON::BI__builtin_neon_vsetq_lane_bf16:
    // No support for now as no real/test case for them
    // at the moment, the implementation should be the same as above
    // vset_lane or vsetq_lane intrinsics
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

  auto Ty = GetNeonType(this, Type);
  if (!Ty)
    return nullptr;

  // Not all intrinsics handled by the common case work for AArch64 yet, so only
  // defer to common code if it's been added to our special map.
  Builtin = findARMVectorIntrinsicInMap(AArch64SIMDIntrinsicMap, BuiltinID,
                                        AArch64SIMDIntrinsicsProvenSorted);
  if (Builtin) {
    llvm_unreachable("NYI");
  }

  if (mlir::Value V =
          buildAArch64TblBuiltinExpr(*this, BuiltinID, E, Ops, Arch))
    return V;

  mlir::Type VTy = Ty;
  llvm::SmallVector<mlir::Value, 4> args;
  switch (BuiltinID) {
  default:
    return nullptr;
  case NEON::BI__builtin_neon_vbsl_v:
  case NEON::BI__builtin_neon_vbslq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vfma_lane_v:
  case NEON::BI__builtin_neon_vfmaq_lane_v: { // Only used for FP types
    // The ARM builtins (and instructions) have the addend as the first
    // operand, but the 'fma' intrinsics have it last. Swap it around here.
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vfma_laneq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vfmaq_laneq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vfmah_lane_f16:
  case NEON::BI__builtin_neon_vfmas_lane_f32:
  case NEON::BI__builtin_neon_vfmah_laneq_f16:
  case NEON::BI__builtin_neon_vfmas_laneq_f32:
  case NEON::BI__builtin_neon_vfmad_lane_f64:
  case NEON::BI__builtin_neon_vfmad_laneq_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmull_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vmax_v:
  case NEON::BI__builtin_neon_vmaxq_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vmaxh_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmin_v:
  case NEON::BI__builtin_neon_vminq_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vminh_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vabd_v:
  case NEON::BI__builtin_neon_vabdq_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vpadal_v:
  case NEON::BI__builtin_neon_vpadalq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vpmin_v:
  case NEON::BI__builtin_neon_vpminq_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vpmax_v:
  case NEON::BI__builtin_neon_vpmaxq_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vminnm_v:
  case NEON::BI__builtin_neon_vminnmq_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vminnmh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vmaxnm_v:
  case NEON::BI__builtin_neon_vmaxnmq_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vmaxnmh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vrecpss_f32: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrecpsd_f64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vrecpsh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vqshrun_n_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vqrshrun_n_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vqshrn_n_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vrshrn_n_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vqrshrn_n_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vrndah_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrnda_v:
  case NEON::BI__builtin_neon_vrndaq_v: {
    assert(!MissingFeatures::buildConstrainedFPCall());
    return buildNeonCall(BuiltinID, *this, {Ty}, Ops, "llvm.round", Ty,
                         getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vrndih_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndmh_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndm_v:
  case NEON::BI__builtin_neon_vrndmq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndnh_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndn_v:
  case NEON::BI__builtin_neon_vrndnq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndns_f32: {
    mlir::Value arg0 = buildScalarExpr(E->getArg(0));
    args.push_back(arg0);
    return buildNeonCall(NEON::BI__builtin_neon_vrndns_f32, *this,
                         {arg0.getType()}, args, "llvm.roundeven.f32",
                         getCIRGenModule().FloatTy, getLoc(E->getExprLoc()));
  }
  case NEON::BI__builtin_neon_vrndph_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndp_v:
  case NEON::BI__builtin_neon_vrndpq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndxh_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndx_v:
  case NEON::BI__builtin_neon_vrndxq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrndh_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrnd32x_f32:
  case NEON::BI__builtin_neon_vrnd32xq_f32:
  case NEON::BI__builtin_neon_vrnd32x_f64:
  case NEON::BI__builtin_neon_vrnd32xq_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrnd32z_f32:
  case NEON::BI__builtin_neon_vrnd32zq_f32:
  case NEON::BI__builtin_neon_vrnd32z_f64:
  case NEON::BI__builtin_neon_vrnd32zq_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrnd64x_f32:
  case NEON::BI__builtin_neon_vrnd64xq_f32:
  case NEON::BI__builtin_neon_vrnd64x_f64:
  case NEON::BI__builtin_neon_vrnd64xq_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrnd64z_f32:
  case NEON::BI__builtin_neon_vrnd64zq_f32:
  case NEON::BI__builtin_neon_vrnd64z_f64:
  case NEON::BI__builtin_neon_vrnd64zq_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrnd_v:
  case NEON::BI__builtin_neon_vrndq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcvt_f64_v:
  case NEON::BI__builtin_neon_vcvtq_f64_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vcvt_f64_f32: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vcvt_f32_f64: {
    llvm_unreachable("NYI");
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
    llvm_unreachable("NYI");
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
    llvm_unreachable("NYI");
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
    llvm_unreachable("NYI");
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
    llvm_unreachable("NYI");
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
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmulx_v:
  case NEON::BI__builtin_neon_vmulxq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmulxh_lane_f16:
  case NEON::BI__builtin_neon_vmulxh_laneq_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmul_lane_v:
  case NEON::BI__builtin_neon_vmul_laneq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vnegd_s64:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vnegh_f16:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vpmaxnm_v:
  case NEON::BI__builtin_neon_vpmaxnmq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vpminnm_v:
  case NEON::BI__builtin_neon_vpminnmq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsqrth_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsqrt_v:
  case NEON::BI__builtin_neon_vsqrtq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vrbit_v:
  case NEON::BI__builtin_neon_vrbitq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddv_u8:
    // FIXME: These are handled by the AArch64 scalar code.
    llvm_unreachable("NYI");
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddv_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddv_u16:
    llvm_unreachable("NYI");
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddv_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddvq_u8:
    llvm_unreachable("NYI");
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddvq_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddvq_u16:
    llvm_unreachable("NYI");
    [[fallthrough]];
  case NEON::BI__builtin_neon_vaddvq_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_u8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_u16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_u8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_u16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxv_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxvq_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminv_u8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminv_u16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminvq_u8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminvq_u16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminv_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminv_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminvq_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminvq_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminv_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminvq_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxnmv_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmaxnmvq_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminnmv_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vminnmvq_f16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vmul_n_f64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlv_u8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlv_u16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlvq_u8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlvq_u16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlv_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlv_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlvq_s8: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vaddlvq_s16: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsri_n_v:
  case NEON::BI__builtin_neon_vsriq_n_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsli_n_v:
  case NEON::BI__builtin_neon_vsliq_n_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsra_n_v:
  case NEON::BI__builtin_neon_vsraq_n_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vrsra_n_v:
  case NEON::BI__builtin_neon_vrsraq_n_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld1_v:
  case NEON::BI__builtin_neon_vld1q_v: {
    return builder.createAlignedLoad(Ops[0].getLoc(), VTy, Ops[0],
                                     PtrOp0.getAlignment());
  }
  case NEON::BI__builtin_neon_vst1_v:
  case NEON::BI__builtin_neon_vst1q_v: {
    Ops[1] = builder.createBitcast(Ops[1], VTy);
    (void)builder.createAlignedStore(Ops[1].getLoc(), Ops[1], Ops[0],
                                     PtrOp0.getAlignment());
    return Ops[1];
  }
  case NEON::BI__builtin_neon_vld1_lane_v:
  case NEON::BI__builtin_neon_vld1q_lane_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vldap1_lane_s64:
  case NEON::BI__builtin_neon_vldap1q_lane_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld1_dup_v:
  case NEON::BI__builtin_neon_vld1q_dup_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vst1_lane_v:
  case NEON::BI__builtin_neon_vst1q_lane_v:
    llvm_unreachable("NYI");
  case NEON::BI__builtin_neon_vstl1_lane_s64:
  case NEON::BI__builtin_neon_vstl1q_lane_s64: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld2_v:
  case NEON::BI__builtin_neon_vld2q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld3_v:
  case NEON::BI__builtin_neon_vld3q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld4_v:
  case NEON::BI__builtin_neon_vld4q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld2_dup_v:
  case NEON::BI__builtin_neon_vld2q_dup_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld3_dup_v:
  case NEON::BI__builtin_neon_vld3q_dup_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld4_dup_v:
  case NEON::BI__builtin_neon_vld4q_dup_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld2_lane_v:
  case NEON::BI__builtin_neon_vld2q_lane_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld3_lane_v:
  case NEON::BI__builtin_neon_vld3q_lane_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vld4_lane_v:
  case NEON::BI__builtin_neon_vld4q_lane_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vst2_v:
  case NEON::BI__builtin_neon_vst2q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vst2_lane_v:
  case NEON::BI__builtin_neon_vst2q_lane_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vst3_v:
  case NEON::BI__builtin_neon_vst3q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vst3_lane_v:
  case NEON::BI__builtin_neon_vst3q_lane_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vst4_v:
  case NEON::BI__builtin_neon_vst4q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vst4_lane_v:
  case NEON::BI__builtin_neon_vst4q_lane_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vtrn_v:
  case NEON::BI__builtin_neon_vtrnq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vuzp_v:
  case NEON::BI__builtin_neon_vuzpq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vzip_v:
  case NEON::BI__builtin_neon_vzipq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbl1q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbl2q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbl3q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbl4q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbx1q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbx2q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbx3q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vqtbx4q_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vsqadd_v:
  case NEON::BI__builtin_neon_vsqaddq_v: {
    llvm_unreachable("NYI");
  }
  case NEON::BI__builtin_neon_vuqadd_v:
  case NEON::BI__builtin_neon_vuqaddq_v: {
    llvm_unreachable("NYI");
  }
  }
}
