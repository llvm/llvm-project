//===-------- NVPTX.cpp - Emit LLVM Code for builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGBuiltin.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

namespace {
// Helper classes for mapping MMA builtins to particular LLVM intrinsic variant.
struct NVPTXMmaLdstInfo {
  unsigned NumResults;  // Number of elements to load/store
  // Intrinsic IDs for row/col variants. 0 if particular layout is unsupported.
  unsigned IID_col;
  unsigned IID_row;
};

#define MMA_INTR(geom_op_type, layout) \
  Intrinsic::nvvm_wmma_##geom_op_type##_##layout##_stride
#define MMA_LDST(n, geom_op_type)                                              \
  { n, MMA_INTR(geom_op_type, col), MMA_INTR(geom_op_type, row) }

static NVPTXMmaLdstInfo getNVPTXMmaLdstInfo(unsigned BuiltinID) {
  switch (BuiltinID) {
  // FP MMA loads
  case NVPTX::BI__hmma_m16n16k16_ld_a:
    return MMA_LDST(8, m16n16k16_load_a_f16);
  case NVPTX::BI__hmma_m16n16k16_ld_b:
    return MMA_LDST(8, m16n16k16_load_b_f16);
  case NVPTX::BI__hmma_m16n16k16_ld_c_f16:
    return MMA_LDST(4, m16n16k16_load_c_f16);
  case NVPTX::BI__hmma_m16n16k16_ld_c_f32:
    return MMA_LDST(8, m16n16k16_load_c_f32);
  case NVPTX::BI__hmma_m32n8k16_ld_a:
    return MMA_LDST(8, m32n8k16_load_a_f16);
  case NVPTX::BI__hmma_m32n8k16_ld_b:
    return MMA_LDST(8, m32n8k16_load_b_f16);
  case NVPTX::BI__hmma_m32n8k16_ld_c_f16:
    return MMA_LDST(4, m32n8k16_load_c_f16);
  case NVPTX::BI__hmma_m32n8k16_ld_c_f32:
    return MMA_LDST(8, m32n8k16_load_c_f32);
  case NVPTX::BI__hmma_m8n32k16_ld_a:
    return MMA_LDST(8, m8n32k16_load_a_f16);
  case NVPTX::BI__hmma_m8n32k16_ld_b:
    return MMA_LDST(8, m8n32k16_load_b_f16);
  case NVPTX::BI__hmma_m8n32k16_ld_c_f16:
    return MMA_LDST(4, m8n32k16_load_c_f16);
  case NVPTX::BI__hmma_m8n32k16_ld_c_f32:
    return MMA_LDST(8, m8n32k16_load_c_f32);

  // Integer MMA loads
  case NVPTX::BI__imma_m16n16k16_ld_a_s8:
    return MMA_LDST(2, m16n16k16_load_a_s8);
  case NVPTX::BI__imma_m16n16k16_ld_a_u8:
    return MMA_LDST(2, m16n16k16_load_a_u8);
  case NVPTX::BI__imma_m16n16k16_ld_b_s8:
    return MMA_LDST(2, m16n16k16_load_b_s8);
  case NVPTX::BI__imma_m16n16k16_ld_b_u8:
    return MMA_LDST(2, m16n16k16_load_b_u8);
  case NVPTX::BI__imma_m16n16k16_ld_c:
    return MMA_LDST(8, m16n16k16_load_c_s32);
  case NVPTX::BI__imma_m32n8k16_ld_a_s8:
    return MMA_LDST(4, m32n8k16_load_a_s8);
  case NVPTX::BI__imma_m32n8k16_ld_a_u8:
    return MMA_LDST(4, m32n8k16_load_a_u8);
  case NVPTX::BI__imma_m32n8k16_ld_b_s8:
    return MMA_LDST(1, m32n8k16_load_b_s8);
  case NVPTX::BI__imma_m32n8k16_ld_b_u8:
    return MMA_LDST(1, m32n8k16_load_b_u8);
  case NVPTX::BI__imma_m32n8k16_ld_c:
    return MMA_LDST(8, m32n8k16_load_c_s32);
  case NVPTX::BI__imma_m8n32k16_ld_a_s8:
    return MMA_LDST(1, m8n32k16_load_a_s8);
  case NVPTX::BI__imma_m8n32k16_ld_a_u8:
    return MMA_LDST(1, m8n32k16_load_a_u8);
  case NVPTX::BI__imma_m8n32k16_ld_b_s8:
    return MMA_LDST(4, m8n32k16_load_b_s8);
  case NVPTX::BI__imma_m8n32k16_ld_b_u8:
    return MMA_LDST(4, m8n32k16_load_b_u8);
  case NVPTX::BI__imma_m8n32k16_ld_c:
    return MMA_LDST(8, m8n32k16_load_c_s32);

  // Sub-integer MMA loads.
  // Only row/col layout is supported by A/B fragments.
  case NVPTX::BI__imma_m8n8k32_ld_a_s4:
    return {1, 0, MMA_INTR(m8n8k32_load_a_s4, row)};
  case NVPTX::BI__imma_m8n8k32_ld_a_u4:
    return {1, 0, MMA_INTR(m8n8k32_load_a_u4, row)};
  case NVPTX::BI__imma_m8n8k32_ld_b_s4:
    return {1, MMA_INTR(m8n8k32_load_b_s4, col), 0};
  case NVPTX::BI__imma_m8n8k32_ld_b_u4:
    return {1, MMA_INTR(m8n8k32_load_b_u4, col), 0};
  case NVPTX::BI__imma_m8n8k32_ld_c:
    return MMA_LDST(2, m8n8k32_load_c_s32);
  case NVPTX::BI__bmma_m8n8k128_ld_a_b1:
    return {1, 0, MMA_INTR(m8n8k128_load_a_b1, row)};
  case NVPTX::BI__bmma_m8n8k128_ld_b_b1:
    return {1, MMA_INTR(m8n8k128_load_b_b1, col), 0};
  case NVPTX::BI__bmma_m8n8k128_ld_c:
    return MMA_LDST(2, m8n8k128_load_c_s32);

  // Double MMA loads
  case NVPTX::BI__dmma_m8n8k4_ld_a:
    return MMA_LDST(1, m8n8k4_load_a_f64);
  case NVPTX::BI__dmma_m8n8k4_ld_b:
    return MMA_LDST(1, m8n8k4_load_b_f64);
  case NVPTX::BI__dmma_m8n8k4_ld_c:
    return MMA_LDST(2, m8n8k4_load_c_f64);

  // Alternate float MMA loads
  case NVPTX::BI__mma_bf16_m16n16k16_ld_a:
    return MMA_LDST(4, m16n16k16_load_a_bf16);
  case NVPTX::BI__mma_bf16_m16n16k16_ld_b:
    return MMA_LDST(4, m16n16k16_load_b_bf16);
  case NVPTX::BI__mma_bf16_m8n32k16_ld_a:
    return MMA_LDST(2, m8n32k16_load_a_bf16);
  case NVPTX::BI__mma_bf16_m8n32k16_ld_b:
    return MMA_LDST(8, m8n32k16_load_b_bf16);
  case NVPTX::BI__mma_bf16_m32n8k16_ld_a:
    return MMA_LDST(8, m32n8k16_load_a_bf16);
  case NVPTX::BI__mma_bf16_m32n8k16_ld_b:
    return MMA_LDST(2, m32n8k16_load_b_bf16);
  case NVPTX::BI__mma_tf32_m16n16k8_ld_a:
    return MMA_LDST(4, m16n16k8_load_a_tf32);
  case NVPTX::BI__mma_tf32_m16n16k8_ld_b:
    return MMA_LDST(4, m16n16k8_load_b_tf32);
  case NVPTX::BI__mma_tf32_m16n16k8_ld_c:
    return MMA_LDST(8, m16n16k8_load_c_f32);

  // NOTE: We need to follow inconsitent naming scheme used by NVCC.  Unlike
  // PTX and LLVM IR where stores always use fragment D, NVCC builtins always
  // use fragment C for both loads and stores.
  // FP MMA stores.
  case NVPTX::BI__hmma_m16n16k16_st_c_f16:
    return MMA_LDST(4, m16n16k16_store_d_f16);
  case NVPTX::BI__hmma_m16n16k16_st_c_f32:
    return MMA_LDST(8, m16n16k16_store_d_f32);
  case NVPTX::BI__hmma_m32n8k16_st_c_f16:
    return MMA_LDST(4, m32n8k16_store_d_f16);
  case NVPTX::BI__hmma_m32n8k16_st_c_f32:
    return MMA_LDST(8, m32n8k16_store_d_f32);
  case NVPTX::BI__hmma_m8n32k16_st_c_f16:
    return MMA_LDST(4, m8n32k16_store_d_f16);
  case NVPTX::BI__hmma_m8n32k16_st_c_f32:
    return MMA_LDST(8, m8n32k16_store_d_f32);

  // Integer and sub-integer MMA stores.
  // Another naming quirk. Unlike other MMA builtins that use PTX types in the
  // name, integer loads/stores use LLVM's i32.
  case NVPTX::BI__imma_m16n16k16_st_c_i32:
    return MMA_LDST(8, m16n16k16_store_d_s32);
  case NVPTX::BI__imma_m32n8k16_st_c_i32:
    return MMA_LDST(8, m32n8k16_store_d_s32);
  case NVPTX::BI__imma_m8n32k16_st_c_i32:
    return MMA_LDST(8, m8n32k16_store_d_s32);
  case NVPTX::BI__imma_m8n8k32_st_c_i32:
    return MMA_LDST(2, m8n8k32_store_d_s32);
  case NVPTX::BI__bmma_m8n8k128_st_c_i32:
    return MMA_LDST(2, m8n8k128_store_d_s32);

  // Double MMA store
  case NVPTX::BI__dmma_m8n8k4_st_c_f64:
    return MMA_LDST(2, m8n8k4_store_d_f64);

  // Alternate float MMA store
  case NVPTX::BI__mma_m16n16k8_st_c_f32:
    return MMA_LDST(8, m16n16k8_store_d_f32);

  default:
    llvm_unreachable("Unknown MMA builtin");
  }
}
#undef MMA_LDST
#undef MMA_INTR


struct NVPTXMmaInfo {
  unsigned NumEltsA;
  unsigned NumEltsB;
  unsigned NumEltsC;
  unsigned NumEltsD;

  // Variants are ordered by layout-A/layout-B/satf, where 'row' has priority
  // over 'col' for layout. The index of non-satf variants is expected to match
  // the undocumented layout constants used by CUDA's mma.hpp.
  std::array<unsigned, 8> Variants;

  unsigned getMMAIntrinsic(int Layout, bool Satf) {
    unsigned Index = Layout + 4 * Satf;
    if (Index >= Variants.size())
      return 0;
    return Variants[Index];
  }
};

  // Returns an intrinsic that matches Layout and Satf for valid combinations of
  // Layout and Satf, 0 otherwise.
static NVPTXMmaInfo getNVPTXMmaInfo(unsigned BuiltinID) {
  // clang-format off
#define MMA_VARIANTS(geom, type)                                    \
      Intrinsic::nvvm_wmma_##geom##_mma_row_row_##type,             \
      Intrinsic::nvvm_wmma_##geom##_mma_row_col_##type,             \
      Intrinsic::nvvm_wmma_##geom##_mma_col_row_##type,             \
      Intrinsic::nvvm_wmma_##geom##_mma_col_col_##type
#define MMA_SATF_VARIANTS(geom, type)                               \
      MMA_VARIANTS(geom, type),                                     \
      Intrinsic::nvvm_wmma_##geom##_mma_row_row_##type##_satfinite, \
      Intrinsic::nvvm_wmma_##geom##_mma_row_col_##type##_satfinite, \
      Intrinsic::nvvm_wmma_##geom##_mma_col_row_##type##_satfinite, \
      Intrinsic::nvvm_wmma_##geom##_mma_col_col_##type##_satfinite
// Sub-integer MMA only supports row.col layout.
#define MMA_VARIANTS_I4(geom, type) \
      0, \
      Intrinsic::nvvm_wmma_##geom##_mma_row_col_##type,             \
      0, \
      0, \
      0, \
      Intrinsic::nvvm_wmma_##geom##_mma_row_col_##type##_satfinite, \
      0, \
      0
// b1 MMA does not support .satfinite.
#define MMA_VARIANTS_B1_XOR(geom, type) \
      0, \
      Intrinsic::nvvm_wmma_##geom##_mma_xor_popc_row_col_##type,             \
      0, \
      0, \
      0, \
      0, \
      0, \
      0
#define MMA_VARIANTS_B1_AND(geom, type) \
      0, \
      Intrinsic::nvvm_wmma_##geom##_mma_and_popc_row_col_##type,             \
      0, \
      0, \
      0, \
      0, \
      0, \
      0
  // clang-format on
  switch (BuiltinID) {
  // FP MMA
  // Note that 'type' argument of MMA_SATF_VARIANTS uses D_C notation, while
  // NumEltsN of return value are ordered as A,B,C,D.
  case NVPTX::BI__hmma_m16n16k16_mma_f16f16:
    return {8, 8, 4, 4, {{MMA_SATF_VARIANTS(m16n16k16, f16_f16)}}};
  case NVPTX::BI__hmma_m16n16k16_mma_f32f16:
    return {8, 8, 4, 8, {{MMA_SATF_VARIANTS(m16n16k16, f32_f16)}}};
  case NVPTX::BI__hmma_m16n16k16_mma_f16f32:
    return {8, 8, 8, 4, {{MMA_SATF_VARIANTS(m16n16k16, f16_f32)}}};
  case NVPTX::BI__hmma_m16n16k16_mma_f32f32:
    return {8, 8, 8, 8, {{MMA_SATF_VARIANTS(m16n16k16, f32_f32)}}};
  case NVPTX::BI__hmma_m32n8k16_mma_f16f16:
    return {8, 8, 4, 4, {{MMA_SATF_VARIANTS(m32n8k16, f16_f16)}}};
  case NVPTX::BI__hmma_m32n8k16_mma_f32f16:
    return {8, 8, 4, 8, {{MMA_SATF_VARIANTS(m32n8k16, f32_f16)}}};
  case NVPTX::BI__hmma_m32n8k16_mma_f16f32:
    return {8, 8, 8, 4, {{MMA_SATF_VARIANTS(m32n8k16, f16_f32)}}};
  case NVPTX::BI__hmma_m32n8k16_mma_f32f32:
    return {8, 8, 8, 8, {{MMA_SATF_VARIANTS(m32n8k16, f32_f32)}}};
  case NVPTX::BI__hmma_m8n32k16_mma_f16f16:
    return {8, 8, 4, 4, {{MMA_SATF_VARIANTS(m8n32k16, f16_f16)}}};
  case NVPTX::BI__hmma_m8n32k16_mma_f32f16:
    return {8, 8, 4, 8, {{MMA_SATF_VARIANTS(m8n32k16, f32_f16)}}};
  case NVPTX::BI__hmma_m8n32k16_mma_f16f32:
    return {8, 8, 8, 4, {{MMA_SATF_VARIANTS(m8n32k16, f16_f32)}}};
  case NVPTX::BI__hmma_m8n32k16_mma_f32f32:
    return {8, 8, 8, 8, {{MMA_SATF_VARIANTS(m8n32k16, f32_f32)}}};

  // Integer MMA
  case NVPTX::BI__imma_m16n16k16_mma_s8:
    return {2, 2, 8, 8, {{MMA_SATF_VARIANTS(m16n16k16, s8)}}};
  case NVPTX::BI__imma_m16n16k16_mma_u8:
    return {2, 2, 8, 8, {{MMA_SATF_VARIANTS(m16n16k16, u8)}}};
  case NVPTX::BI__imma_m32n8k16_mma_s8:
    return {4, 1, 8, 8, {{MMA_SATF_VARIANTS(m32n8k16, s8)}}};
  case NVPTX::BI__imma_m32n8k16_mma_u8:
    return {4, 1, 8, 8, {{MMA_SATF_VARIANTS(m32n8k16, u8)}}};
  case NVPTX::BI__imma_m8n32k16_mma_s8:
    return {1, 4, 8, 8, {{MMA_SATF_VARIANTS(m8n32k16, s8)}}};
  case NVPTX::BI__imma_m8n32k16_mma_u8:
    return {1, 4, 8, 8, {{MMA_SATF_VARIANTS(m8n32k16, u8)}}};

  // Sub-integer MMA
  case NVPTX::BI__imma_m8n8k32_mma_s4:
    return {1, 1, 2, 2, {{MMA_VARIANTS_I4(m8n8k32, s4)}}};
  case NVPTX::BI__imma_m8n8k32_mma_u4:
    return {1, 1, 2, 2, {{MMA_VARIANTS_I4(m8n8k32, u4)}}};
  case NVPTX::BI__bmma_m8n8k128_mma_xor_popc_b1:
    return {1, 1, 2, 2, {{MMA_VARIANTS_B1_XOR(m8n8k128, b1)}}};
  case NVPTX::BI__bmma_m8n8k128_mma_and_popc_b1:
    return {1, 1, 2, 2, {{MMA_VARIANTS_B1_AND(m8n8k128, b1)}}};

  // Double MMA
  case NVPTX::BI__dmma_m8n8k4_mma_f64:
    return {1, 1, 2, 2, {{MMA_VARIANTS(m8n8k4, f64)}}};

  // Alternate FP MMA
  case NVPTX::BI__mma_bf16_m16n16k16_mma_f32:
    return {4, 4, 8, 8, {{MMA_VARIANTS(m16n16k16, bf16)}}};
  case NVPTX::BI__mma_bf16_m8n32k16_mma_f32:
    return {2, 8, 8, 8, {{MMA_VARIANTS(m8n32k16, bf16)}}};
  case NVPTX::BI__mma_bf16_m32n8k16_mma_f32:
    return {8, 2, 8, 8, {{MMA_VARIANTS(m32n8k16, bf16)}}};
  case NVPTX::BI__mma_tf32_m16n16k8_mma_f32:
    return {4, 4, 8, 8, {{MMA_VARIANTS(m16n16k8, tf32)}}};
  default:
    llvm_unreachable("Unexpected builtin ID.");
  }
#undef MMA_VARIANTS
#undef MMA_SATF_VARIANTS
#undef MMA_VARIANTS_I4
#undef MMA_VARIANTS_B1_AND
#undef MMA_VARIANTS_B1_XOR
}

static Value *MakeLdu(unsigned IntrinsicID, CodeGenFunction &CGF,
                      const CallExpr *E) {
  Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
  QualType ArgType = E->getArg(0)->getType();
  clang::CharUnits Align = CGF.CGM.getNaturalPointeeTypeAlignment(ArgType);
  llvm::Type *ElemTy = CGF.ConvertTypeForMem(ArgType->getPointeeType());
  return CGF.Builder.CreateCall(
      CGF.CGM.getIntrinsic(IntrinsicID, {ElemTy, Ptr->getType()}),
      {Ptr, ConstantInt::get(CGF.Builder.getInt32Ty(), Align.getQuantity())});
}

static Value *MakeLdg(CodeGenFunction &CGF, const CallExpr *E) {
  Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
  QualType ArgType = E->getArg(0)->getType();
  clang::CharUnits AlignV = CGF.CGM.getNaturalPointeeTypeAlignment(ArgType);
  llvm::Type *ElemTy = CGF.ConvertTypeForMem(ArgType->getPointeeType());

  // Use addrspace(1) for NVPTX ADDRESS_SPACE_GLOBAL
  auto *ASC = CGF.Builder.CreateAddrSpaceCast(Ptr, CGF.Builder.getPtrTy(1));
  auto *LD = CGF.Builder.CreateAlignedLoad(ElemTy, ASC, AlignV.getAsAlign());
  MDNode *MD = MDNode::get(CGF.Builder.getContext(), {});
  LD->setMetadata(LLVMContext::MD_invariant_load, MD);

  return LD;
}

static Value *MakeScopedAtomic(unsigned IntrinsicID, CodeGenFunction &CGF,
                               const CallExpr *E) {
  Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Type *ElemTy =
      CGF.ConvertTypeForMem(E->getArg(0)->getType()->getPointeeType());
  return CGF.Builder.CreateCall(
      CGF.CGM.getIntrinsic(IntrinsicID, {ElemTy, Ptr->getType()}),
      {Ptr, CGF.EmitScalarExpr(E->getArg(1))});
}

static Value *MakeCpAsync(unsigned IntrinsicID, unsigned IntrinsicIDS,
                          CodeGenFunction &CGF, const CallExpr *E,
                          int SrcSize) {
  return E->getNumArgs() == 3
             ? CGF.Builder.CreateCall(CGF.CGM.getIntrinsic(IntrinsicIDS),
                                      {CGF.EmitScalarExpr(E->getArg(0)),
                                       CGF.EmitScalarExpr(E->getArg(1)),
                                       CGF.EmitScalarExpr(E->getArg(2))})
             : CGF.Builder.CreateCall(CGF.CGM.getIntrinsic(IntrinsicID),
                                      {CGF.EmitScalarExpr(E->getArg(0)),
                                       CGF.EmitScalarExpr(E->getArg(1))});
}

static Value *MakeHalfType(unsigned IntrinsicID, unsigned BuiltinID,
                           const CallExpr *E, CodeGenFunction &CGF) {
  auto &C = CGF.CGM.getContext();
  if (!(C.getLangOpts().NativeHalfType ||
        !C.getTargetInfo().useFP16ConversionIntrinsics())) {
    CGF.CGM.Error(E->getExprLoc(), C.BuiltinInfo.getQuotedName(BuiltinID) +
                                       " requires native half type support.");
    return nullptr;
  }

  if (BuiltinID == NVPTX::BI__nvvm_ldg_h || BuiltinID == NVPTX::BI__nvvm_ldg_h2)
    return MakeLdg(CGF, E);

  if (IntrinsicID == Intrinsic::nvvm_ldu_global_f)
    return MakeLdu(IntrinsicID, CGF, E);

  SmallVector<Value *, 16> Args;
  auto *F = CGF.CGM.getIntrinsic(IntrinsicID);
  auto *FTy = F->getFunctionType();
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  C.GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");
  for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
    assert((ICEArguments & (1 << i)) == 0);
    auto *ArgValue = CGF.EmitScalarExpr(E->getArg(i));
    auto *PTy = FTy->getParamType(i);
    if (PTy != ArgValue->getType())
      ArgValue = CGF.Builder.CreateBitCast(ArgValue, PTy);
    Args.push_back(ArgValue);
  }

  return CGF.Builder.CreateCall(F, Args);
}
} // namespace

Value *CodeGenFunction::EmitNVPTXBuiltinExpr(unsigned BuiltinID,
                                             const CallExpr *E) {
  switch (BuiltinID) {
  case NVPTX::BI__nvvm_atom_add_gen_i:
  case NVPTX::BI__nvvm_atom_add_gen_l:
  case NVPTX::BI__nvvm_atom_add_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::Add, E);

  case NVPTX::BI__nvvm_atom_sub_gen_i:
  case NVPTX::BI__nvvm_atom_sub_gen_l:
  case NVPTX::BI__nvvm_atom_sub_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::Sub, E);

  case NVPTX::BI__nvvm_atom_and_gen_i:
  case NVPTX::BI__nvvm_atom_and_gen_l:
  case NVPTX::BI__nvvm_atom_and_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::And, E);

  case NVPTX::BI__nvvm_atom_or_gen_i:
  case NVPTX::BI__nvvm_atom_or_gen_l:
  case NVPTX::BI__nvvm_atom_or_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::Or, E);

  case NVPTX::BI__nvvm_atom_xor_gen_i:
  case NVPTX::BI__nvvm_atom_xor_gen_l:
  case NVPTX::BI__nvvm_atom_xor_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::Xor, E);

  case NVPTX::BI__nvvm_atom_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_xchg_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::Xchg, E);

  case NVPTX::BI__nvvm_atom_max_gen_i:
  case NVPTX::BI__nvvm_atom_max_gen_l:
  case NVPTX::BI__nvvm_atom_max_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::Max, E);

  case NVPTX::BI__nvvm_atom_max_gen_ui:
  case NVPTX::BI__nvvm_atom_max_gen_ul:
  case NVPTX::BI__nvvm_atom_max_gen_ull:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::UMax, E);

  case NVPTX::BI__nvvm_atom_min_gen_i:
  case NVPTX::BI__nvvm_atom_min_gen_l:
  case NVPTX::BI__nvvm_atom_min_gen_ll:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::Min, E);

  case NVPTX::BI__nvvm_atom_min_gen_ui:
  case NVPTX::BI__nvvm_atom_min_gen_ul:
  case NVPTX::BI__nvvm_atom_min_gen_ull:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::UMin, E);

  case NVPTX::BI__nvvm_atom_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cas_gen_ll:
    // __nvvm_atom_cas_gen_* should return the old value rather than the
    // success flag.
    return MakeAtomicCmpXchgValue(*this, E, /*ReturnBool=*/false);

  case NVPTX::BI__nvvm_atom_add_gen_f:
  case NVPTX::BI__nvvm_atom_add_gen_d: {
    Address DestAddr = EmitPointerWithAlignment(E->getArg(0));
    Value *Val = EmitScalarExpr(E->getArg(1));

    return Builder.CreateAtomicRMW(llvm::AtomicRMWInst::FAdd, DestAddr, Val,
                                   AtomicOrdering::SequentiallyConsistent);
  }

  case NVPTX::BI__nvvm_atom_inc_gen_ui:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::UIncWrap, E);

  case NVPTX::BI__nvvm_atom_dec_gen_ui:
    return MakeBinaryAtomicValue(*this, llvm::AtomicRMWInst::UDecWrap, E);

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
    // PTX Interoperability section 2.2: "For a vector with an even number of
    // elements, its alignment is set to number of elements times the alignment
    // of its member: n*alignof(t)."
    return MakeLdg(*this, E);

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
    return MakeLdu(Intrinsic::nvvm_ldu_global_i, *this, E);
  case NVPTX::BI__nvvm_ldu_f:
  case NVPTX::BI__nvvm_ldu_f2:
  case NVPTX::BI__nvvm_ldu_f4:
  case NVPTX::BI__nvvm_ldu_d:
  case NVPTX::BI__nvvm_ldu_d2:
    return MakeLdu(Intrinsic::nvvm_ldu_global_f, *this, E);

  case NVPTX::BI__nvvm_atom_cta_add_gen_i:
  case NVPTX::BI__nvvm_atom_cta_add_gen_l:
  case NVPTX::BI__nvvm_atom_cta_add_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_add_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_add_gen_i:
  case NVPTX::BI__nvvm_atom_sys_add_gen_l:
  case NVPTX::BI__nvvm_atom_sys_add_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_add_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_add_gen_f:
  case NVPTX::BI__nvvm_atom_cta_add_gen_d:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_add_gen_f_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_add_gen_f:
  case NVPTX::BI__nvvm_atom_sys_add_gen_d:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_add_gen_f_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_exch_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_exch_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_max_gen_i:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_max_gen_l:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ull:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_max_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_max_gen_i:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_max_gen_l:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ull:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_max_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_min_gen_i:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_min_gen_l:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ull:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_min_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_min_gen_i:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_min_gen_l:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ull:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_min_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_inc_gen_ui:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_inc_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_cta_dec_gen_ui:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_dec_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_inc_gen_ui:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_inc_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_sys_dec_gen_ui:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_dec_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_and_gen_i:
  case NVPTX::BI__nvvm_atom_cta_and_gen_l:
  case NVPTX::BI__nvvm_atom_cta_and_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_and_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_and_gen_i:
  case NVPTX::BI__nvvm_atom_sys_and_gen_l:
  case NVPTX::BI__nvvm_atom_sys_and_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_and_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_or_gen_i:
  case NVPTX::BI__nvvm_atom_cta_or_gen_l:
  case NVPTX::BI__nvvm_atom_cta_or_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_or_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_or_gen_i:
  case NVPTX::BI__nvvm_atom_sys_or_gen_l:
  case NVPTX::BI__nvvm_atom_sys_or_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_or_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_xor_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_xor_gen_i_cta, *this, E);
  case NVPTX::BI__nvvm_atom_sys_xor_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_ll:
    return MakeScopedAtomic(Intrinsic::nvvm_atomic_xor_gen_i_sys, *this, E);
  case NVPTX::BI__nvvm_atom_cta_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_ll: {
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    llvm::Type *ElemTy =
        ConvertTypeForMem(E->getArg(0)->getType()->getPointeeType());
    return Builder.CreateCall(
        CGM.getIntrinsic(
            Intrinsic::nvvm_atomic_cas_gen_i_cta, {ElemTy, Ptr->getType()}),
        {Ptr, EmitScalarExpr(E->getArg(1)), EmitScalarExpr(E->getArg(2))});
  }
  case NVPTX::BI__nvvm_atom_sys_cas_gen_us:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_i:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_l:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_ll: {
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    llvm::Type *ElemTy =
        ConvertTypeForMem(E->getArg(0)->getType()->getPointeeType());
    return Builder.CreateCall(
        CGM.getIntrinsic(
            Intrinsic::nvvm_atomic_cas_gen_i_sys, {ElemTy, Ptr->getType()}),
        {Ptr, EmitScalarExpr(E->getArg(1)), EmitScalarExpr(E->getArg(2))});
  }
  case NVPTX::BI__nvvm_match_all_sync_i32p:
  case NVPTX::BI__nvvm_match_all_sync_i64p: {
    Value *Mask = EmitScalarExpr(E->getArg(0));
    Value *Val = EmitScalarExpr(E->getArg(1));
    Address PredOutPtr = EmitPointerWithAlignment(E->getArg(2));
    Value *ResultPair = Builder.CreateCall(
        CGM.getIntrinsic(BuiltinID == NVPTX::BI__nvvm_match_all_sync_i32p
                             ? Intrinsic::nvvm_match_all_sync_i32p
                             : Intrinsic::nvvm_match_all_sync_i64p),
        {Mask, Val});
    Value *Pred = Builder.CreateZExt(Builder.CreateExtractValue(ResultPair, 1),
                                     PredOutPtr.getElementType());
    Builder.CreateStore(Pred, PredOutPtr);
    return Builder.CreateExtractValue(ResultPair, 0);
  }

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
  // Integer MMA loads.
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
  // Sub-integer MMA loads.
  case NVPTX::BI__imma_m8n8k32_ld_a_s4:
  case NVPTX::BI__imma_m8n8k32_ld_a_u4:
  case NVPTX::BI__imma_m8n8k32_ld_b_s4:
  case NVPTX::BI__imma_m8n8k32_ld_b_u4:
  case NVPTX::BI__imma_m8n8k32_ld_c:
  case NVPTX::BI__bmma_m8n8k128_ld_a_b1:
  case NVPTX::BI__bmma_m8n8k128_ld_b_b1:
  case NVPTX::BI__bmma_m8n8k128_ld_c:
  // Double MMA loads.
  case NVPTX::BI__dmma_m8n8k4_ld_a:
  case NVPTX::BI__dmma_m8n8k4_ld_b:
  case NVPTX::BI__dmma_m8n8k4_ld_c:
  // Alternate float MMA loads.
  case NVPTX::BI__mma_bf16_m16n16k16_ld_a:
  case NVPTX::BI__mma_bf16_m16n16k16_ld_b:
  case NVPTX::BI__mma_bf16_m8n32k16_ld_a:
  case NVPTX::BI__mma_bf16_m8n32k16_ld_b:
  case NVPTX::BI__mma_bf16_m32n8k16_ld_a:
  case NVPTX::BI__mma_bf16_m32n8k16_ld_b:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_a:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_b:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_c: {
    Address Dst = EmitPointerWithAlignment(E->getArg(0));
    Value *Src = EmitScalarExpr(E->getArg(1));
    Value *Ldm = EmitScalarExpr(E->getArg(2));
    std::optional<llvm::APSInt> isColMajorArg =
        E->getArg(3)->getIntegerConstantExpr(getContext());
    if (!isColMajorArg)
      return nullptr;
    bool isColMajor = isColMajorArg->getSExtValue();
    NVPTXMmaLdstInfo II = getNVPTXMmaLdstInfo(BuiltinID);
    unsigned IID = isColMajor ? II.IID_col : II.IID_row;
    if (IID == 0)
      return nullptr;

    Value *Result =
        Builder.CreateCall(CGM.getIntrinsic(IID, Src->getType()), {Src, Ldm});

    // Save returned values.
    assert(II.NumResults);
    if (II.NumResults == 1) {
      Builder.CreateAlignedStore(Result, Dst.emitRawPointer(*this),
                                 CharUnits::fromQuantity(4));
    } else {
      for (unsigned i = 0; i < II.NumResults; ++i) {
        Builder.CreateAlignedStore(
            Builder.CreateBitCast(Builder.CreateExtractValue(Result, i),
                                  Dst.getElementType()),
            Builder.CreateGEP(Dst.getElementType(), Dst.emitRawPointer(*this),
                              llvm::ConstantInt::get(IntTy, i)),
            CharUnits::fromQuantity(4));
      }
    }
    return Result;
  }

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
  case NVPTX::BI__mma_m16n16k8_st_c_f32: {
    Value *Dst = EmitScalarExpr(E->getArg(0));
    Address Src = EmitPointerWithAlignment(E->getArg(1));
    Value *Ldm = EmitScalarExpr(E->getArg(2));
    std::optional<llvm::APSInt> isColMajorArg =
        E->getArg(3)->getIntegerConstantExpr(getContext());
    if (!isColMajorArg)
      return nullptr;
    bool isColMajor = isColMajorArg->getSExtValue();
    NVPTXMmaLdstInfo II = getNVPTXMmaLdstInfo(BuiltinID);
    unsigned IID = isColMajor ? II.IID_col : II.IID_row;
    if (IID == 0)
      return nullptr;
    Function *Intrinsic =
        CGM.getIntrinsic(IID, Dst->getType());
    llvm::Type *ParamType = Intrinsic->getFunctionType()->getParamType(1);
    SmallVector<Value *, 10> Values = {Dst};
    for (unsigned i = 0; i < II.NumResults; ++i) {
      Value *V = Builder.CreateAlignedLoad(
          Src.getElementType(),
          Builder.CreateGEP(Src.getElementType(), Src.emitRawPointer(*this),
                            llvm::ConstantInt::get(IntTy, i)),
          CharUnits::fromQuantity(4));
      Values.push_back(Builder.CreateBitCast(V, ParamType));
    }
    Values.push_back(Ldm);
    Value *Result = Builder.CreateCall(Intrinsic, Values);
    return Result;
  }

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
  case NVPTX::BI__mma_tf32_m16n16k8_mma_f32: {
    Address Dst = EmitPointerWithAlignment(E->getArg(0));
    Address SrcA = EmitPointerWithAlignment(E->getArg(1));
    Address SrcB = EmitPointerWithAlignment(E->getArg(2));
    Address SrcC = EmitPointerWithAlignment(E->getArg(3));
    std::optional<llvm::APSInt> LayoutArg =
        E->getArg(4)->getIntegerConstantExpr(getContext());
    if (!LayoutArg)
      return nullptr;
    int Layout = LayoutArg->getSExtValue();
    if (Layout < 0 || Layout > 3)
      return nullptr;
    llvm::APSInt SatfArg;
    if (BuiltinID == NVPTX::BI__bmma_m8n8k128_mma_xor_popc_b1 ||
        BuiltinID == NVPTX::BI__bmma_m8n8k128_mma_and_popc_b1)
      SatfArg = 0;  // .b1 does not have satf argument.
    else if (std::optional<llvm::APSInt> OptSatfArg =
                 E->getArg(5)->getIntegerConstantExpr(getContext()))
      SatfArg = *OptSatfArg;
    else
      return nullptr;
    bool Satf = SatfArg.getSExtValue();
    NVPTXMmaInfo MI = getNVPTXMmaInfo(BuiltinID);
    unsigned IID = MI.getMMAIntrinsic(Layout, Satf);
    if (IID == 0)  // Unsupported combination of Layout/Satf.
      return nullptr;

    SmallVector<Value *, 24> Values;
    Function *Intrinsic = CGM.getIntrinsic(IID);
    llvm::Type *AType = Intrinsic->getFunctionType()->getParamType(0);
    // Load A
    for (unsigned i = 0; i < MI.NumEltsA; ++i) {
      Value *V = Builder.CreateAlignedLoad(
          SrcA.getElementType(),
          Builder.CreateGEP(SrcA.getElementType(), SrcA.emitRawPointer(*this),
                            llvm::ConstantInt::get(IntTy, i)),
          CharUnits::fromQuantity(4));
      Values.push_back(Builder.CreateBitCast(V, AType));
    }
    // Load B
    llvm::Type *BType = Intrinsic->getFunctionType()->getParamType(MI.NumEltsA);
    for (unsigned i = 0; i < MI.NumEltsB; ++i) {
      Value *V = Builder.CreateAlignedLoad(
          SrcB.getElementType(),
          Builder.CreateGEP(SrcB.getElementType(), SrcB.emitRawPointer(*this),
                            llvm::ConstantInt::get(IntTy, i)),
          CharUnits::fromQuantity(4));
      Values.push_back(Builder.CreateBitCast(V, BType));
    }
    // Load C
    llvm::Type *CType =
        Intrinsic->getFunctionType()->getParamType(MI.NumEltsA + MI.NumEltsB);
    for (unsigned i = 0; i < MI.NumEltsC; ++i) {
      Value *V = Builder.CreateAlignedLoad(
          SrcC.getElementType(),
          Builder.CreateGEP(SrcC.getElementType(), SrcC.emitRawPointer(*this),
                            llvm::ConstantInt::get(IntTy, i)),
          CharUnits::fromQuantity(4));
      Values.push_back(Builder.CreateBitCast(V, CType));
    }
    Value *Result = Builder.CreateCall(Intrinsic, Values);
    llvm::Type *DType = Dst.getElementType();
    for (unsigned i = 0; i < MI.NumEltsD; ++i)
      Builder.CreateAlignedStore(
          Builder.CreateBitCast(Builder.CreateExtractValue(Result, i), DType),
          Builder.CreateGEP(Dst.getElementType(), Dst.emitRawPointer(*this),
                            llvm::ConstantInt::get(IntTy, i)),
          CharUnits::fromQuantity(4));
    return Result;
  }
  // The following builtins require half type support
  case NVPTX::BI__nvvm_ex2_approx_f16:
    return MakeHalfType(Intrinsic::nvvm_ex2_approx_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_ex2_approx_f16x2:
    return MakeHalfType(Intrinsic::nvvm_ex2_approx_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_ff2f16x2_rn:
    return MakeHalfType(Intrinsic::nvvm_ff2f16x2_rn, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_ff2f16x2_rn_relu:
    return MakeHalfType(Intrinsic::nvvm_ff2f16x2_rn_relu, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_ff2f16x2_rz:
    return MakeHalfType(Intrinsic::nvvm_ff2f16x2_rz, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_ff2f16x2_rz_relu:
    return MakeHalfType(Intrinsic::nvvm_ff2f16x2_rz_relu, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_f16:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_ftz_f16:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_ftz_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_ftz_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_ftz_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_ftz_relu_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_ftz_relu_f16x2, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_ftz_sat_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_ftz_sat_f16x2, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fma_rn_relu_f16:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_relu_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_relu_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_relu_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_sat_f16:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_sat_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fma_rn_sat_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fma_rn_sat_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_ftz_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_ftz_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_nan_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_nan_f16x2, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_nan_xorsign_abs_f16, BuiltinID,
                        E, *this);
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_nan_xorsign_abs_f16x2,
                        BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_xorsign_abs_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_ftz_xorsign_abs_f16x2, BuiltinID,
                        E, *this);
  case NVPTX::BI__nvvm_fmax_nan_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_nan_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_nan_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_nan_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_nan_xorsign_abs_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_nan_xorsign_abs_f16x2, BuiltinID,
                        E, *this);
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmax_xorsign_abs_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmax_xorsign_abs_f16x2, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmin_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_ftz_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_ftz_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_nan_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_nan_f16x2, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_nan_xorsign_abs_f16, BuiltinID,
                        E, *this);
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_nan_xorsign_abs_f16x2,
                        BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_xorsign_abs_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_ftz_xorsign_abs_f16x2, BuiltinID,
                        E, *this);
  case NVPTX::BI__nvvm_fmin_nan_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_nan_f16, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_nan_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_nan_f16x2, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_nan_xorsign_abs_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_nan_xorsign_abs_f16x2, BuiltinID,
                        E, *this);
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16:
    return MakeHalfType(Intrinsic::nvvm_fmin_xorsign_abs_f16, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16x2:
    return MakeHalfType(Intrinsic::nvvm_fmin_xorsign_abs_f16x2, BuiltinID, E,
                        *this);
  case NVPTX::BI__nvvm_fabs_f:
  case NVPTX::BI__nvvm_abs_bf16:
  case NVPTX::BI__nvvm_abs_bf16x2:
  case NVPTX::BI__nvvm_fabs_f16:
  case NVPTX::BI__nvvm_fabs_f16x2:
    return Builder.CreateUnaryIntrinsic(Intrinsic::nvvm_fabs,
                                        EmitScalarExpr(E->getArg(0)));
  case NVPTX::BI__nvvm_fabs_ftz_f:
  case NVPTX::BI__nvvm_fabs_ftz_f16:
  case NVPTX::BI__nvvm_fabs_ftz_f16x2:
    return Builder.CreateUnaryIntrinsic(Intrinsic::nvvm_fabs_ftz,
                                        EmitScalarExpr(E->getArg(0)));
  case NVPTX::BI__nvvm_fabs_d:
    return Builder.CreateUnaryIntrinsic(Intrinsic::fabs,
                                        EmitScalarExpr(E->getArg(0)));
  case NVPTX::BI__nvvm_ldg_h:
  case NVPTX::BI__nvvm_ldg_h2:
    return MakeHalfType(Intrinsic::not_intrinsic, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_ldu_h:
  case NVPTX::BI__nvvm_ldu_h2:
    return MakeHalfType(Intrinsic::nvvm_ldu_global_f, BuiltinID, E, *this);
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_4:
    return MakeCpAsync(Intrinsic::nvvm_cp_async_ca_shared_global_4,
                       Intrinsic::nvvm_cp_async_ca_shared_global_4_s, *this, E,
                       4);
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_8:
    return MakeCpAsync(Intrinsic::nvvm_cp_async_ca_shared_global_8,
                       Intrinsic::nvvm_cp_async_ca_shared_global_8_s, *this, E,
                       8);
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_16:
    return MakeCpAsync(Intrinsic::nvvm_cp_async_ca_shared_global_16,
                       Intrinsic::nvvm_cp_async_ca_shared_global_16_s, *this, E,
                       16);
  case NVPTX::BI__nvvm_cp_async_cg_shared_global_16:
    return MakeCpAsync(Intrinsic::nvvm_cp_async_cg_shared_global_16,
                       Intrinsic::nvvm_cp_async_cg_shared_global_16_s, *this, E,
                       16);
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_x:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_clusterid_x));
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_y:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_clusterid_y));
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_z:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_clusterid_z));
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_w:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_clusterid_w));
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_x:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_nclusterid_x));
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_y:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_nclusterid_y));
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_z:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_nclusterid_z));
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_w:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_nclusterid_w));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_x:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_x));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_y:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_y));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_z:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_z));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_w:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_w));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_x:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_x));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_y:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_y));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_z:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_z));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_w:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_w));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctarank:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_ctarank));
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctarank:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_read_ptx_sreg_cluster_nctarank));
  case NVPTX::BI__nvvm_is_explicit_cluster:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_is_explicit_cluster));
  case NVPTX::BI__nvvm_isspacep_shared_cluster:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_isspacep_shared_cluster),
        EmitScalarExpr(E->getArg(0)));
  case NVPTX::BI__nvvm_mapa:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_mapa),
        {EmitScalarExpr(E->getArg(0)), EmitScalarExpr(E->getArg(1))});
  case NVPTX::BI__nvvm_mapa_shared_cluster:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_mapa_shared_cluster),
        {EmitScalarExpr(E->getArg(0)), EmitScalarExpr(E->getArg(1))});
  case NVPTX::BI__nvvm_getctarank:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_getctarank),
        EmitScalarExpr(E->getArg(0)));
  case NVPTX::BI__nvvm_getctarank_shared_cluster:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_getctarank_shared_cluster),
        EmitScalarExpr(E->getArg(0)));
  case NVPTX::BI__nvvm_barrier_cluster_arrive:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_barrier_cluster_arrive));
  case NVPTX::BI__nvvm_barrier_cluster_arrive_relaxed:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_barrier_cluster_arrive_relaxed));
  case NVPTX::BI__nvvm_barrier_cluster_wait:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_barrier_cluster_wait));
  case NVPTX::BI__nvvm_fence_sc_cluster:
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::nvvm_fence_sc_cluster));
  default:
    return nullptr;
  }
}
