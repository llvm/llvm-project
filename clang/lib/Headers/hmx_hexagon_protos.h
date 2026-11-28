//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

#ifndef _HMX_HEXAGON_PROTOS_H_
#define _HMX_HEXAGON_PROTOS_H_ 1

#ifdef __HMX__
/* ==========================================================================
   Assembly Syntax:       mxclracc.hf
   C Intrinsic Prototype: void Q6_mxclracc_hf()
   Instruction Type:      ST
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_mxclracc_hf __builtin_HEXAGON_M8_mxclracc_hf

/* ==========================================================================
   Assembly Syntax:       bias=mxmem2(Rs32)
   C Intrinsic Prototype: void Q6_bias_mxmem2_A(Address Rs)
   Instruction Type:      LD
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_bias_mxmem2_A __builtin_HEXAGON_M8_mxmem2_bias

/* ==========================================================================
   Assembly Syntax:       mxmem2(Rs32)=bias
   C Intrinsic Prototype: void Q6_mxmem2_bias_A(Address Rs)
   Instruction Type:      ST
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_mxmem2_bias_A __builtin_HEXAGON_M8_mxmem2_st_bias

/* ==========================================================================
   Assembly Syntax:       activation.hf=mxmem(Rs32,Rt32)
   C Intrinsic Prototype: void Q6_activation_hf_mxmem_RR(UWord32 Rs, Word32 Rt)
   Instruction Type:      LD
   Execution Slots:       SLOT01
   ========================================================================== */

#define Q6_activation_hf_mxmem_RR __builtin_HEXAGON_M8_mxmem_blk_sm_act_hf

/* ==========================================================================
   Assembly Syntax:       activation.hf=mxmem(Rs32,Rt32):deep
   C Intrinsic Prototype: void Q6_activation_hf_mxmem_RR_deep(UWord32 Rs, Word32 Rt)
   Instruction Type:      LD
   Execution Slots:       SLOT01
   ========================================================================== */

#define Q6_activation_hf_mxmem_RR_deep __builtin_HEXAGON_M8_mxmem_sm_act_hf

/* ==========================================================================
   Assembly Syntax:       weight.hf=mxmem(Rs32,Rt32)
   C Intrinsic Prototype: void Q6_weight_hf_mxmem_RR(UWord32 Rs, Word32 Rt)
   Instruction Type:      LD
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_weight_hf_mxmem_RR __builtin_HEXAGON_M8_mxmem_wei_hf

/* ==========================================================================
   Assembly Syntax:       weight.hf=mxmem(Rs32,Rt32):deep
   C Intrinsic Prototype: void Q6_weight_hf_mxmem_RR_deep(UWord32 Rs, Word32 Rt)
   Instruction Type:      LD
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_weight_hf_mxmem_RR_deep __builtin_HEXAGON_M8_mxmemdp_wei_hf

/* ==========================================================================
   Assembly Syntax:       activation.hf=mxmem(Rs32,Rt32):single
   C Intrinsic Prototype: void Q6_activation_hf_mxmem_RR_single(UWord32 Rs, Word32 Rt)
   Instruction Type:      LD
   Execution Slots:       SLOT01
   ========================================================================== */

#define Q6_activation_hf_mxmem_RR_single __builtin_HEXAGON_M8_mxmems_blk_sm_act_hf

#if __HMX_ARCH__ >= 73
/* ==========================================================================
   Assembly Syntax:       cvt.hf=acc(Rs32)
   C Intrinsic Prototype: void Q6_cvt_hf_acc_R(Word32 Rs)
   Instruction Type:      ST
   Execution Slots:       SLOT02
   ========================================================================== */

#define Q6_cvt_hf_acc_R __builtin_HEXAGON_M8_cvt_rs_hf
#endif

#if __HMX_ARCH__ >= 73
/* ==========================================================================
   Assembly Syntax:       mxmem(Rs32,Rt32)=cvt
   C Intrinsic Prototype: void Q6_mxmem_cvt_RR(Word32 Rs, Word32 Rt)
   Instruction Type:      ST
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_mxmem_cvt_RR __builtin_HEXAGON_M8_mxmem
#endif

#endif /* __HMX__ */

#endif
