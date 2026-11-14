//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

#ifndef _HLX_HEXAGON_PROTOS_H_
#define _HLX_HEXAGON_PROTOS_H_ 1

#ifdef __HLX__

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.tmp=mem(Rs32)
   C Intrinsic Prototype: HLX_Vector Q6_L_tmp_mem_A(void* BaseAddress)
   Instruction Type:      LD
   Execution Slots:       SLOT1
   ========================================================================== */

#define Q6_L_tmp_mem_A __builtin_HEXAGON_LX_L16b_tmp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.tmp=mem(Rs32)
   C Intrinsic Prototype: HLX_VectorPair Q6_LL_tmp_mem_A(void* BaseAddress)
   Instruction Type:      MAPPING
   Execution Slots:       SLOT0123
   ========================================================================== */

#define Q6_LL_tmp_mem_A __builtin_HEXAGON_LX_LL16b_tmp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       mem(Rs32)=Ku16.new
   C Intrinsic Prototype: void Q6_mem_new_AL(void* BaseAddress, HLX_Vector Ku)
   Instruction Type:      ST
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_mem_new_AL __builtin_HEXAGON_LX_S16b_new
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       mem(Rs32)=Kuu16.new
   C Intrinsic Prototype: void Q6_mem_new_ALL(void* BaseAddress, HLX_VectorPair Kuu)
   Instruction Type:      MAPPING
   Execution Slots:       SLOT0123
   ========================================================================== */

#define Q6_mem_new_ALL __builtin_HEXAGON_LX_SS16b_new
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=abs(Lu16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_abs_Lh(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_abs_Lh __builtin_HEXAGON_LX_abs_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=abs(Lu16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Lh_abs_Lh_sat(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_abs_Lh_sat __builtin_HEXAGON_LX_abs_uh_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uh=abs(Luu16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuh_abs_LWh(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuh_abs_LWh __builtin_HEXAGON_LX_abs_uhp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=abs(Luu16.h):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_abs_LWh_sat(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_abs_LWh_sat __builtin_HEXAGON_LX_abs_uhp_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=abs(Luu16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_abs_LWw(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_abs_LWw __builtin_HEXAGON_LX_abs_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=abs(Luu16.w):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_abs_LWw_sat(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_abs_LWw_sat __builtin_HEXAGON_LX_abs_uw_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=absdiff(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_absdiff_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_absdiff_LhLh __builtin_HEXAGON_LX_absdiff_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=absdiff(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_absdiff_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_absdiff_LuhLuh __builtin_HEXAGON_LX_absdiff_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=absdiff(Luu16.uw,Lvv16.uw)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_absdiff_LWuwLWuw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_absdiff_LWuwLWuw __builtin_HEXAGON_LX_absdiff_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=absdiff(Luu16.w,Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_absdiff_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_absdiff_LWwLWw __builtin_HEXAGON_LX_absdiff_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=add(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_add_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_add_LhLh __builtin_HEXAGON_LX_add_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=add(Lu16.h,Lv16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Lh_add_LhLh_sat(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_add_LhLh_sat __builtin_HEXAGON_LX_add_h_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=add(Luu16.h,Lvv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_add_LWhLWh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_add_LWhLWh __builtin_HEXAGON_LX_add_hp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=add(Luu16.h,Lvv16.h):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_add_LWhLWh_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_add_LWhLWh_sat __builtin_HEXAGON_LX_add_hp_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=add(Lu16.uh,Lv16.uh):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_add_LuhLuh_sat(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_add_LuhLuh_sat __builtin_HEXAGON_LX_add_uh_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=add(Lu16.h,Lv16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_add_LhLh_sat(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_add_LhLh_sat __builtin_HEXAGON_LX_add_uhh_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uh=add(Luu16.h,Lvv16.h):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuh_add_LWhLWh_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuh_add_LWhLWh_sat __builtin_HEXAGON_LX_add_uhp_h_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uh=add(Luu16.uh,Lvv16.uh):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuh_add_LWuhLWuh_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuh_add_LWuhLWuh_sat __builtin_HEXAGON_LX_add_uhp_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=add(Luu16.uw,Lvv16.uw):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_LWuwLWuw_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_add_LWuwLWuw_sat __builtin_HEXAGON_LX_add_uw_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=add(Luu16.w,Lvv16.w):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_LWwLWw_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_add_LWwLWw_sat __builtin_HEXAGON_LX_add_uw_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=add(Luu16.w,Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_add_LWwLWw __builtin_HEXAGON_LX_add_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=add(Luu16.w,Lvv16.w):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_LWwLWw_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_add_LWwLWw_sat __builtin_HEXAGON_LX_add_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=add(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_add_LuhLuh __builtin_HEXAGON_LX_add_wuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=add(Luu16.w,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_LWwLh(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_add_LWwLh __builtin_HEXAGON_LX_add_wwh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=add(Luu16.w,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_LWwLuh(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_add_LWwLuh __builtin_HEXAGON_LX_add_wwuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=and(Lu16,Lv16)
   C Intrinsic Prototype: HLX_Vector Q6_L_and_LL(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_L_and_LL __builtin_HEXAGON_LX_and
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16=and(Luu16,Lvv16)
   C Intrinsic Prototype: HLX_VectorPair Q6_LW_and_LWLW(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LW_and_LWLW __builtin_HEXAGON_LX_and_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=andn(Lu16,Lv16)
   C Intrinsic Prototype: HLX_Vector Q6_L_andn_LL(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_L_andn_LL __builtin_HEXAGON_LX_andn
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16=andn(Luu16,Lvv16)
   C Intrinsic Prototype: HLX_VectorPair Q6_LW_andn_LWLW(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LW_andn_LWLW __builtin_HEXAGON_LX_andn_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=Lu16.h
   C Intrinsic Prototype: HLX_Vector Q6_Lh_equals_Lh(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_equals_Lh __builtin_HEXAGON_LX_assign
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=Rt32.h
   C Intrinsic Prototype: HLX_Vector Q6_Lh_equals_Rh(Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_equals_Rh __builtin_HEXAGON_LX_assign_r
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=Rt32.w
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_equals_Rw(Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_equals_Rw __builtin_HEXAGON_LX_assign_rw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=#s5
   C Intrinsic Prototype: HLX_Vector Q6_Lh_equals_Is(Word32 Is5)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_equals_Is __builtin_HEXAGON_LX_assign_s
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=#u5
   C Intrinsic Prototype: HLX_Vector Q6_Luh_equals_I(Word32 Iu5)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_equals_I __builtin_HEXAGON_LX_assign_u
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=Luu16.w
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_equals_LWw(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_equals_LWw __builtin_HEXAGON_LX_assign_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=avg(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_avg_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_avg_LhLh __builtin_HEXAGON_LX_avg_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=avg(Lu16.h,Lv16.h):rnd
   C Intrinsic Prototype: HLX_Vector Q6_Lh_avg_LhLh_rnd(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_avg_LhLh_rnd __builtin_HEXAGON_LX_avg_h_rnd
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=avg(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_avg_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_avg_LuhLuh __builtin_HEXAGON_LX_avg_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=avg(Lu16.uh,Lv16.uh):rnd
   C Intrinsic Prototype: HLX_Vector Q6_Luh_avg_LuhLuh_rnd(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_avg_LuhLuh_rnd __builtin_HEXAGON_LX_avg_uh_rnd
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=avg(Luu16.uw,Lvv16.uw)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_avg_LWuwLWuw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_avg_LWuwLWuw __builtin_HEXAGON_LX_avg_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=avg(Luu16.uw,Lvv16.uw):rnd
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_avg_LWuwLWuw_rnd(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_avg_LWuwLWuw_rnd __builtin_HEXAGON_LX_avg_uw_rnd
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=avg(Luu16.w,Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_avg_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_avg_LWwLWw __builtin_HEXAGON_LX_avg_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=avg(Luu16.w,Lvv16.w):rnd
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_avg_LWwLWw_rnd(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_avg_LWwLWw_rnd __builtin_HEXAGON_LX_avg_w_rnd
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=cmp.gt(Lu16.h, Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_cmp_gt_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_cmp_gt_LhLh __builtin_HEXAGON_LX_cmp_gt_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=cmp.gt(Luu16.h, Lvv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_cmp_gt_LWhLWh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_cmp_gt_LWhLWh __builtin_HEXAGON_LX_cmp_gt_hp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=cmp.gt(Lu16.uh, Lv16.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_cmp_gt_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_cmp_gt_LuhLuh __builtin_HEXAGON_LX_cmp_gt_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=cmp.gt(Luu16.uh, Lvv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_cmp_gt_LWuhLWuh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_cmp_gt_LWuhLWuh __builtin_HEXAGON_LX_cmp_gt_uhp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=cmp.gt(Luu16.uw, Lvv16.uw)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_cmp_gt_LWuwLWuw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_cmp_gt_LWuwLWuw __builtin_HEXAGON_LX_cmp_gt_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=cmp.gt(Luu16.w, Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_cmp_gt_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_cmp_gt_LWwLWw __builtin_HEXAGON_LX_cmp_gt_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16=combine(Lu16,Lv16)
   C Intrinsic Prototype: HLX_VectorPair Q6_LW_combine_LL(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LW_combine_LL __builtin_HEXAGON_LX_combine_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.b=packhi(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lb_packhi_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lb_packhi_LhLh __builtin_HEXAGON_LX_conv_b_h_hi
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.b=packlo(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lb_packlo_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lb_packlo_LhLh __builtin_HEXAGON_LX_conv_b_h_lo
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=unpackhi(Lu16.b)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_unpackhi_Lb(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_unpackhi_Lb __builtin_HEXAGON_LX_conv_h_b_hi
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=unpacklo(Lu16.b)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_unpacklo_Lb(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_unpacklo_Lb __builtin_HEXAGON_LX_conv_h_b_lo
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=zxt(Lu16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_zxt_Luh(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWuw_zxt_Luh __builtin_HEXAGON_LX_conv_uw_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=sxt(Lu16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_sxt_Lh(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_sxt_Lh __builtin_HEXAGON_LX_conv_w_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=hi(Luu16)
   C Intrinsic Prototype: HLX_Vector Q6_L_hi_LW(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT0123
   ========================================================================== */

#define Q6_L_hi_LW __builtin_HEXAGON_LX_hi
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=lo(Luu16)
   C Intrinsic Prototype: HLX_Vector Q6_L_lo_LW(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT0123
   ========================================================================== */

#define Q6_L_lo_LW __builtin_HEXAGON_LX_lo
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=lut4(Lu16.uh,Rtt32.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_lut4_LuhPh(HLX_Vector Lu, Word64 Rtt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_lut4_LuhPh __builtin_HEXAGON_LX_lut4
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=lut4(Lu16.uh,Rtt32.h):mask
   C Intrinsic Prototype: HLX_Vector Q6_Lh_lut4_LuhPh_mask(HLX_Vector Lu, Word64 Rtt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_lut4_LuhPh_mask __builtin_HEXAGON_LX_lut4_mask
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=lut4(Lu16.uh,Rtt32.uh):>>1
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_lut4_LuhPuh_rs1(HLX_Vector Lu, Word64 Rtt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_lut4_LuhPuh_rs1 __builtin_HEXAGON_LX_lut4_sh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=lut4(Lu16.uh,Rtt32.uh):>>1:neg
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_lut4_LuhPuh_rs1_neg(HLX_Vector Lu, Word64 Rtt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_lut4_LuhPuh_rs1_neg __builtin_HEXAGON_LX_lut4_sh_neg
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=max(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_max_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_max_LhLh __builtin_HEXAGON_LX_max_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=max(Luu16.h,Lvv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_max_LWhLWh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_max_LWhLWh __builtin_HEXAGON_LX_max_hp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=max(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_max_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_max_LuhLuh __builtin_HEXAGON_LX_max_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uh=max(Luu16.uh,Lvv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuh_max_LWuhLWuh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuh_max_LWuhLWuh __builtin_HEXAGON_LX_max_uhp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=max(Luu16.uw,Lvv16.uw)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_max_LWuwLWuw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_max_LWuwLWuw __builtin_HEXAGON_LX_max_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=max(Luu16.w,Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_max_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_max_LWwLWw __builtin_HEXAGON_LX_max_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=min(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_min_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_min_LhLh __builtin_HEXAGON_LX_min_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=min(Luu16.h,Lvv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_min_LWhLWh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_min_LWhLWh __builtin_HEXAGON_LX_min_hp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=min(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_min_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_min_LuhLuh __builtin_HEXAGON_LX_min_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uh=min(Luu16.uh,Lvv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuh_min_LWuhLWuh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuh_min_LWuhLWuh __builtin_HEXAGON_LX_min_uhp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=min(Luu16.uw,Lvv16.uw)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_min_LWuwLWuw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_min_LWuwLWuw __builtin_HEXAGON_LX_min_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=min(Luu16.w,Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_min_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_min_LWwLWw __builtin_HEXAGON_LX_min_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.uh,hi(Lxx16).uh),Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_hi_LuhLWuhLWw(HLX_Vector Lu, HLX_VectorPair Lxx, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_hi_LuhLWuhLWw __builtin_HEXAGON_LX_mpa_uwuhuw_si
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=mpy(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpyacc_LhLh(HLX_VectorPair Lxx, HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpyacc_LhLh __builtin_HEXAGON_LX_mpa_wh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.h,Lv16.uh),hi(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_hi_LhLuhLWuh(HLX_Vector Lu, HLX_Vector Lv, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_hi_LhLuhLWuh __builtin_HEXAGON_LX_mpa_wh_hi
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=mpy(Lu16.h,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpyacc_LhLuh(HLX_VectorPair Lxx, HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpyacc_LhLuh __builtin_HEXAGON_LX_mpa_whuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w=add(mpy(Lu16.h,Lv16.uh),lo(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_mpy_lo_LhLuhLWuh(HLX_Vector Lu, HLX_Vector Lv, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_add_mpy_lo_LhLuhLWuh __builtin_HEXAGON_LX_mpa_whuh_lo
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=mpy(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpyacc_LuhLuh(HLX_VectorPair Lxx, HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpyacc_LuhLuh __builtin_HEXAGON_LX_mpa_wuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.uh,Lv16.uh),hi(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_hi_LuhLuhLWuh(HLX_Vector Lu, HLX_Vector Lv, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_hi_LuhLuhLWuh __builtin_HEXAGON_LX_mpa_wuh_hi
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.uh,Lv16.uh),lo(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_lo_LuhLuhLWuh(HLX_Vector Lu, HLX_Vector Lv, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_lo_LuhLuhLWuh __builtin_HEXAGON_LX_mpa_wuh_lo
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w=add(mpy(Lu16.uh,hi(Lxx16).h),Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_mpy_hi_LuhLWhLWw(HLX_Vector Lu, HLX_VectorPair Lxx, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_add_mpy_hi_LuhLWhLWw __builtin_HEXAGON_LX_mpa_wuhw_si
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.uh,hi(Lxx16).uh),Rt32.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_hi_LuhLWuhRw(HLX_Vector Lu, HLX_VectorPair Lxx, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_hi_LuhLWuhRw __builtin_HEXAGON_LX_mpar_uwuhuw_si
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=mpy(Lu16.h,Rt32.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpyacc_LhRh(HLX_VectorPair Lxx, HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpyacc_LhRh __builtin_HEXAGON_LX_mpar_wh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.h,Rt32.uh),hi(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_hi_LhRuhLWuh(HLX_Vector Lu, Word32 Rt, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_hi_LhRuhLWuh __builtin_HEXAGON_LX_mpar_wh_hi
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=mpy(Lu16.h,Rt32.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpyacc_LhRuh(HLX_VectorPair Lxx, HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpyacc_LhRuh __builtin_HEXAGON_LX_mpar_whuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w=add(mpy(Lu16.h,Rt32.uh),lo(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_mpy_lo_LhRuhLWuh(HLX_Vector Lu, Word32 Rt, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_add_mpy_lo_LhRuhLWuh __builtin_HEXAGON_LX_mpar_whuh_lo
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=mpy(Lu16.uh,Rt32.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpyacc_LuhRuh(HLX_VectorPair Lxx, HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpyacc_LuhRuh __builtin_HEXAGON_LX_mpar_wuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.uh,Rt32.uh),hi(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_hi_LuhRuhLWuh(HLX_Vector Lu, Word32 Rt, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_hi_LuhRuhLWuh __builtin_HEXAGON_LX_mpar_wuh_hi
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.uw=add(mpy(Lu16.uh,Rt32.uh),lo(Lxx16).uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_add_mpy_lo_LuhRuhLWuh(HLX_Vector Lu, Word32 Rt, HLX_VectorPair Lxx)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_add_mpy_lo_LuhRuhLWuh __builtin_HEXAGON_LX_mpar_wuh_lo
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w=add(mpy(Lu16.uh,hi(Lxx16).h),Rt32.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_add_mpy_hi_LuhLWhRw(HLX_Vector Lu, HLX_VectorPair Lxx, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_add_mpy_hi_LuhLWhRw __builtin_HEXAGON_LX_mpar_wuhw_si
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=mpy(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_mpy_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_mpy_LuhLuh __builtin_HEXAGON_LX_mpy_uw_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=mpy(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpy_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpy_LhLh __builtin_HEXAGON_LX_mpy_w_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=mpy(Lu16.h,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpy_LhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpy_LhLuh __builtin_HEXAGON_LX_mpy_w_huh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=mpy(Lu16.uh,Rt32.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_mpy_LuhRuh(HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWuw_mpy_LuhRuh __builtin_HEXAGON_LX_mpyr_uw_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=mpy(Lu16.h,Rt32.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpy_LhRh(HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpy_LhRh __builtin_HEXAGON_LX_mpyr_w_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=mpy(Lu16.h,Rt32.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_mpy_LhRuh(HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT3
   ========================================================================== */

#define Q6_LWw_mpy_LhRuh __builtin_HEXAGON_LX_mpyr_w_huh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lx16.h=mux(Lx16.h,Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_mux_LhLhLh(HLX_Vector Lx, HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_mux_LhLhLh __builtin_HEXAGON_LX_mux_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=navg(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_navg_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_navg_LhLh __builtin_HEXAGON_LX_navg_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=navg(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_navg_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_navg_LuhLuh __builtin_HEXAGON_LX_navg_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=navg(Luu16.uw,Lvv16.uw)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_navg_LWuwLWuw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_navg_LWuwLWuw __builtin_HEXAGON_LX_navg_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=navg(Luu16.w,Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_navg_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_navg_LWwLWw __builtin_HEXAGON_LX_navg_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=not(Lu16)
   C Intrinsic Prototype: HLX_Vector Q6_L_not_L(HLX_Vector Lu)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_L_not_L __builtin_HEXAGON_LX_not
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16=not(Luu16)
   C Intrinsic Prototype: HLX_VectorPair Q6_LW_not_LW(HLX_VectorPair Luu)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LW_not_LW __builtin_HEXAGON_LX_not_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=or(Lu16,Lv16)
   C Intrinsic Prototype: HLX_Vector Q6_L_or_LL(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_L_or_LL __builtin_HEXAGON_LX_or
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16=or(Luu16,Lvv16)
   C Intrinsic Prototype: HLX_VectorPair Q6_LW_or_LWLW(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LW_or_LWLW __builtin_HEXAGON_LX_or_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=orn(Lu16,Lv16)
   C Intrinsic Prototype: HLX_Vector Q6_L_orn_LL(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_L_orn_LL __builtin_HEXAGON_LX_orn
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16=orn(Luu16,Lvv16)
   C Intrinsic Prototype: HLX_VectorPair Q6_LW_orn_LWLW(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LW_orn_LWLW __builtin_HEXAGON_LX_orn_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LhLh __builtin_HEXAGON_LX_shr_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Luu16.w,Lv16.h):rnd:sat
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LWwLh_rnd_sat(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LWwLh_rnd_sat __builtin_HEXAGON_LX_shr_h_w_rnd_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Luu16.w,Lv16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LWwLh_sat(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LWwLh_sat __builtin_HEXAGON_LX_shr_h_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Luu16.w,Lv16.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LWwLuh(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LWwLuh __builtin_HEXAGON_LX_shr_h_w_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Lu16.uh,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LuhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LuhLh __builtin_HEXAGON_LX_shr_uh_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.uw,Lv16.h):rnd:sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWuwLh_rnd_sat(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWuwLh_rnd_sat __builtin_HEXAGON_LX_shr_uh_uw_rnd_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.uw,Lv16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWuwLh_sat(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWuwLh_sat __builtin_HEXAGON_LX_shr_uh_uw_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.w,Lv16.h):rnd:sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWwLh_rnd_sat(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWwLh_rnd_sat __builtin_HEXAGON_LX_shr_uh_w_rnd_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.w,Lv16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWwLh_sat(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWwLh_sat __builtin_HEXAGON_LX_shr_uh_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=shr(Luu16.uw,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_shr_LWuwLh(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_shr_LWuwLh __builtin_HEXAGON_LX_shr_uw_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=shr(Luu16.w,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_shr_LWwLh(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_shr_LWwLh __builtin_HEXAGON_LX_shr_w_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lx16.h+=shr(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shracc_LhLh(HLX_Vector Lx, HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shracc_LhLh __builtin_HEXAGON_LX_shra_h_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lx16.h+=shr(Lu16.uh,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shracc_LuhLh(HLX_Vector Lx, HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shracc_LuhLh __builtin_HEXAGON_LX_shra_h_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=shr(Luu16.uw,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_shracc_LWuwLh(HLX_VectorPair Lxx, HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_shracc_LWuwLh __builtin_HEXAGON_LX_shra_w_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=shr(Luu16.w,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_shracc_LWwLh(HLX_VectorPair Lxx, HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_shracc_LWwLh __builtin_HEXAGON_LX_shra_w_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lx16.h+=shr(Lu16.h,Rt32.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shracc_LhRh(HLX_Vector Lx, HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shracc_LhRh __builtin_HEXAGON_LX_shrar_h_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lx16.h+=shr(Lu16.uh,Rt32.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shracc_LuhRh(HLX_Vector Lx, HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shracc_LuhRh __builtin_HEXAGON_LX_shrar_h_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=shr(Luu16.uw,Rt32.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_shracc_LWuwRh(HLX_VectorPair Lxx, HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_shracc_LWuwRh __builtin_HEXAGON_LX_shrar_w_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Lxx16.w+=shr(Luu16.w,Rt32.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_shracc_LWwRh(HLX_VectorPair Lxx, HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_shracc_LWwRh __builtin_HEXAGON_LX_shrar_w_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Lu16.h,Rt32.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LhRh(HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LhRh __builtin_HEXAGON_LX_shrr_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Luu16.w,Rt32.h):rnd:sat
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LWwRh_rnd_sat(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LWwRh_rnd_sat __builtin_HEXAGON_LX_shrr_h_w_rnd_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Luu16.w,Rt32.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LWwRh_sat(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LWwRh_sat __builtin_HEXAGON_LX_shrr_h_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=shr(Luu16.w,Rt32.uh)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_shr_LWwRuh(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_shr_LWwRuh __builtin_HEXAGON_LX_shrr_h_w_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Lu16.uh,Rt32.h)
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LuhRh(HLX_Vector Lu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LuhRh __builtin_HEXAGON_LX_shrr_uh_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.uw,Rt32.h):rnd:sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWuwRh_rnd_sat(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWuwRh_rnd_sat __builtin_HEXAGON_LX_shrr_uh_uw_rnd_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.uw,Rt32.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWuwRh_sat(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWuwRh_sat __builtin_HEXAGON_LX_shrr_uh_uw_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.w,Rt32.h):rnd:sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWwRh_rnd_sat(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWwRh_rnd_sat __builtin_HEXAGON_LX_shrr_uh_w_rnd_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=shr(Luu16.w,Rt32.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_shr_LWwRh_sat(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Luh_shr_LWwRh_sat __builtin_HEXAGON_LX_shrr_uh_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=shr(Luu16.uw,Rt32.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_shr_LWuwRh(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_shr_LWuwRh __builtin_HEXAGON_LX_shrr_uw_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=shr(Luu16.w,Rt32.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_shr_LWwRh(HLX_VectorPair Luu, Word32 Rt)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_shr_LWwRh __builtin_HEXAGON_LX_shrr_w_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=sub(Lu16.h,Lv16.h)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_sub_LhLh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_sub_LhLh __builtin_HEXAGON_LX_sub_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=sub(Lu16.h,Lv16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Lh_sub_LhLh_sat(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Lh_sub_LhLh_sat __builtin_HEXAGON_LX_sub_h_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=sub(Luu16.h,Lvv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_sub_LWhLWh(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_sub_LWhLWh __builtin_HEXAGON_LX_sub_hp
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.h=sub(Luu16.h,Lvv16.h):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWh_sub_LWhLWh_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWh_sub_LWhLWh_sat __builtin_HEXAGON_LX_sub_hp_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=sub(ilog2(Lu16.h),#u4)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_sub_ilog2_LhI(HLX_Vector Lu, Word32 Iu4)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_sub_ilog2_LhI __builtin_HEXAGON_LX_sub_ilog2_h
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=sub(ilog2(Lu16.uh),#u4)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_sub_ilog2_LuhI(HLX_Vector Lu, Word32 Iu4)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_sub_ilog2_LuhI __builtin_HEXAGON_LX_sub_ilog2_uh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=sub(ilog2(Luu16.uw),#u5)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_sub_ilog2_LWuwI(HLX_VectorPair Luu, Word32 Iu5)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_sub_ilog2_LWuwI __builtin_HEXAGON_LX_sub_ilog2_uw
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.h=sub(ilog2(Luu16.w),#u5)
   C Intrinsic Prototype: HLX_Vector Q6_Lh_sub_ilog2_LWwI(HLX_VectorPair Luu, Word32 Iu5)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_Lh_sub_ilog2_LWwI __builtin_HEXAGON_LX_sub_ilog2_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=sub(Lu16.uh,Lv16.uh):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_sub_LuhLuh_sat(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_sub_LuhLuh_sat __builtin_HEXAGON_LX_sub_uh_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16.uh=sub(Lu16.h,Lv16.h):sat
   C Intrinsic Prototype: HLX_Vector Q6_Luh_sub_LhLh_sat(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_Luh_sub_LhLh_sat __builtin_HEXAGON_LX_sub_uhh_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uh=sub(Luu16.h,Lvv16.h):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuh_sub_LWhLWh_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuh_sub_LWhLWh_sat __builtin_HEXAGON_LX_sub_uhp_h_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uh=sub(Luu16.uh,Lvv16.uh):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuh_sub_LWuhLWuh_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuh_sub_LWuhLWuh_sat __builtin_HEXAGON_LX_sub_uhp_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=sub(Luu16.uw,Lvv16.uw):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_sub_LWuwLWuw_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_sub_LWuwLWuw_sat __builtin_HEXAGON_LX_sub_uw_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.uw=sub(Luu16.w,Lvv16.w):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWuw_sub_LWwLWw_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWuw_sub_LWwLWw_sat __builtin_HEXAGON_LX_sub_uw_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=sub(Luu16.w,Lvv16.w)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_sub_LWwLWw(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_sub_LWwLWw __builtin_HEXAGON_LX_sub_w
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=sub(Luu16.w,Lvv16.w):sat
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_sub_LWwLWw_sat(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LWw_sub_LWwLWw_sat __builtin_HEXAGON_LX_sub_w_sat
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=sub(Lu16.uh,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_sub_LuhLuh(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_sub_LuhLuh __builtin_HEXAGON_LX_sub_wuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=sub(Luu16.w,Lv16.h)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_sub_LWwLh(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_sub_LWwLh __builtin_HEXAGON_LX_sub_wwh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16.w=sub(Luu16.w,Lv16.uh)
   C Intrinsic Prototype: HLX_VectorPair Q6_LWw_sub_LWwLuh(HLX_VectorPair Luu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_LWw_sub_LWwLuh __builtin_HEXAGON_LX_sub_wwuh
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ld16=xor(Lu16,Lv16)
   C Intrinsic Prototype: HLX_Vector Q6_L_xor_LL(HLX_Vector Lu, HLX_Vector Lv)
   Instruction Type:      M
   Execution Slots:       SLOT23
   ========================================================================== */

#define Q6_L_xor_LL __builtin_HEXAGON_LX_xor
#endif

#if __HLX_ARCH__ >= 85
/* ==========================================================================
   Assembly Syntax:       Ldd16=xor(Luu16,Lvv16)
   C Intrinsic Prototype: HLX_VectorPair Q6_LW_xor_LWLW(HLX_VectorPair Luu, HLX_VectorPair Lvv)
   Instruction Type:      M
   Execution Slots:       SLOT2
   ========================================================================== */

#define Q6_LW_xor_LWLW __builtin_HEXAGON_LX_xor_w
#endif

#endif /* __HLX__ */

#endif
