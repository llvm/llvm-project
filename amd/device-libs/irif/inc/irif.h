/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef IRIF_H
#define IRIF_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Generic intrinsics
extern __attribute__((const)) half __llvm_sqrt_f16(half) __asm("llvm.sqrt.f16");
extern __attribute__((const)) half __llvm_exp2_f16(half) __asm("llvm.exp2.f16");
extern __attribute__((const)) half __llvm_log2_f16(half) __asm("llvm.log2.f16");
extern __attribute__((const)) half __llvm_sin_f16(half) __asm("llvm.sin.f16");
extern __attribute__((const)) half __llvm_cos_f16(half) __asm("llvm.cos.f16");

extern __attribute__((const)) half __llvm_fma_f16(half, half, half) __asm("llvm.fma.f16");
extern __attribute__((const)) half2 __llvm_fma_2f16(half2, half2, half2) __asm("llvm.fma.v2f16");

extern __attribute__((const)) half __llvm_fabs_f16(half) __asm("llvm.fabs.f16");
extern __attribute__((const)) half2 __llvm_fabs_2f16(half2) __asm("llvm.fabs.v2f16");

extern __attribute__((const)) half __llvm_minnum_f16(half, half) __asm("llvm.minnum.f16");
extern __attribute__((const)) half2 __llvm_minnum_2f16(half2, half2) __asm("llvm.minnum.v2f16");

extern __attribute__((const)) half __llvm_maxnum_f16(half, half) __asm("llvm.maxnum.f16");
extern __attribute__((const)) half2 __llvm_maxnum_2f16(half2, half2) __asm("llvm.maxnum.v2f16");

extern __attribute__((const)) half __llvm_copysign_f16(half, half) __asm("llvm.copysign.f16");
extern __attribute__((const)) half2 __llvm_copysign_2f16(half2, half2) __asm("llvm.copysign.v2f16");

extern __attribute__((const)) half __llvm_floor_f16(half) __asm("llvm.floor.f16");
extern __attribute__((const)) half2 __llvm_floor_2f16(half2) __asm("llvm.floor.v2f16");

extern __attribute__((const)) half __llvm_ceil_f16(half) __asm("llvm.ceil.f16");
extern __attribute__((const)) half2 __llvm_ceil_2f16(half2) __asm("llvm.ceil.v2f16");

extern __attribute__((const)) half __llvm_trunc_f16(half) __asm("llvm.trunc.f16");
extern __attribute__((const)) half2 __llvm_trunc_2f16(half2) __asm("llvm.trunc.v2f16");

extern __attribute__((const)) half __llvm_rint_f16(half) __asm("llvm.rint.f16");
extern __attribute__((const)) half2 __llvm_rint_2f16(half2) __asm("llvm.rint.v2f16");


extern __attribute__((const)) half __llvm_fmuladd_f16(half, half, half) __asm("llvm.fmuladd.f16");
extern __attribute__((const)) half2 __llvm_fmuladd_2f16(half2, half2, half2) __asm("llvm.fmuladd.v2f16");
extern __attribute__((const)) float __llvm_fmuladd_f32(float, float, float) __asm("llvm.fmuladd.f32");
extern __attribute__((const)) double __llvm_fmuladd_f64(double, double, double) __asm("llvm.fmuladd.f64");

extern __attribute__((const)) half __llvm_canonicalize_f16(half) __asm("llvm.canonicalize.f16");
extern __attribute__((const)) half2 __llvm_canonicalize_2f16(half2) __asm("llvm.canonicalize.v2f16");

// Intrinsics requiring wrapping
extern __attribute__((const)) uchar __llvm_ctlz_i8(uchar);
extern __attribute__((const)) ushort __llvm_ctlz_i16(ushort);
extern __attribute__((const)) uint __llvm_ctlz_i32(uint);
extern __attribute__((const)) ulong __llvm_ctlz_i64(ulong);

extern __attribute__((const)) uchar __llvm_cttz_i8(uchar);
extern __attribute__((const)) ushort __llvm_cttz_i16(ushort);
extern __attribute__((const)) uint __llvm_cttz_i32(uint);
extern __attribute__((const)) ulong __llvm_cttz_i64(ulong);

// Fence intrinsics
extern void __llvm_fence_acq_wi(void);
extern void __llvm_fence_acq_sg(void);
extern void __llvm_fence_acq_wg(void);
extern void __llvm_fence_acq_dev(void);
extern void __llvm_fence_acq_sys(void);
extern void __llvm_fence_rel_wi(void);
extern void __llvm_fence_rel_sg(void);
extern void __llvm_fence_rel_wg(void);
extern void __llvm_fence_rel_dev(void);
extern void __llvm_fence_rel_sys(void);
extern void __llvm_fence_ar_wi(void);
extern void __llvm_fence_ar_sg(void);
extern void __llvm_fence_ar_wg(void);
extern void __llvm_fence_ar_dev(void);
extern void __llvm_fence_ar_sys(void);
extern void __llvm_fence_sc_wi(void);
extern void __llvm_fence_sc_sg(void);
extern void __llvm_fence_sc_wg(void);
extern void __llvm_fence_sc_dev(void);
extern void __llvm_fence_sc_sys(void);

// Atomics
extern uint __llvm_ld_atomic_a1_x_dev_i32(__global uint *);
extern ulong __llvm_ld_atomic_a1_x_dev_i64(__global ulong *);
extern uint __llvm_ld_atomic_a3_x_wg_i32(__local uint *);
extern ulong __llvm_ld_atomic_a3_x_wg_i64(__local ulong *);

extern void __llvm_st_atomic_a1_x_dev_i32(__global uint *, uint);
extern void __llvm_st_atomic_a1_x_dev_i64(__global ulong *, ulong);
extern void __llvm_st_atomic_a3_x_wg_i32(__local uint *, uint);
extern void __llvm_st_atomic_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_add_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_add_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_add_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_add_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_and_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_and_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_and_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_and_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_or_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_or_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_or_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_or_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_max_a1_x_dev_i32(__global int *, int);
extern uint __llvm_atomic_umax_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_max_a1_x_dev_i64(__global long *, long);
extern ulong __llvm_atomic_umax_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_max_a3_x_wg_i32(__local int *, int);
extern uint __llvm_atomic_umax_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_max_a3_x_wg_i64(__local long *, long);
extern ulong __llvm_atomic_umax_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_min_a1_x_dev_i32(__global int *, int);
extern uint __llvm_atomic_umin_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_min_a1_x_dev_i64(__global long *, long);
extern ulong __llvm_atomic_umin_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_min_a3_x_wg_i32(__local int *, int);
extern uint __llvm_atomic_umin_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_min_a3_x_wg_i64(__local long *, long);
extern ulong __llvm_atomic_umin_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_cmpxchg_a1_x_x_dev_i32(__global uint *, uint, uint);
extern ulong __llvm_cmpxchg_a1_x_x_dev_i64(__global ulong *, ulong, ulong);
extern uint __llvm_cmpxchg_a3_x_x_wg_i32(__local uint *, uint, uint);
extern ulong __llvm_cmpxchg_a3_x_x_wg_i64(__local ulong *, ulong, ulong);

// Constrained floating point
extern __attribute__((const)) half __llvm_add_rte_f16(half, half);
extern __attribute__((const)) half __llvm_add_rtn_f16(half, half);
extern __attribute__((const)) half __llvm_add_rtp_f16(half, half);
extern __attribute__((const)) half __llvm_add_rtz_f16(half, half);
extern __attribute__((const)) float __llvm_add_rte_f32(float, float);
extern __attribute__((const)) float __llvm_add_rtn_f32(float, float);
extern __attribute__((const)) float __llvm_add_rtp_f32(float, float);
extern __attribute__((const)) float __llvm_add_rtz_f32(float, float);
extern __attribute__((const)) double __llvm_add_rte_f64(double, double);
extern __attribute__((const)) double __llvm_add_rtn_f64(double, double);
extern __attribute__((const)) double __llvm_add_rtp_f64(double, double);
extern __attribute__((const)) double __llvm_add_rtz_f64(double, double);
extern __attribute__((const)) half __llvm_sub_rte_f16(half, half);
extern __attribute__((const)) half __llvm_sub_rtn_f16(half, half);
extern __attribute__((const)) half __llvm_sub_rtp_f16(half, half);
extern __attribute__((const)) half __llvm_sub_rtz_f16(half, half);
extern __attribute__((const)) float __llvm_sub_rte_f32(float, float);
extern __attribute__((const)) float __llvm_sub_rtn_f32(float, float);
extern __attribute__((const)) float __llvm_sub_rtp_f32(float, float);
extern __attribute__((const)) float __llvm_sub_rtz_f32(float, float);
extern __attribute__((const)) double __llvm_sub_rte_f64(double, double);
extern __attribute__((const)) double __llvm_sub_rtn_f64(double, double);
extern __attribute__((const)) double __llvm_sub_rtp_f64(double, double);
extern __attribute__((const)) double __llvm_sub_rtz_f64(double, double);
extern __attribute__((const)) half __llvm_mul_rte_f16(half, half);
extern __attribute__((const)) half __llvm_mul_rtn_f16(half, half);
extern __attribute__((const)) half __llvm_mul_rtp_f16(half, half);
extern __attribute__((const)) half __llvm_mul_rtz_f16(half, half);
extern __attribute__((const)) float __llvm_mul_rte_f32(float, float);
extern __attribute__((const)) float __llvm_mul_rtn_f32(float, float);
extern __attribute__((const)) float __llvm_mul_rtp_f32(float, float);
extern __attribute__((const)) float __llvm_mul_rtz_f32(float, float);
extern __attribute__((const)) double __llvm_mul_rte_f64(double, double);
extern __attribute__((const)) double __llvm_mul_rtn_f64(double, double);
extern __attribute__((const)) double __llvm_mul_rtp_f64(double, double);
extern __attribute__((const)) double __llvm_mul_rtz_f64(double, double);
extern __attribute__((const)) half __llvm_div_rte_f16(half, half);
extern __attribute__((const)) half __llvm_div_rtn_f16(half, half);
extern __attribute__((const)) half __llvm_div_rtp_f16(half, half);
extern __attribute__((const)) half __llvm_div_rtz_f16(half, half);
extern __attribute__((const)) float __llvm_div_rte_f32(float, float);
extern __attribute__((const)) float __llvm_div_rtn_f32(float, float);
extern __attribute__((const)) float __llvm_div_rtp_f32(float, float);
extern __attribute__((const)) float __llvm_div_rtz_f32(float, float);
extern __attribute__((const)) double __llvm_div_rte_f64(double, double);
extern __attribute__((const)) double __llvm_div_rtn_f64(double, double);
extern __attribute__((const)) double __llvm_div_rtp_f64(double, double);
extern __attribute__((const)) double __llvm_div_rtz_f64(double, double);
extern __attribute__((const)) half __llvm_sqrt_rte_f16(half);
extern __attribute__((const)) half __llvm_sqrt_rtn_f16(half);
extern __attribute__((const)) half __llvm_sqrt_rtp_f16(half);
extern __attribute__((const)) half __llvm_sqrt_rtz_f16(half);
extern __attribute__((const)) float __llvm_sqrt_rte_f32(float);
extern __attribute__((const)) float __llvm_sqrt_rtn_f32(float);
extern __attribute__((const)) float __llvm_sqrt_rtp_f32(float);
extern __attribute__((const)) float __llvm_sqrt_rtz_f32(float);
extern __attribute__((const)) double __llvm_sqrt_rte_f64(double);
extern __attribute__((const)) double __llvm_sqrt_rtn_f64(double);
extern __attribute__((const)) double __llvm_sqrt_rtp_f64(double);
extern __attribute__((const)) double __llvm_sqrt_rtz_f64(double);
extern __attribute__((const)) half __llvm_fma_rte_f16(half, half, half);
extern __attribute__((const)) half __llvm_fma_rtn_f16(half, half, half);
extern __attribute__((const)) half __llvm_fma_rtp_f16(half, half, half);
extern __attribute__((const)) half __llvm_fma_rtz_f16(half, half, half);
extern __attribute__((const)) float __llvm_fma_rte_f32(float, float, float);
extern __attribute__((const)) float __llvm_fma_rtn_f32(float, float, float);
extern __attribute__((const)) float __llvm_fma_rtp_f32(float, float, float);
extern __attribute__((const)) float __llvm_fma_rtz_f32(float, float, float);
extern __attribute__((const)) double __llvm_fma_rte_f64(double, double, double);
extern __attribute__((const)) double __llvm_fma_rtn_f64(double, double, double);
extern __attribute__((const)) double __llvm_fma_rtp_f64(double, double, double);
extern __attribute__((const)) double __llvm_fma_rtz_f64(double, double, double);

// AMDGPU intrinsics
extern __attribute__((const)) bool __llvm_amdgcn_class_f16(half, int) __asm("llvm.amdgcn.class.f16");

extern __attribute__((const)) half __llvm_amdgcn_fract_f16(half) __asm("llvm.amdgcn.fract.f16");
extern __attribute__((const)) half __llvm_amdgcn_rcp_f16(half) __asm("llvm.amdgcn.rcp.f16");
extern __attribute__((const)) half __llvm_amdgcn_rsq_f16(half) __asm("llvm.amdgcn.rsq.f16");
extern __attribute__((const)) half __llvm_amdgcn_ldexp_f16(half, int) __asm("llvm.amdgcn.ldexp.f16");


extern __attribute__((const)) half __llvm_amdgcn_frexp_mant_f16(half) __asm("llvm.amdgcn.frexp.mant.f16");
extern __attribute__((const)) short __llvm_amdgcn_frexp_exp_i16_f16(half) __asm("llvm.amdgcn.frexp.exp.i16.f16");

extern __attribute__((const)) half __llvm_amdgcn_fmed3_f16(half, half, half) __asm("llvm.amdgcn.fmed3.f16");

// llvm.amdgcn.update.dpp.i32 <old> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
extern uint __llvm_amdgcn_update_dpp_i32(uint, uint, uint, uint, uint, bool) __asm("llvm.amdgcn.update.dpp.i32");

// Operand bits: [0..3]=VM_CNT, [4..6]=EXP_CNT (Export), [8..11]=LGKM_CNT (LDS, GDS, Konstant, Message)
extern void __llvm_amdgcn_s_waitcnt(int) __asm("llvm.amdgcn.s.waitcnt");

extern void __llvm_amdgcn_s_dcache_inv_vol(void) __asm("llvm.amdgcn.s.dcache.inv.vol");
extern void __llvm_amdgcn_s_dcache_wb(void) __asm("llvm.amdgcn.s.dcache.wb");
extern void __llvm_amdgcn_s_dcache_wb_vol(void) __asm("llvm.amdgcn.s.dcache.wb.vol");
extern void __llvm_amdcgn_buffer_wbinvl1_vol(void) __asm("llvm.amdgcn.buffer.wbinvl1.vol");

extern __attribute__((const)) uint __llvm_amdgcn_mbcnt_lo(uint, uint) __asm("llvm.amdgcn.mbcnt.lo");
extern __attribute__((const)) uint __llvm_amdgcn_mbcnt_hi(uint, uint) __asm("llvm.amdgcn.mbcnt.hi");

extern __attribute__((const)) uint __llvm_amdgcn_ubfe_i32(uint, uint, uint) __asm("llvm.amdgcn.ubfe.i32");
extern __attribute__((const)) int __llvm_amdgcn_sbfe_i32(int, uint, uint) __asm("llvm.amdgcn.sbfe.i32");

extern __attribute__((const)) uint __llvm_amdgcn_alignbit(uint, uint, uint) __asm("llvm.amdgcn.alignbit");
extern __attribute__((const)) uint __llvm_amdgcn_alignbyte(uint, uint, uint) __asm("llvm.amdgcn.alignbyte");

extern __attribute__((const)) ulong __llvm_amdgcn_mqsad_pk_u16_u8(ulong, uint, ulong) __asm("llvm.amdgcn.mqsad.pk.u16.u8");
extern __attribute__((const)) uint __llvm_amdgcn_cvt_pk_u8_f32(float, uint, uint) __asm("llvm.amdgcn.cvt.pk.u8.f32");
extern __attribute__((const)) ulong __llvm_amdgcn_qsad_pk_u16_u8(ulong, uint, ulong) __asm("llvm.amdgcn.qsad.pk.u16.u8");
extern __attribute__((const)) uint __llvm_amdgcn_sad_u8(uint, uint, uint) __asm("llvm.amdgcn.sad.u8");
extern __attribute__((const)) uint __llvm_amdgcn_sad_hi_u8(uint, uint, uint) __asm("llvm.amdgcn.sad.hi.u8");
extern __attribute__((const)) uint __llvm_amdgcn_sad_u16(uint, uint, uint) __asm("llvm.amdgcn.sad.u16");
extern __attribute__((const)) uint __llvm_amdgcn_msad_u8(uint, uint, uint) __asm("llvm.amdgcn.msad.u8");

// Buffer Load/Store

extern __attribute__((pure)) float4 __llvm_amdgcn_buffer_load_format_v4f32(uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.load.format.v4f32");
extern __attribute__((pure)) half4 __llvm_amdgcn_buffer_load_format_v4f16(uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.load.format.v4f16");
extern void __llvm_amdgcn_buffer_store_format_v4f32(float4 p, uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.store.format.v4f32");
extern void __llvm_amdgcn_buffer_store_format_v4f16(half4 p, uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.store.format.v4f16");

// Image load, store, sample, gather
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_1d_v4f32_i32(uint dmask, uint ix, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.1d.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2d_v4f32_i32(uint dmask, uint ix, uint iy, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2d.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_3d_v4f32_i32(uint dmask, uint ix, uint iy, uint iz, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.3d.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_cube_v4f32_i32(uint dmask, uint ix, uint iy, uint iface, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.cube.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_1darray_v4f32_i32(uint dmask, uint ix, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.1darray.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2darray_v4f32_i32(uint dmask, uint ix, uint iy, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2darray.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2dmsaa_v4f32_i32(uint dmask, uint ix, uint iy, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2dmsaa.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2darraymsaa_v4f32_i32(uint dmask, uint ix, uint iy, uint islice, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2darraymsaa.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_1d_v4f32_i32(uint dmask, uint ix, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.1d.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_2d_v4f32_i32(uint dmask, uint ix, uint iy, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.2d.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_3d_v4f32_i32(uint dmask, uint ix, uint iy, uint iz, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.3d.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_cube_v4f32_i32(uint dmask, uint ix, uint iy, uint iface, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.cube.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_1darray_v4f32_i32(uint dmask, uint ix, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.1darray.v4f32.i32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_2darray_v4f32_i32(uint dmask, uint ix, uint iy, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.2darray.v4f32.i32");

extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_1d_v4f16_i32(uint dmask, uint ix, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.1d.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2d_v4f16_i32(uint dmask, uint ix, uint iy, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2d.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_3d_v4f16_i32(uint dmask, uint ix, uint iy, uint iz, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.3d.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_cube_v4f16_i32(uint dmask, uint ix, uint iy, uint iface, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.cube.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_1darray_v4f16_i32(uint dmask, uint ix, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.1darray.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2darray_v4f16_i32(uint dmask, uint ix, uint iy, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2darray.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2dmsaa_v4f16_i32(uint dmask, uint ix, uint iy, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2dmsaa.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2darraymsaa_v4f16_i32(uint dmask, uint ix, uint iy, uint islice, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2darraymsaa.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_1d_v4f16_i32(uint dmask, uint ix, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.1d.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_2d_v4f16_i32(uint dmask, uint ix, uint iy, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.2d.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_3d_v4f16_i32(uint dmask, uint ix, uint iy, uint iz, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.3d.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_cube_v4f16_i32(uint dmask, uint ix, uint iy, uint iface, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.cube.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_1darray_v4f16_i32(uint dmask, uint ix, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.1darray.v4f16.i32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_2darray_v4f16_i32(uint dmask, uint ix, uint iy, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.2darray.v4f16.i32");

extern __attribute__((pure)) float __llvm_amdgcn_image_load_1d_f32_i32(uint dmask, uint ix, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.1d.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_2d_f32_i32(uint dmask, uint ix, uint iy, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2d.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_3d_f32_i32(uint dmask, uint ix, uint iy, uint iz, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.3d.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_cube_f32_i32(uint dmask, uint ix, uint iy, uint iface, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.cube.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_1darray_f32_i32(uint dmask, uint ix, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.1darray.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_2darray_f32_i32(uint dmask, uint ix, uint iy, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2darray.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_2dmsaa_f32_i32(uint dmask, uint ix, uint iy, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2dmsaa.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_2darraymsaa_f32_i32(uint dmask, uint ix, uint iy, uint islice, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.2darraymsaa.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_1d_f32_i32(uint dmask, uint ix, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.1d.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_2d_f32_i32(uint dmask, uint ix, uint iy, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.2d.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_3d_f32_i32(uint dmask, uint ix, uint iy, uint iz, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.3d.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_cube_f32_i32(uint dmask, uint ix, uint iy, uint iface, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.cube.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_1darray_f32_i32(uint dmask, uint ix, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.1darray.f32.i32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_2darray_f32_i32(uint dmask, uint ix, uint iy, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.load.mip.2darray.f32.i32");

extern void __llvm_amdgcn_image_store_1d_v4f32_i32(float4 pix, uint dmask, uint ix, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.1d.v4f32.i32");
 extern void __llvm_amdgcn_image_store_2d_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2d.v4f32.i32");
 extern void __llvm_amdgcn_image_store_3d_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint iz, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.3d.v4f32.i32");
 extern void __llvm_amdgcn_image_store_cube_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint iface, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.cube.v4f32.i32");
 extern void __llvm_amdgcn_image_store_1darray_v4f32_i32(float4 pix, uint dmask, uint ix, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.1darray.v4f32.i32");
 extern void __llvm_amdgcn_image_store_2darray_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2darray.v4f32.i32");
 extern void __llvm_amdgcn_image_store_2dmsaa_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2dmsaa.v4f32.i32");
 extern void __llvm_amdgcn_image_store_2darraymsaa_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint islice, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2darraymsaa.v4f32.i32");
 extern void __llvm_amdgcn_image_store_mip_1d_v4f32_i32(float4 pix, uint dmask, uint ix, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.1d.v4f32.i32");
 extern void __llvm_amdgcn_image_store_mip_2d_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.2d.v4f32.i32");
 extern void __llvm_amdgcn_image_store_mip_3d_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint iz, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.3d.v4f32.i32");
 extern void __llvm_amdgcn_image_store_mip_cube_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint iface, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.cube.v4f32.i32");
 extern void __llvm_amdgcn_image_store_mip_1darray_v4f32_i32(float4 pix, uint dmask, uint ix, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.1darray.v4f32.i32");
 extern void __llvm_amdgcn_image_store_mip_2darray_v4f32_i32(float4 pix, uint dmask, uint ix, uint iy, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.2darray.v4f32.i32");

extern void __llvm_amdgcn_image_store_1d_v4f16_i32(half4 pix, uint dmask, uint ix, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.1d.v4f16.i32");
 extern void __llvm_amdgcn_image_store_2d_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2d.v4f16.i32");
 extern void __llvm_amdgcn_image_store_3d_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint iz, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.3d.v4f16.i32");
 extern void __llvm_amdgcn_image_store_cube_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint iface, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.cube.v4f16.i32");
 extern void __llvm_amdgcn_image_store_1darray_v4f16_i32(half4 pix, uint dmask, uint ix, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.1darray.v4f16.i32");
 extern void __llvm_amdgcn_image_store_2darray_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2darray.v4f16.i32");
 extern void __llvm_amdgcn_image_store_2dmsaa_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2dmsaa.v4f16.i32");
 extern void __llvm_amdgcn_image_store_2darraymsaa_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint islice, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2darraymsaa.v4f16.i32");
 extern void __llvm_amdgcn_image_store_mip_1d_v4f16_i32(half4 pix, uint dmask, uint ix, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.1d.v4f16.i32");
 extern void __llvm_amdgcn_image_store_mip_2d_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.2d.v4f16.i32");
 extern void __llvm_amdgcn_image_store_mip_3d_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint iz, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.3d.v4f16.i32");
 extern void __llvm_amdgcn_image_store_mip_cube_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint iface, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.cube.v4f16.i32");
 extern void __llvm_amdgcn_image_store_mip_1darray_v4f16_i32(half4 pix, uint dmask, uint ix, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.1darray.v4f16.i32");
 extern void __llvm_amdgcn_image_store_mip_2darray_v4f16_i32(half4 pix, uint dmask, uint ix, uint iy, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.2darray.v4f16.i32");

extern void __llvm_amdgcn_image_store_1d_f32_i32(float pix, uint dmask, uint ix, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.1d.f32.i32");
 extern void __llvm_amdgcn_image_store_2d_f32_i32(float pix, uint dmask, uint ix, uint iy, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2d.f32.i32");
 extern void __llvm_amdgcn_image_store_3d_f32_i32(float pix, uint dmask, uint ix, uint iy, uint iz, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.3d.f32.i32");
 extern void __llvm_amdgcn_image_store_cube_f32_i32(float pix, uint dmask, uint ix, uint iy, uint iface, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.cube.f32.i32");
 extern void __llvm_amdgcn_image_store_1darray_f32_i32(float pix, uint dmask, uint ix, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.1darray.f32.i32");
 extern void __llvm_amdgcn_image_store_2darray_f32_i32(float pix, uint dmask, uint ix, uint iy, uint islice, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2darray.f32.i32");
 extern void __llvm_amdgcn_image_store_2dmsaa_f32_i32(float pix, uint dmask, uint ix, uint iy, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2dmsaa.f32.i32");
 extern void __llvm_amdgcn_image_store_2darraymsaa_f32_i32(float pix, uint dmask, uint ix, uint iy, uint islice, uint ifrag, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.2darraymsaa.f32.i32");
 extern void __llvm_amdgcn_image_store_mip_1d_f32_i32(float pix, uint dmask, uint ix, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.1d.f32.i32");
 extern void __llvm_amdgcn_image_store_mip_2d_f32_i32(float pix, uint dmask, uint ix, uint iy, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.2d.f32.i32");
 extern void __llvm_amdgcn_image_store_mip_3d_f32_i32(float pix, uint dmask, uint ix, uint iy, uint iz, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.3d.f32.i32");
 extern void __llvm_amdgcn_image_store_mip_cube_f32_i32(float pix, uint dmask, uint ix, uint iy, uint iface, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.cube.f32.i32");
 extern void __llvm_amdgcn_image_store_mip_1darray_f32_i32(float pix, uint dmask, uint ix, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.1darray.f32.i32");
 extern void __llvm_amdgcn_image_store_mip_2darray_f32_i32(float pix, uint dmask, uint ix, uint iy, uint islice, uint imip, uint8 t, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.store.mip.2darray.f32.i32");

extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_1d_v4f32_f32(uint dmask, float x, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.1d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_1d_v4f32_f32(uint dmask, float x, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.1d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_1d_v4f32_f32_f32(uint dmask, float dxdh, float dxdv, float x, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.1d.v4f32.f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_2d_v4f32_f32(uint dmask, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.2d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_2d_v4f32_f32(uint dmask, float x, float y, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.2d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_2d_v4f32_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_3d_v4f32_f32(uint dmask, float x, float y, float z, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.3d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_3d_v4f32_f32(uint dmask, float x, float y, float z, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.3d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_3d_v4f32_f32_f32(uint dmask, float dxdh, float dydh, float dzdh, float dxdv, float dydv, float dzdv, float x, float y, float z, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.3d.v4f32.f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_cube_v4f32_f32(uint dmask, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.cube.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_cube_v4f32_f32(uint dmask, float x, float y, float face, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.cube.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_cube_v4f32_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.cube.v4f32.f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_1darray_v4f32_f32(uint dmask, float x, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.1darray.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_1darray_v4f32_f32(uint dmask, float x, float slice, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.1darray.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_1darray_v4f32_f32_f32(uint dmask, float dxdh, float dxdv, float x, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.1darray.v4f32.f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_2darray_v4f32_f32(uint dmask, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.2darray.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_2darray_v4f32_f32(uint dmask, float x, float y, float slice, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.2darray.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_2darray_v4f32_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.2darray.v4f32.f32.f32");

extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_1d_v4f16_f32(uint dmask, float x, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.1d.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_1d_v4f16_f32(uint dmask, float x, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.1d.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_1d_v4f16_f32_f32(uint dmask, float dxdh, float dxdv, float x, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.1d.v4f16.f32.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_2d_v4f16_f32(uint dmask, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.2d.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_2d_v4f16_f32(uint dmask, float x, float y, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.2d.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_2d_v4f16_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.2d.v4f16.f32.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_3d_v4f16_f32(uint dmask, float x, float y, float z, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.3d.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_3d_v4f16_f32(uint dmask, float x, float y, float z, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.3d.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_3d_v4f16_f32_f32(uint dmask, float dxdh, float dydh, float dzdh, float dxdv, float dydv, float dzdv, float x, float y, float z, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.3d.v4f16.f32.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_cube_v4f16_f32(uint dmask, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.cube.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_cube_v4f16_f32(uint dmask, float x, float y, float face, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.cube.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_cube_v4f16_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.cube.v4f16.f32.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_1darray_v4f16_f32(uint dmask, float x, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.1darray.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_1darray_v4f16_f32(uint dmask, float x, float slice, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.1darray.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_1darray_v4f16_f32_f32(uint dmask, float dxdh, float dxdv, float x, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.1darray.v4f16.f32.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_2darray_v4f16_f32(uint dmask, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.2darray.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_2darray_v4f16_f32(uint dmask, float x, float y, float slice, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.2darray.v4f16.f32");
 extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_2darray_v4f16_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.2darray.v4f16.f32.f32");

extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_1d_f32_f32(uint dmask, float x, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.1d.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_1d_f32_f32(uint dmask, float x, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.1d.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_1d_f32_f32_f32(uint dmask, float dxdh, float dxdv, float x, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.1d.f32.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_2d_f32_f32(uint dmask, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.2d.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_2d_f32_f32(uint dmask, float x, float y, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.2d.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_2d_f32_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.2d.f32.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_3d_f32_f32(uint dmask, float x, float y, float z, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.3d.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_3d_f32_f32(uint dmask, float x, float y, float z, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.3d.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_3d_f32_f32_f32(uint dmask, float dxdh, float dydh, float dzdh, float dxdv, float dydv, float dzdv, float x, float y, float z, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.3d.f32.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_cube_f32_f32(uint dmask, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.cube.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_cube_f32_f32(uint dmask, float x, float y, float face, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.cube.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_cube_f32_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.cube.f32.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_1darray_f32_f32(uint dmask, float x, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.1darray.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_1darray_f32_f32(uint dmask, float x, float slice, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.1darray.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_1darray_f32_f32_f32(uint dmask, float dxdh, float dxdv, float x, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.1darray.f32.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_2darray_f32_f32(uint dmask, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.lz.2darray.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_2darray_f32_f32(uint dmask, float x, float y, float slice, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.l.2darray.f32.f32");
 extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_2darray_f32_f32_f32(uint dmask, float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.sample.d.2darray.f32.f32.f32");

extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32(uint dmask, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.gather4.lz.2d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_l_2d_v4f32_f32(uint dmask, float x, float y, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.image.amdgcn.gather4.l.2d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_cube_v4f32_f32(uint dmask, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.gather4.lz.cube.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_l_cube_v4f32_f32(uint dmask, float x, float y, float face, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.gather4.l.cube.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2darray_v4f32_f32(uint dmask, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.gather4.lz.2darray.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_l_2darray_v4f32_f32(uint dmask, float x, float y, float slice, float lod, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.gather4.l.2darray.v4f32.f32");

extern __attribute__((pure)) float4 __llvm_amdgcn_gather_4h_2d_v4f32_f32(uint dmask, float x, float y, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.amdgcn.image.gather.4h.2d.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_gather_4h_cube_v4f32_f32(uint dmask, float x, float y, float face, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.image.amdgcn.gather.4h.cube.v4f32.f32");
 extern __attribute__((pure)) float4 __llvm_amdgcn_gather_4h_2darray_v4f32_f32(uint dmask, float x, float y, float slice, uint8 t, uint4 s, bool unorm, uint texfailctrl, uint cachepolicy) __asm("llvm.image.amdgcn.gather.4h.2darray.v4f32.f32");

#pragma OPENCL EXTENSION cl_khr_fp16 : disable
#endif // IRIF_H
