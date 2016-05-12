
#ifndef LLVM_H
#define LLVM_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Generic intrinsics
extern __attribute__((const)) half __llvm_sqrt_f16(half) __asm("llvm.sqrt.f16");
extern __attribute__((const)) float __llvm_sqrt_f32(float) __asm("llvm.sqrt.f32");
extern __attribute__((const)) double __llvm_sqrt_f64(double) __asm("llvm.sqrt.f64");

extern __attribute__((const)) half __llvm_sin_f16(half) __asm("llvm.sin.f16");
extern __attribute__((const)) float __llvm_sin_f32(float) __asm("llvm.sin.f32");

extern __attribute__((const)) half __llvm_cos_f16(half) __asm("llvm.cos.f16");
extern __attribute__((const)) float __llvm_cos_f32(float) __asm("llvm.cos.f32");

extern __attribute__((const)) half __llvm_exp2_f16(half) __asm("llvm.exp2.f16");
extern __attribute__((const)) float __llvm_exp2_f32(float) __asm("llvm.exp2.f32");

extern __attribute__((const)) half __llvm_log2_f16(half) __asm("llvm.log2.f16");
extern __attribute__((const)) float __llvm_log2_f32(float) __asm("llvm.log2.f32");

extern __attribute__((const)) half __llvm_fma_f16(half, half, half) __asm("llvm.fma.f16");
extern __attribute__((const)) float __llvm_fma_f32(float, float, float) __asm("llvm.fma.f32");
extern __attribute__((const)) double __llvm_fma_f64(double, double, double) __asm("llvm.fma.f64");

extern __attribute__((const)) half __llvm_fabs_f16(half) __asm("llvm.fabs.f16");
extern __attribute__((const)) float __llvm_fabs_f32(float) __asm("llvm.fabs.f32");
extern __attribute__((const)) double __llvm_fabs_f64(double) __asm("llvm.fabs.f64");

extern __attribute__((const)) half __llvm_minnum_f16(half, half) __asm("llvm.minnum.f16");
extern __attribute__((const)) float __llvm_minnum_f32(float, float) __asm("llvm.minnum.f32");
extern __attribute__((const)) double __llvm_minnum_f64(double, double) __asm("llvm.minnum.f64");

extern __attribute__((const)) half __llvm_maxnum_f16(half, half) __asm("llvm.maxnum.f16");
extern __attribute__((const)) float __llvm_maxnum_f32(float, float) __asm("llvm.maxnum.f32");
extern __attribute__((const)) double __llvm_maxnum_f64(double, double) __asm("llvm.maxnum.f64");

extern __attribute__((const)) half __llvm_copysign_f16(half, half) __asm("llvm.copysign.f16");
extern __attribute__((const)) float __llvm_copysign_f32(float, float) __asm("llvm.copysign.f32");
extern __attribute__((const)) double __llvm_copysign_f64(double, double) __asm("llvm.copysign.f64");

extern __attribute__((const)) half __llvm_floor_f16(half) __asm("llvm.floor.f16");
extern __attribute__((const)) float __llvm_floor_f32(float) __asm("llvm.floor.f32");
extern __attribute__((const)) double __llvm_floor_f64(double) __asm("llvm.floor.f64");

extern __attribute__((const)) half __llvm_ceil_f16(half) __asm("llvm.ceil.f16");
extern __attribute__((const)) float __llvm_ceil_f32(float) __asm("llvm.ceil.f32");
extern __attribute__((const)) double __llvm_ceil_f64(double) __asm("llvm.ceil.f64");

extern __attribute__((const)) half __llvm_trunc_f16(half) __asm("llvm.trunc.f16");
extern __attribute__((const)) float __llvm_trunc_f32(float) __asm("llvm.trunc.f32");
extern __attribute__((const)) double __llvm_trunc_f64(double) __asm("llvm.trunc.f64");

extern __attribute__((const)) half __llvm_rint_f16(half) __asm("llvm.rint.f16");
extern __attribute__((const)) float __llvm_rint_f32(float) __asm("llvm.rint.f32");
extern __attribute__((const)) double __llvm_rint_f64(double) __asm("llvm.rint.f64");

extern __attribute__((const)) half __llvm_nearbyint_f16(half) __asm("llvm.nearbyint.f16");
extern __attribute__((const)) float __llvm_nearbyint_f32(float) __asm("llvm.nearbyint.f32");
extern __attribute__((const)) double __llvm_nearbyint_f64(double) __asm("llvm.nearbyint.f64");

extern __attribute__((const)) half __llvm_round_f16(half) __asm("llvm.round.f16");
extern __attribute__((const)) float __llvm_round_f32(float) __asm("llvm.round.f32");
extern __attribute__((const)) double __llvm_round_f64(double) __asm("llvm.round.f64");

extern __attribute__((const)) int __llvm_bitreverse_i32(int) __asm("llvm.bitreverse.i32");

extern __attribute__((const)) int __llvm_ctpop_i32(int) __asm("llvm.ctpop.i32");
extern __attribute__((const)) long __llvm_ctpop_i64(long) __asm("llvm.ctpop.i64");

extern __attribute__((const)) half __llvm_fmuladd_f16(half, half, half) __asm("llvm.fmuladd.f16");
extern __attribute__((const)) float __llvm_fmuladd_f32(float, float, float) __asm("llvm.fmuladd.f32");
extern __attribute__((const)) double __llvm_fmuladd_f64(double, double, double) __asm("llvm.fmuladd.f64");

extern __attribute__((const)) half __llvm_canonicalize_f16(half) __asm("llvm.canonicalize.f16");
extern __attribute__((const)) float __llvm_canonicalize_f32(float) __asm("llvm.canonicalize.f32");
extern __attribute__((const)) double __llvm_canonicalize_f64(double) __asm("llvm.canonicalize.f64");

// Intrinsics requiring wrapping
extern bool __llvm_sadd_with_overflow_i16(short, short, __private short*);
extern bool __llvm_uadd_with_overflow_i16(ushort, ushort, __private ushort*);
extern bool __llvm_sadd_with_overflow_i32(int, int, __private int*);
extern bool __llvm_uadd_with_overflow_i32(uint, uint, __private uint*);
extern bool __llvm_sadd_with_overflow_i64(long, long, __private long*);
extern bool __llvm_uadd_with_overflow_i64(ulong, ulong, __private ulong*);

extern bool __llvm_ssub_with_overflow_i16(short, short, __private short*);
extern bool __llvm_usub_with_overflow_i16(ushort, ushort, __private ushort*);
extern bool __llvm_ssub_with_overflow_i32(int, int, __private int*);
extern bool __llvm_usub_with_overflow_i32(uint, uint, __private uint*);
extern bool __llvm_ssub_with_overflow_i64(long, long, __private long*);
extern bool __llvm_usub_with_overflow_i64(ulong, ulong, __private ulong*);

extern bool __llvm_smul_with_overflow_i16(short, short, __private short*);
extern bool __llvm_umul_with_overflow_i16(ushort, ushort, __private ushort*);
extern bool __llvm_smul_with_overflow_i32(int, int, __private int*);
extern bool __llvm_umul_with_overflow_i32(uint, uint, __private uint*);
extern bool __llvm_smul_with_overflow_i64(long, long, __private long*);
extern bool __llvm_umul_with_overflow_i64(ulong, ulong, __private ulong*);

extern __attribute__((const)) int __llvm_ctlz_i32(int);
extern __attribute__((const)) int __llvm_cttz_i32(int);

// AMDGPU intrinsics
extern __attribute__((const)) bool __llvm_amdgcn_class_f16(half, int) __asm("llvm.amdgcn.class.f16");
extern __attribute__((const)) bool __llvm_amdgcn_class_f32(float, int) __asm("llvm.amdgcn.class.f32");
extern __attribute__((const)) bool __llvm_amdgcn_class_f64(double, int) __asm("llvm.amdgcn.class.f64");

extern __attribute__((const)) half __llvm_amdgcn_fract_f16(half) __asm("llvm.amdgcn.fract.f16");
extern __attribute__((const)) float __llvm_amdgcn_fract_f32(float) __asm("llvm.amdgcn.fract.f32");
extern __attribute__((const)) double __llvm_amdgcn_fract_f64(double) __asm("llvm.amdgcn.fract.f64");

extern __attribute__((const)) float __llvm_amdgcn_cos_f32(float) __asm("llvm.amdgcn.cos.f32");

extern __attribute__((const)) half __llvm_amdgcn_exp_f16(float) __asm("llvm.amdgcn.exp.f16");
extern __attribute__((const)) float __llvm_amdgcn_exp_f32(float) __asm("llvm.amdgcn.exp.f32");

extern __attribute__((const)) half __llvm_amdgcn_log_f16(float) __asm("llvm.amdgcn.log.f16");
extern __attribute__((const)) float __llvm_amdgcn_log_f32(float) __asm("llvm.amdgcn.log.f32");

extern __attribute__((const)) half __llvm_amdgcn_rcp_f16(half) __asm("llvm.amdgcn.rcp.f16");
extern __attribute__((const)) float __llvm_amdgcn_rcp_f32(float) __asm("llvm.amdgcn.rcp.f32");
extern __attribute__((const)) double __llvm_amdgcn_rcp_f64(double) __asm("llvm.amdgcn.rcp.f64");

extern __attribute__((const)) half __llvm_amdgcn_rsq_f16(half) __asm("llvm.amdgcn.rsq.f16");
extern __attribute__((const)) float __llvm_amdgcn_rsq_f32(float) __asm("llvm.amdgcn.rsq.f32");
extern __attribute__((const)) double __llvm_amdgcn_rsq_f64(double) __asm("llvm.amdgcn.rsq.f64");

extern __attribute__((const)) float __llvm_amdgcn_sin_f32(float) __asm("llvm.amdgcn.sin.f32");

extern __attribute__((const)) half __llvm_amdgcn_ldexp_f16(half, int) __asm("llvm.amdgcn.ldexp.f16");
extern __attribute__((const)) float __llvm_amdgcn_ldexp_f32(float, int) __asm("llvm.amdgcn.ldexp.f32");
extern __attribute__((const)) double __llvm_amdgcn_ldexp_f64(double, int) __asm("llvm.amdgcn.ldexp.f64");

extern __attribute__((const)) half __llvm_amdgcn_frexp_mant_f16(half) __asm("llvm.amdgcn.frexp.mant.f16");
extern __attribute__((const)) float __llvm_amdgcn_frexp_mant_f32(float) __asm("llvm.amdgcn.frexp.mant.f32");
extern __attribute__((const)) double __llvm_amdgcn_frexp_mant_f64(double) __asm("llvm.amdgcn.frexp.mant.f64");

extern __attribute__((const)) int __llvm_amdgcn_frexp_exp_f16(half) __asm("llvm.amdgcn.frexp.exp.f16");
extern __attribute__((const)) int __llvm_amdgcn_frexp_exp_f32(float) __asm("llvm.amdgcn.frexp.exp.f32");
extern __attribute__((const)) int __llvm_amdgcn_frexp_exp_f64(double) __asm("llvm.amdgcn.frexp.exp.f64");

extern __attribute__((const)) double __llvm_amdgcn_trig_preop_f64(double, int) __asm("llvm.amdgcn.trig.preop.f64");

extern __attribute__((const)) float __llvm_amdgcn_max3_f32(float, float, float) __asm("llvm.amdgcn.max3.f32");
extern __attribute__((const)) float __llvm_amdgcn_med3_f32(float, float, float) __asm("llvm.amdgcn.med3.f32");
extern __attribute__((const)) float __llvm_amdgcn_min3_f32(float, float, float) __asm("llvm.amdgcn.min3.f32");

extern __attribute__((const)) int __llvm_amdgcn_max3_i32(int, int, int) __asm("llvm.amdgcn.max3.i32");
extern __attribute__((const)) int __llvm_amdgcn_med3_i32(int, int, int) __asm("llvm.amdgcn.med3.i32");
extern __attribute__((const)) int __llvm_amdgcn_min3_i32(int, int, int) __asm("llvm.amdgcn.min3.i32");

extern __attribute__((const)) uint __llvm_amdgcn_max3u_i32(uint, uint, uint) __asm("llvm.amdgcn.max3u.i32");
extern __attribute__((const)) uint __llvm_amdgcn_med3u_i32(uint, uint, uint) __asm("llvm.amdgcn.med3u.i32");
extern __attribute__((const)) uint __llvm_amdgcn_min3u_i32(uint, uint, uint) __asm("llvm.amdgcn.min3u.i32");

extern void __llvm_amdgcn_s_barrier(void) __asm("llvm.amdgcn.s.barrier");

extern __attribute__((const)) uint __llvm_amdgcn_workitem_id_x(void) __asm("llvm.amdgcn.workitem.id.x");
extern __attribute__((const)) uint __llvm_amdgcn_workitem_id_y(void) __asm("llvm.amdgcn.workitem.id.y");
extern __attribute__((const)) uint __llvm_amdgcn_workitem_id_z(void) __asm("llvm.amdgcn.workitem.id.z");

extern __attribute__((const)) uint __llvm_amdgcn_workgroup_id_x(void) __asm("llvm.amdgcn.workitem.id.x");
extern __attribute__((const)) uint __llvm_amdgcn_workgroup_id_y(void) __asm("llvm.amdgcn.workitem.id.y");
extern __attribute__((const)) uint __llvm_amdgcn_workgroup_id_z(void) __asm("llvm.amdgcn.workitem.id.z");

extern __attribute__((const)) __constant void *__llvm_amdgcn_dispatch_ptr(void) __asm("llvm.amdgcn.dispatch.ptr");

extern void __llvm_amdgcn_s_sleep(uint) __asm("llvm.amdgcn.s.sleep");
extern ulong __llvm_amdgcn_s_memtime(void) __asm("llvm.amdgcn.s.memtime");
extern ulong __llvm_amdgcn_s_memrealtime(void) __asm("llvm.amdgcn.s.memrealtime");

extern uint __attribute__((pure)) __llvm_amdgcn_ds_bpermute(uint, uint) __asm("llvm.amdgcn.ds.bpermute");

#pragma OPENCL EXTENSION cl_khr_fp16 : disable
#endif // LLVM_H
