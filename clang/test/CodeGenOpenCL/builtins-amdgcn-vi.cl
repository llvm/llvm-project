// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu tonga -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1010 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1012 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef unsigned long ulong;
typedef unsigned int  uint;

// CHECK-LABEL: @test_div_fixup_f16
// CHECK: {{.*}}call{{.*}} half @llvm.amdgcn.div.fixup.f16
void test_div_fixup_f16(global half* out, half a, half b, half c)
{
  *out = __builtin_amdgcn_div_fixuph(a, b, c);
}

// CHECK-LABEL: @test_rcp_f16
// CHECK: {{.*}}call{{.*}} half @llvm.amdgcn.rcp.f16
void test_rcp_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_rcph(a);
}

// CHECK-LABEL: @test_sqrt_f16
// CHECK: {{.*}}call{{.*}} half @llvm.{{((amdgcn.){0,1})}}sqrt.f16
void test_sqrt_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_sqrth(a);
}

// CHECK-LABEL: @test_rsq_f16
// CHECK: {{.*}}call{{.*}} half @llvm.amdgcn.rsq.f16
void test_rsq_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_rsqh(a);
}

// CHECK-LABEL: @test_sin_f16
// CHECK: {{.*}}call{{.*}} half @llvm.amdgcn.sin.f16
void test_sin_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_sinh(a);
}

// CHECK-LABEL: @test_cos_f16
// CHECK: {{.*}}call{{.*}} half @llvm.amdgcn.cos.f16
void test_cos_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_cosh(a);
}

// CHECK-LABEL: @test_ldexp_f16
// CHECK: [[TRUNC:%[0-9a-z]+]] = trunc i32
// CHECK: {{.*}}call{{.*}} half @llvm.ldexp.f16.i16(half %a, i16 [[TRUNC]])
void test_ldexp_f16(global half* out, half a, int b)
{
  *out = __builtin_amdgcn_ldexph(a, b);
}

// CHECK-LABEL: @test_frexp_mant_f16
// CHECK: {{.*}}call{{.*}} half @llvm.amdgcn.frexp.mant.f16
void test_frexp_mant_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_frexp_manth(a);
}

// CHECK-LABEL: @test_frexp_exp_f16
// CHECK: {{.*}}call{{.*}} i16 @llvm.amdgcn.frexp.exp.i16.f16
void test_frexp_exp_f16(global short* out, half a)
{
  *out = __builtin_amdgcn_frexp_exph(a);
}

// CHECK-LABEL: @test_fract_f16
// CHECK: {{.*}}call{{.*}} half @llvm.amdgcn.fract.f16
void test_fract_f16(global half* out, half a)
{
  *out = __builtin_amdgcn_fracth(a);
}

// CHECK-LABEL: @test_class_f16
// CHECK: {{.*}}call{{.*}} i1 @llvm.amdgcn.class.f16
void test_class_f16(global half* out, half a, int b)
{
  *out = __builtin_amdgcn_classh(a, b);
}

// CHECK-LABEL: @test_s_memrealtime
// CHECK: {{.*}}call{{.*}} i64 @llvm.amdgcn.s.memrealtime()
void test_s_memrealtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memrealtime();
}

// CHECK-LABEL: @test_s_dcache_wb()
// CHECK: {{.*}}call{{.*}} void @llvm.amdgcn.s.dcache.wb()
void test_s_dcache_wb()
{
  __builtin_amdgcn_s_dcache_wb();
}

// CHECK-LABEL: @test_mov_dpp_int
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %src, i32 0, i32 0, i32 0, i1 false)
void test_mov_dpp_int(global int* out, int src)
{
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, 0, false);
}

// CHECK-LABEL: @test_mov_dpp_long
// CHECK:      %0 = tail call{{.*}} i64 @llvm.amdgcn.update.dpp.i64(i64 poison, i64 %x, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: store i64 %0,
void test_mov_dpp_long(long x, global long *p) {
  *p = __builtin_amdgcn_mov_dpp(x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_mov_dpp_float
// CHECK:      %0 = bitcast float %x to i32
// CHECK-NEXT: %1 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: store i32 %1,
void test_mov_dpp_float(float x, global float *p) {
  *p = __builtin_amdgcn_mov_dpp(x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_mov_dpp_double
// CHECK:      %0 = bitcast double %x to i64
// CHECK-NEXT: %1 = tail call{{.*}} i64 @llvm.amdgcn.update.dpp.i64(i64 poison, i64 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: store i64 %1,
void test_mov_dpp_double(double x, global double *p) {
  *p = __builtin_amdgcn_mov_dpp(x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_mov_dpp_short
// CHECK:      %0 = zext i16 %x to i32
// CHECK-NEXT: %1 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: %2 = trunc i32 %1 to i16
// CHECK-NEXT: store i16 %2,
void test_mov_dpp_short(short x, global short *p) {
  *p = __builtin_amdgcn_mov_dpp(x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_mov_dpp_char
// CHECK:      %0 = zext i8 %x to i32
// CHECK-NEXT: %1 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: %2 = trunc i32 %1 to i8
// CHECK-NEXT: store i8 %2,
void test_mov_dpp_char(char x, global char *p) {
  *p = __builtin_amdgcn_mov_dpp(x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_mov_dpp_half
// CHECK:      %0 = load i16,
// CHECK:      %1 = zext i16 %0 to i32
// CHECK-NEXT: %2 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: %3 = trunc i32 %2 to i16
// CHECK-NEXT: store i16 %3,
void test_mov_dpp_half(half *x, global half *p) {
  *p = __builtin_amdgcn_mov_dpp(*x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_update_dpp_int
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 %arg1, i32 %arg2, i32 0, i32 0, i32 0, i1 false)
void test_update_dpp_int(global int* out, int arg1, int arg2)
{
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0, false);
}

// CHECK-LABEL: @test_update_dpp_long
// CHECK:      %0 = tail call{{.*}} i64 @llvm.amdgcn.update.dpp.i64(i64 %x, i64 %x, i32 257, i32 15, i32 15, i1 false)
// CHECk-NEXT: store i64 %0,
void test_update_dpp_long(long x, global long *p) {
  *p = __builtin_amdgcn_update_dpp(x, x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_update_dpp_float
// CHECK:      %0 = bitcast float %x to i32
// CHECK-NEXT: %1 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 %0, i32 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: store i32 %1,
void test_update_dpp_float(float x, global float *p) {
  *p = __builtin_amdgcn_update_dpp(x, x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_update_dpp_double
// CHECK:      %0 = bitcast double %x to i64
// CHECK-NEXT: %1 = tail call{{.*}} i64 @llvm.amdgcn.update.dpp.i64(i64 %0, i64 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: store i64 %1,
void test_update_dpp_double(double x, global double *p) {
  *p = __builtin_amdgcn_update_dpp(x, x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_update_dpp_short
// CHECK:      %0 = zext i16 %x to i32
// CHECK-NEXT: %1 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 %0, i32 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: %2 = trunc i32 %1 to i16
// CHECK-NEXT: store i16 %2,
void test_update_dpp_short(short x, global short *p) {
  *p = __builtin_amdgcn_update_dpp(x, x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_update_dpp_char
// CHECK:      %0 = zext i8 %x to i32
// CHECK-NEXT: %1 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 %0, i32 %0, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: %2 = trunc i32 %1 to i8
// CHECK-NEXT: store i8 %2,
void test_update_dpp_char(char x, global char *p) {
  *p = __builtin_amdgcn_update_dpp(x, x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_update_dpp_half
// CHECK:      %0 = load i16,
// CHECK:      %1 = zext i16 %0 to i32
// CHECK-NEXT: %2 = tail call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 %1, i32 %1, i32 257, i32 15, i32 15, i1 false)
// CHECK-NEXT: %3 = trunc i32 %2 to i16
// CHECK-NEXT: store i16 %3,
void test_update_dpp_half(half *x, global half *p) {
  *p = __builtin_amdgcn_update_dpp(*x, *x, 0x101, 0xf, 0xf, 0);
}

// CHECK-LABEL: @test_update_dpp_int_uint
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 %arg1, i32 %arg2, i32 0, i32 0, i32 0, i1 false)
void test_update_dpp_int_uint(global int* out, int arg1, unsigned int arg2)
{
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0, false);
}

// CHECK-LABEL: @test_update_dpp_lit_int
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 5, i32 %arg1, i32 0, i32 0, i32 0, i1 false)
void test_update_dpp_lit_int(global int* out, int arg1)
{
  *out = __builtin_amdgcn_update_dpp(5, arg1, 0, 0, 0, false);
}

__constant int gi = 5;

// CHECK-LABEL: @test_update_dpp_const_int
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.update.dpp.i32(i32 5, i32 %arg1, i32 0, i32 0, i32 0, i1 false)
void test_update_dpp_const_int(global int* out, int arg1)
{
  *out = __builtin_amdgcn_update_dpp(gi, arg1, 0, 0, 0, false);
}

// CHECK-LABEL: @test_ds_fadd
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src monotonic, align 4{{$}}
// CHECK: atomicrmw volatile fadd ptr addrspace(3) %out, float %src monotonic, align 4{{$}}

// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src acquire, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src acquire, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src release, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src acq_rel, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src seq_cst, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src seq_cst, align 4{{$}}

// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src syncscope("agent") monotonic, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src syncscope("workgroup") monotonic, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src syncscope("wavefront") monotonic, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src syncscope("singlethread") monotonic, align 4{{$}}
// CHECK: atomicrmw fadd ptr addrspace(3) %out, float %src monotonic, align 4{{$}}
#if !defined(__SPIRV__)
void test_ds_faddf(local float *out, float src) {
#else
  void test_ds_faddf(__attribute__((address_space(3))) float *out, float src) {
#endif

  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM, true);

  // Test all orders.
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_CONSUME, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM, false); // invalid

  // Test all syncscopes.
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE, false);
  *out = __builtin_amdgcn_ds_faddf(out, src, __ATOMIC_RELAXED, 5, false); // invalid
}

// CHECK-LABEL: @test_ds_fmin
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src monotonic, align 4{{$}}
// CHECK: atomicrmw volatile fmin ptr addrspace(3) %out, float %src monotonic, align 4{{$}}

// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src acquire, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src acquire, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src release, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src acq_rel, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src seq_cst, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src seq_cst, align 4{{$}}

// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src syncscope("agent") monotonic, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src syncscope("workgroup") monotonic, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src syncscope("wavefront") monotonic, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src syncscope("singlethread") monotonic, align 4{{$}}
// CHECK: atomicrmw fmin ptr addrspace(3) %out, float %src monotonic, align 4{{$}}

#if !defined(__SPIRV__)
void test_ds_fminf(local float *out, float src) {
#else
void test_ds_fminf(__attribute__((address_space(3))) float *out, float src) {
#endif
  *out = __builtin_amdgcn_ds_fminf(out, src, 0, 0, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, 0, 0, true);

  // Test all orders.
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_CONSUME, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM, false); // invalid

  // Test all syncscopes.
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE, false);
  *out = __builtin_amdgcn_ds_fminf(out, src, __ATOMIC_RELAXED, 5, false); // invalid
}

// CHECK-LABEL: @test_ds_fmax
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src monotonic, align 4{{$}}
// CHECK: atomicrmw volatile fmax ptr addrspace(3) %out, float %src monotonic, align 4{{$}}

// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src acquire, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src acquire, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src release, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src acq_rel, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src seq_cst, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src seq_cst, align 4{{$}}

// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src syncscope("agent") monotonic, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src syncscope("workgroup") monotonic, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src syncscope("wavefront") monotonic, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src syncscope("singlethread") monotonic, align 4{{$}}
// CHECK: atomicrmw fmax ptr addrspace(3) %out, float %src monotonic, align 4{{$}}

#if !defined(__SPIRV__)
void test_ds_fmaxf(local float *out, float src) {
#else
void test_ds_fmaxf(__attribute__((address_space(3))) float *out, float src) {
#endif
  *out = __builtin_amdgcn_ds_fmaxf(out, src, 0, 0, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, 0, 0, true);

  // Test all orders.
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_CONSUME, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM, false); // invalid

  // Test all syncscopes.
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE, false);
  *out = __builtin_amdgcn_ds_fmaxf(out, src, __ATOMIC_RELAXED, 5, false); // invalid
}

// CHECK-LABEL: @test_s_memtime
// CHECK: {{.*}}call{{.*}} i64 @llvm.amdgcn.s.memtime()
void test_s_memtime(global ulong* out)
{
  *out = __builtin_amdgcn_s_memtime();
}

// CHECK-LABEL: @test_perm
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.perm(i32 %a, i32 %b, i32 %s)
void test_perm(global uint* out, uint a, uint b, uint s)
{
  *out = __builtin_amdgcn_perm(a, b, s);
}

// CHECK-LABEL: @test_groupstaticsize
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.groupstaticsize()
void test_groupstaticsize(global uint* out)
{
  *out = __builtin_amdgcn_groupstaticsize();
}
