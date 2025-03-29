// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-darwin-apple %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CHECK-LABEL: define{{.*}} void @test_half_builtins
void test_half_builtins(half h0, half h1, half h2, int i0) {
  volatile half res;

  // CHECK: call half @llvm.acos.f16(half %h0)
  res = __builtin_acosf16(h0);

  // CHECK: call half @llvm.asin.f16(half %h0)
  res = __builtin_asinf16(h0);

  // CHECK: call half @llvm.atan.f16(half %h0)
  res = __builtin_atanf16(h0);

  // CHECK: call half @llvm.atan2.f16(half %h0, half %h1)
  res = __builtin_atan2f16(h0, h1);

  // CHECK: call half @llvm.copysign.f16(half %h0, half %h1)
  res = __builtin_copysignf16(h0, h1);

  // CHECK: call half @llvm.fabs.f16(half %h0)
  res = __builtin_fabsf16(h0);

  // CHECK: call half @llvm.ceil.f16(half %h0)
  res = __builtin_ceilf16(h0);

  // CHECK: call half @llvm.cos.f16(half %h0)
  res = __builtin_cosf16(h0);

  // CHECK: call half @llvm.cosh.f16(half %h0)
  res = __builtin_coshf16(h0);

  // CHECK: call half @llvm.exp.f16(half %h0)
  res = __builtin_expf16(h0);

  // CHECK: call half @llvm.exp2.f16(half %h0)
  res = __builtin_exp2f16(h0);

  // CHECK: call half @llvm.exp10.f16(half %h0)
  res = __builtin_exp10f16(h0);

  // CHECK: call half @llvm.floor.f16(half %h0)
  res = __builtin_floorf16(h0);

  // CHECK: call half @llvm.fma.f16(half %h0, half %h1, half %h2)
  res = __builtin_fmaf16(h0, h1 ,h2);

  // CHECK: call half @llvm.maxnum.f16(half %h0, half %h1)
  res = __builtin_fmaxf16(h0, h1);

  // CHECK: call half @llvm.minnum.f16(half %h0, half %h1)
  res = __builtin_fminf16(h0, h1);

  // CHECK: frem half %h0, %h1
  res = __builtin_fmodf16(h0, h1);

  // CHECK: call half @llvm.pow.f16(half %h0, half %h1)
  res = __builtin_powf16(h0, h1);

  // CHECK: call half @llvm.log10.f16(half %h0)
  res = __builtin_log10f16(h0);

  // CHECK: call half @llvm.log2.f16(half %h0)
  res = __builtin_log2f16(h0);

  // CHECK: call half @llvm.log.f16(half %h0)
  res = __builtin_logf16(h0);

  // CHECK: call half @llvm.rint.f16(half %h0)
  res = __builtin_rintf16(h0);

  // CHECK: call half @llvm.round.f16(half %h0)
  res = __builtin_roundf16(h0);

  // CHECK: call half @llvm.sin.f16(half %h0)
  res = __builtin_sinf16(h0);

  // CHECK: call half @llvm.sinh.f16(half %h0)
  res = __builtin_sinhf16(h0);

  // CHECK: call half @llvm.sqrt.f16(half %h0)
  res = __builtin_sqrtf16(h0);

  // CHECK: call half @llvm.tan.f16(half %h0)
  res = __builtin_tanf16(h0);

  // CHECK: call half @llvm.tanh.f16(half %h0)
  res = __builtin_tanhf16(h0);

  // CHECK: call half @llvm.trunc.f16(half %h0)
  res = __builtin_truncf16(h0);

  // CHECK: call half @llvm.canonicalize.f16(half %h0)
  res = __builtin_canonicalizef16(h0);

  // CHECK: call half @llvm.ldexp.f16.i32(half %h0, i32 %i0)
  res = __builtin_ldexpf16(h0, i0);
}
