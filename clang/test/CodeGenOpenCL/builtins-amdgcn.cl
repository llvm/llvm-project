// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// CHECK-LABEL: @test_div_scale_f64
// CHECK: call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { double, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { double, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i32
// CHECK: store i32 [[FLAGEXT]]
void test_div_scale_f64(global double* out, global int* flagout, double a, double b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scale(a, b, true, &flag);
  *flagout = flag;
}

// CHECK-LABEL: @test_div_scale_f32
// CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true)
// CHECK-DAG: [[FLAG:%.+]] = extractvalue { float, i1 } %{{.+}}, 1
// CHECK-DAG: [[VAL:%.+]] = extractvalue { float, i1 } %{{.+}}, 0
// CHECK: [[FLAGEXT:%.+]] = zext i1 [[FLAG]] to i32
// CHECK: store i32 [[FLAGEXT]]
void test_div_scale_f32(global float* out, global int* flagout, float a, float b)
{
  bool flag;
  *out = __builtin_amdgcn_div_scalef(a, b, true, &flag);
  *flagout = flag;
}

// CHECK-LABEL: @test_div_fmas_f32
// CHECK: call float @llvm.amdgcn.div.fmas.f32
void test_div_fmas_f32(global float* out, float a, float b, float c, int d)
{
  *out = __builtin_amdgcn_div_fmasf(a, b, c, d);
}

// CHECK-LABEL: @test_div_fmas_f64
// CHECK: call double @llvm.amdgcn.div.fmas.f64
void test_div_fmas_f64(global double* out, double a, double b, double c, int d)
{
  *out = __builtin_amdgcn_div_fmas(a, b, c, d);
}

// CHECK-LABEL: @test_div_fixup_f32
// CHECK: call float @llvm.amdgcn.div.fixup.f32
void test_div_fixup_f32(global float* out, float a, float b, float c)
{
  *out = __builtin_amdgcn_div_fixupf(a, b, c);
}

// CHECK-LABEL: @test_div_fixup_f64
// CHECK: call double @llvm.amdgcn.div.fixup.f64
void test_div_fixup_f64(global double* out, double a, double b, double c)
{
  *out = __builtin_amdgcn_div_fixup(a, b, c);
}

// CHECK-LABEL: @test_trig_preop_f32
// CHECK: call float @llvm.amdgcn.trig.preop.f32
void test_trig_preop_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgcn_trig_preopf(a, b);
}

// CHECK-LABEL: @test_trig_preop_f64
// CHECK: call double @llvm.amdgcn.trig.preop.f64
void test_trig_preop_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgcn_trig_preop(a, b);
}

// CHECK-LABEL: @test_rcp_f32
// CHECK: call float @llvm.amdgcn.rcp.f32
void test_rcp_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_rcpf(a);
}

// CHECK-LABEL: @test_rcp_f64
// CHECK: call double @llvm.amdgcn.rcp.f64
void test_rcp_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_rcp(a);
}

// CHECK-LABEL: @test_rsq_f32
// CHECK: call float @llvm.amdgcn.rsq.f32
void test_rsq_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_rsqf(a);
}

// CHECK-LABEL: @test_rsq_f64
// CHECK: call double @llvm.amdgcn.rsq.f64
void test_rsq_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_rsq(a);
}

// CHECK-LABEL: @test_rsq_clamped_f32
// CHECK: call float @llvm.amdgcn.rsq.clamped.f32
void test_rsq_clamped_f32(global float* out, float a)
{
  *out = __builtin_amdgcn_rsq_clampedf(a);
}

// CHECK-LABEL: @test_rsq_clamped_f64
// CHECK: call double @llvm.amdgcn.rsq.clamped.f64
void test_rsq_clamped_f64(global double* out, double a)
{
  *out = __builtin_amdgcn_rsq_clamped(a);
}

// CHECK-LABEL: @test_ldexp_f32
// CHECK: call float @llvm.amdgcn.ldexp.f32
void test_ldexp_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgcn_ldexpf(a, b);
}

// CHECK-LABEL: @test_ldexp_f64
// CHECK: call double @llvm.amdgcn.ldexp.f64
void test_ldexp_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgcn_ldexp(a, b);
}

// CHECK-LABEL: @test_class_f32
// CHECK: call i1 @llvm.amdgcn.class.f32
void test_class_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgcn_classf(a, b);
}

// CHECK-LABEL: @test_class_f64
// CHECK: call i1 @llvm.amdgcn.class.f64
void test_class_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgcn_class(a, b);
}

// CHECK-LABEL: @test_s_barrier
// CHECK: call void @llvm.amdgcn.s.barrier(
void test_s_barrier()
{
  __builtin_amdgcn_s_barrier();
}

// Legacy intrinsics with AMDGPU prefix

// CHECK-LABEL: @test_legacy_rsq_f32
// CHECK: call float @llvm.amdgcn.rsq.f32
void test_legacy_rsq_f32(global float* out, float a)
{
  *out = __builtin_amdgpu_rsqf(a);
}

// CHECK-LABEL: @test_legacy_rsq_f64
// CHECK: call double @llvm.amdgcn.rsq.f64
void test_legacy_rsq_f64(global double* out, double a)
{
  *out = __builtin_amdgpu_rsq(a);
}

// CHECK-LABEL: @test_legacy_ldexp_f32
// CHECK: call float @llvm.amdgcn.ldexp.f32
void test_legacy_ldexp_f32(global float* out, float a, int b)
{
  *out = __builtin_amdgpu_ldexpf(a, b);
}

// CHECK-LABEL: @test_legacy_ldexp_f64
// CHECK: call double @llvm.amdgcn.ldexp.f64
void test_legacy_ldexp_f64(global double* out, double a, int b)
{
  *out = __builtin_amdgpu_ldexp(a, b);
}
