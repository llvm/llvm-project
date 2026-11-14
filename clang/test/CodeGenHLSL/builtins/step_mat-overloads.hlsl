// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm  \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK -DSPIR_FUNC=""
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm  \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK -DSPIR_FUNC="spir_func "


// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_step_double3x4{{.*}}(
// CHECK:    fptrunc [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x double> %{{.*}} to <12 x float>
// CHECK:    fptrunc [[FMF]] <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[CMP:%.*]] = fcmp [[FMF]] olt <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SEL:%.*]] = select [[FMF]] <12 x i1> [[CMP]], <12 x float> zeroinitializer, <12 x float> splat (float 1.000000e+00)
// CHECK:    ret <12 x float> [[SEL]]
float3x4 test_step_double3x4 (double3x4 p0, double3x4 p1) {
  return step(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_step_uint3x4{{.*}}(
// CHECK:    uitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    uitofp [[FMF]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CMP:%.*]] = fcmp [[FMF]] olt <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SEL:%.*]] = select [[FMF]] <12 x i1> [[CMP]], <12 x float> zeroinitializer, <12 x float> splat (float 1.000000e+00)
// CHECK:    ret <12 x float> [[SEL]]
float3x4 test_step_uint3x4 (uint3x4 p0, uint3x4 p1) {
  return step(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_step_int3x4{{.*}}(
// CHECK:    sitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    sitofp [[FMF]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CMP:%.*]] = fcmp [[FMF]] olt <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SEL:%.*]] = select [[FMF]] <12 x i1> [[CMP]], <12 x float> zeroinitializer, <12 x float> splat (float 1.000000e+00)
// CHECK:    ret <12 x float> [[SEL]]
float3x4 test_step_int3x4 (int3x4 p0, int3x4 p1) {
  return step(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_step_int64_t3x4{{.*}}(
// CHECK:    sitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    sitofp [[FMF]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CMP:%.*]] = fcmp [[FMF]] olt <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SEL:%.*]] = select [[FMF]] <12 x i1> [[CMP]], <12 x float> zeroinitializer, <12 x float> splat (float 1.000000e+00)
// CHECK:    ret <12 x float> [[SEL]]
float3x4 test_step_int64_t3x4 (int64_t3x4 p0, int64_t3x4 p1) {
  return step(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_step_uint64_t3x4{{.*}}(
// CHECK:    uitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    uitofp [[FMF]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CMP:%.*]] = fcmp [[FMF]] olt <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[SEL:%.*]] = select [[FMF]] <12 x i1> [[CMP]], <12 x float> zeroinitializer, <12 x float> splat (float 1.000000e+00)
// CHECK:    ret <12 x float> [[SEL]]
float3x4 test_step_uint64_t3x4 (uint64_t3x4 p0, uint64_t3x4 p1) {
  return step(p0, p1);
}
