// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm  \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK -DSPIR_FUNC=""
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm  \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK -DSPIR_FUNC="spir_func "

using namespace hlsl;

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_lerp_double3x4{{.*}}(
// CHECK:    fptrunc [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x double> %{{.*}} to <12 x float>
// CHECK:    fptrunc [[FMF]] <12 x double> %{{.*}} to <12 x float>
// CHECK:    fptrunc [[FMF]] <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[SUB:%.*]] = fsub [[FMF]] <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul [[FMF]] <12 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd [[FMF]] <12 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <12 x float> [[ADD]]
float3x4 test_lerp_double3x4 (double3x4 p0, double3x4 p1, double3x4 p2) {
  return lerp(p0, p1, p2);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_lerp_uint3x4{{.*}}(
// CHECK:    uitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    uitofp [[FMF]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    uitofp [[FMF]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[SUB:%.*]] = fsub [[FMF]] <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul [[FMF]] <12 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd [[FMF]] <12 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <12 x float> [[ADD]]
float3x4 test_lerp_uint3x4 (uint3x4 p0, uint3x4 p1, uint3x4 p2) {
  return lerp(p0, p1, p2);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_lerp_int3x4{{.*}}(
// CHECK:    sitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    sitofp [[FMF]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    sitofp [[FMF]] <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[SUB:%.*]] = fsub [[FMF]] <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul [[FMF]] <12 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd [[FMF]] <12 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <12 x float> [[ADD]]
float3x4 test_lerp_int3x4 (int3x4 p0, int3x4 p1, int3x4 p2) {
  return lerp(p0, p1, p2);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_lerp_int64_t3x4{{.*}}(
// CHECK:    sitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    sitofp [[FMF]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    sitofp [[FMF]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[SUB:%.*]] = fsub [[FMF]] <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul [[FMF]] <12 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd [[FMF]] <12 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <12 x float> [[ADD]]
float3x4 test_lerp_int64_t3x4 (int64_t3x4 p0, int64_t3x4 p1, int64_t3x4 p2) {
  return lerp(p0, p1, p2);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_lerp_uint64_t3x4{{.*}}(
// CHECK:    uitofp [[FMF:reassoc nnan ninf nsz arcp afn]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    uitofp [[FMF]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    uitofp [[FMF]] <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[SUB:%.*]] = fsub [[FMF]] <12 x float> %{{.*}}, %{{.*}}
// CHECK:    [[MUL:%.*]] = fmul [[FMF]] <12 x float> %{{.*}}, [[SUB]]
// CHECK:    [[ADD:%.*]] = fadd [[FMF]] <12 x float> %{{.*}}, [[MUL]]
// CHECK:    ret <12 x float> [[ADD]]
float3x4 test_lerp_uint64_t3x4 (uint64_t3x4 p0, uint64_t3x4 p1, uint64_t3x4 p2) {
  return lerp(p0, p1, p2);
}
