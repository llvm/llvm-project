// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm  \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK -DSPIR_FUNC="" 
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm  \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK -DSPIR_FUNC="spir_func "


// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_atan2_double3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_double3x4 (double3x4 p0, double3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_atan2_uint3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_uint3x4 (uint3x4 p0, uint3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_atan2_int3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_int3x4 (int3x4 p0, int3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_atan2_int64_t3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_int64_t3x4 (int64_t3x4 p0, int64_t3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define hidden [[SPIR_FUNC]]noundef nofpclass(nan inf) <12 x float> @_{{.*}}test_atan2_uint64_t3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_uint64_t3x4 (uint64_t3x4 p0, uint64_t3x4 p1) {
  return atan2(p0, p1);
}
