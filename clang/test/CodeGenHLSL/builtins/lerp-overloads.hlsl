// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple  dxil-pc-shadermodel6.3-library %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK -DFNATTRS="noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-compute %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTARGET=spv

// CHECK: define [[FNATTRS]] float @_Z16test_lerp_doubled(
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[LERP:%.*]] = call {{.*}} float @llvm.[[TARGET]].lerp.f32(float [[CONV0]], float [[CONV1]], float [[CONV2]])
// CHECK:    ret float [[LERP]]
float test_lerp_double(double p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <2 x float> @_Z17test_lerp_double2Dv2_d(
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]], <2 x float> [[CONV2]])
// CHECK:    ret <2 x float> [[LERP]]
float2 test_lerp_double2(double2 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <3 x float> @_Z17test_lerp_double3Dv3_d(
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]], <3 x float> [[CONV2]])
// CHECK:    ret <3 x float> [[LERP]]
float3 test_lerp_double3(double3 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <4 x float> @_Z17test_lerp_double4Dv4_d(
// CHECK:    [[CONV0:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]], <4 x float> [[CONV2]])
// CHECK:    ret <4 x float> [[LERP]]
float4 test_lerp_double4(double4 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] float @_Z13test_lerp_inti(
// CHECK:    [[CONV0:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    [[LERP:%.*]] = call {{.*}} float @llvm.[[TARGET]].lerp.f32(float [[CONV0]], float [[CONV1]], float [[CONV2]])
// CHECK:    ret float [[LERP]]
float test_lerp_int(int p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <2 x float> @_Z14test_lerp_int2Dv2_i(
// CHECK:    [[CONV0:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]], <2 x float> [[CONV2]])
// CHECK:    ret <2 x float> [[LERP]]
float2 test_lerp_int2(int2 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <3 x float> @_Z14test_lerp_int3Dv3_i(
// CHECK:    [[CONV0:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]], <3 x float> [[CONV2]])
// CHECK:    ret <3 x float> [[LERP]]
float3 test_lerp_int3(int3 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <4 x float> @_Z14test_lerp_int4Dv4_i(
// CHECK:    [[CONV0:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]], <4 x float> [[CONV2]])
// CHECK:    ret <4 x float> [[LERP]]
float4 test_lerp_int4(int4 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] float @_Z14test_lerp_uintj(
// CHECK:    [[CONV0:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    [[LERP:%.*]] = call {{.*}} float @llvm.[[TARGET]].lerp.f32(float [[CONV0]], float [[CONV1]], float [[CONV2]])
// CHECK:    ret float [[LERP]]
float test_lerp_uint(uint p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <2 x float> @_Z15test_lerp_uint2Dv2_j(
// CHECK:    [[CONV0:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]], <2 x float> [[CONV2]])
// CHECK:    ret <2 x float> [[LERP]]
float2 test_lerp_uint2(uint2 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <3 x float> @_Z15test_lerp_uint3Dv3_j(
// CHECK:    [[CONV0:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]], <3 x float> [[CONV2]])
// CHECK:    ret <3 x float> [[LERP]]
float3 test_lerp_uint3(uint3 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <4 x float> @_Z15test_lerp_uint4Dv4_j(
// CHECK:    [[CONV0:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]], <4 x float> [[CONV2]])
// CHECK:    ret <4 x float> [[LERP]]
float4 test_lerp_uint4(uint4 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] float @_Z17test_lerp_int64_tl(
// CHECK:    [[CONV0:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    [[LERP:%.*]] = call {{.*}} float @llvm.[[TARGET]].lerp.f32(float [[CONV0]], float [[CONV1]], float [[CONV2]])
// CHECK:    ret float [[LERP]]
float test_lerp_int64_t(int64_t p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <2 x float> @_Z18test_lerp_int64_t2Dv2_l(
// CHECK:    [[CONV0:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]], <2 x float> [[CONV2]])
// CHECK:    ret <2 x float> [[LERP]]
float2 test_lerp_int64_t2(int64_t2 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <3 x float> @_Z18test_lerp_int64_t3Dv3_l(
// CHECK:    [[CONV0:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]], <3 x float> [[CONV2]])
// CHECK:    ret <3 x float> [[LERP]]
float3 test_lerp_int64_t3(int64_t3 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <4 x float> @_Z18test_lerp_int64_t4Dv4_l(
// CHECK:    [[CONV0:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]], <4 x float> [[CONV2]])
// CHECK:    ret <4 x float> [[LERP]]
float4 test_lerp_int64_t4(int64_t4 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] float @_Z18test_lerp_uint64_tm(
// CHECK:    [[CONV0:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    [[CONV1:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    [[CONV2:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    [[LERP:%.*]] = call {{.*}} float @llvm.[[TARGET]].lerp.f32(float [[CONV0]], float [[CONV1]], float [[CONV2]])
// CHECK:    ret float [[LERP]]
float test_lerp_uint64_t(uint64_t p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <2 x float> @_Z19test_lerp_uint64_t2Dv2_m(
// CHECK:    [[CONV0:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV2:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]], <2 x float> [[CONV2]])
// CHECK:    ret <2 x float> [[LERP]]
float2 test_lerp_uint64_t2(uint64_t2 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <3 x float> @_Z19test_lerp_uint64_t3Dv3_m(
// CHECK:    [[CONV0:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV2:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]], <3 x float> [[CONV2]])
// CHECK:    ret <3 x float> [[LERP]]
float3 test_lerp_uint64_t3(uint64_t3 p0) { return lerp(p0, p0, p0); }

// CHECK: define [[FNATTRS]] <4 x float> @_Z19test_lerp_uint64_t4Dv4_m(
// CHECK:    [[CONV0:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV2:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[LERP:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]], <4 x float> [[CONV2]])
// CHECK:    ret <4 x float> [[LERP]]
float4 test_lerp_uint64_t4(uint64_t4 p0) { return lerp(p0, p0, p0); }
