// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DTARGET=dx -DFNATTRS="hidden noundef nofpclass(nan inf)"
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DTARGET=spv -DFNATTRS="hidden spir_func noundef nofpclass(nan inf)"

// CHECK: define [[FNATTRS]] float @_Z19test_radians_doubled(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} float @llvm.[[TARGET]].radians.f32(float [[CONVI]])
// CHECK:    ret float [[HLSLRADIANSI]]
float test_radians_double(double p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z20test_radians_double2Dv2_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].radians.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[HLSLRADIANSI]]
float2 test_radians_double2(double2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z20test_radians_double3Dv3_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].radians.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[HLSLRADIANSI]]
float3 test_radians_double3(double3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z20test_radians_double4Dv4_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].radians.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[HLSLRADIANSI]]
float4 test_radians_double4(double4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @_Z16test_radians_inti(
// CHECK:    [[CONVI:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} float @llvm.[[TARGET]].radians.f32(float [[CONVI]])
// CHECK:    ret float [[HLSLRADIANSI]]
float test_radians_int(int p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z17test_radians_int2Dv2_i(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].radians.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[HLSLRADIANSI]]
float2 test_radians_int2(int2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z17test_radians_int3Dv3_i(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].radians.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[HLSLRADIANSI]]
float3 test_radians_int3(int3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z17test_radians_int4Dv4_i(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].radians.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[HLSLRADIANSI]]
float4 test_radians_int4(int4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @_Z17test_radians_uintj(
// CHECK:    [[CONVI:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} float @llvm.[[TARGET]].radians.f32(float [[CONVI]])
// CHECK:    ret float [[HLSLRADIANSI]]
float test_radians_uint(uint p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z18test_radians_uint2Dv2_j(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].radians.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[HLSLRADIANSI]]
float2 test_radians_uint2(uint2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z18test_radians_uint3Dv3_j(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].radians.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[HLSLRADIANSI]]
float3 test_radians_uint3(uint3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z18test_radians_uint4Dv4_j(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].radians.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[HLSLRADIANSI]]
float4 test_radians_uint4(uint4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @_Z20test_radians_int64_tl(
// CHECK:    [[CONVI:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} float @llvm.[[TARGET]].radians.f32(float [[CONVI]])
// CHECK:    ret float [[HLSLRADIANSI]]
float test_radians_int64_t(int64_t p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z21test_radians_int64_t2Dv2_l(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].radians.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[HLSLRADIANSI]]
float2 test_radians_int64_t2(int64_t2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z21test_radians_int64_t3Dv3_l(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].radians.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[HLSLRADIANSI]]
float3 test_radians_int64_t3(int64_t3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z21test_radians_int64_t4Dv4_l(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].radians.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[HLSLRADIANSI]]
float4 test_radians_int64_t4(int64_t4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @_Z21test_radians_uint64_tm(
// CHECK:    [[CONVI:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} float @llvm.[[TARGET]].radians.f32(float [[CONVI]])
// CHECK:    ret float [[HLSLRADIANSI]]
float test_radians_uint64_t(uint64_t p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z22test_radians_uint64_t2Dv2_m(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <2 x float> @llvm.[[TARGET]].radians.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[HLSLRADIANSI]]
float2 test_radians_uint64_t2(uint64_t2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z22test_radians_uint64_t3Dv3_m(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <3 x float> @llvm.[[TARGET]].radians.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[HLSLRADIANSI]]
float3 test_radians_uint64_t3(uint64_t3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z22test_radians_uint64_t4Dv4_m(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[HLSLRADIANSI:%.*]] = call {{.*}} <4 x float> @llvm.[[TARGET]].radians.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[HLSLRADIANSI]]
float4 test_radians_uint64_t4(uint64_t4 p0) { return radians(p0); }
