// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK -DFNATTRS="hidden noundef nofpclass(nan inf)"

using hlsl::ceil;

// CHECK: define [[FNATTRS]] float @_Z16test_ceil_doubled(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[v2:%.*]] = call {{.*}} float @llvm.ceil.f32(float [[CONVI]])
// CHECK:    ret float [[v2]]
float test_ceil_double(double p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z17test_ceil_double2Dv2_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[v2:%.*]] = call {{.*}} <2 x float> @llvm.ceil.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[v2]]
float2 test_ceil_double2(double2 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z17test_ceil_double3Dv3_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[v2:%.*]] = call {{.*}} <3 x float> @llvm.ceil.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[v2]]
float3 test_ceil_double3(double3 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z17test_ceil_double4Dv4_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[v2:%.*]] = call {{.*}} <4 x float> @llvm.ceil.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[v2]]
float4 test_ceil_double4(double4 p0) { return ceil(p0); }

// CHECK: define [[FNATTRS]] float @_Z13test_ceil_inti(
// CHECK:    [[CONVI:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    ret float [[CONVI]]
float test_ceil_int(int p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z14test_ceil_int2Dv2_i(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    ret <2 x float> [[CONVI]]
float2 test_ceil_int2(int2 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z14test_ceil_int3Dv3_i(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    ret <3 x float> [[CONVI]]
float3 test_ceil_int3(int3 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z14test_ceil_int4Dv4_i(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    ret <4 x float> [[CONVI]]
float4 test_ceil_int4(int4 p0) { return ceil(p0); }

// CHECK: define [[FNATTRS]] float @_Z14test_ceil_uintj(
// CHECK:    [[CONVI:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    ret float [[CONVI]]
float test_ceil_uint(uint p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z15test_ceil_uint2Dv2_j(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    ret <2 x float> [[CONVI]]
float2 test_ceil_uint2(uint2 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z15test_ceil_uint3Dv3_j(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    ret <3 x float> [[CONVI]]
float3 test_ceil_uint3(uint3 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z15test_ceil_uint4Dv4_j(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    ret <4 x float> [[CONVI]]
float4 test_ceil_uint4(uint4 p0) { return ceil(p0); }

// CHECK: define [[FNATTRS]] float @_Z17test_ceil_int64_tl(
// CHECK:    [[CONVI:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    ret float [[CONVI]]
float test_ceil_int64_t(int64_t p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z18test_ceil_int64_t2Dv2_l(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    ret <2 x float> [[CONVI]]
float2 test_ceil_int64_t2(int64_t2 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z18test_ceil_int64_t3Dv3_l(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    ret <3 x float> [[CONVI]]
float3 test_ceil_int64_t3(int64_t3 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z18test_ceil_int64_t4Dv4_l(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    ret <4 x float> [[CONVI]]
float4 test_ceil_int64_t4(int64_t4 p0) { return ceil(p0); }

// CHECK: define [[FNATTRS]] float @_Z18test_ceil_uint64_tm(
// CHECK:    [[CONVI:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    ret float [[CONVI]]
float test_ceil_uint64_t(uint64_t p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @_Z19test_ceil_uint64_t2Dv2_m(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    ret <2 x float> [[CONVI]]
float2 test_ceil_uint64_t2(uint64_t2 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @_Z19test_ceil_uint64_t3Dv3_m(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    ret <3 x float> [[CONVI]]
float3 test_ceil_uint64_t3(uint64_t3 p0) { return ceil(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @_Z19test_ceil_uint64_t4Dv4_m(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    ret <4 x float> [[CONVI]]
float4 test_ceil_uint64_t4(uint64_t4 p0) { return ceil(p0); }
