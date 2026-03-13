// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -Wdeprecated-declarations -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DFNATTRS="hidden spir_func noundef nofpclass(nan inf)"
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple spirv-unknown-vulkan-compute %s  \
// RUN:   -verify -verify-ignore-unexpected=note

// CHECK: define [[FNATTRS]] float @_Z15test_tan_doubled(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} double %{{.*}} to float
// CHECK:    [[V3:%.*]] = call {{.*}} float @llvm.tan.f32(float [[CONVI]])
// CHECK:    ret float [[V3]]
float test_tan_double ( double p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x 64 bit API lowering for tan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <2 x float> @_Z16test_tan_double2Dv2_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <2 x float> @llvm.tan.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[V3]]
float2 test_tan_double2 ( double2 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x 64 bit API lowering for tan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <3 x float> @_Z16test_tan_double3Dv3_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <3 x float> @llvm.tan.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[V3]]
float3 test_tan_double3 ( double3 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x 64 bit API lowering for tan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <4 x float> @_Z16test_tan_double4Dv4_d(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <4 x float> @llvm.tan.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[V3]]
float4 test_tan_double4 ( double4 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x 64 bit API lowering for tan is deprecated. Explicitly cast parameters to 32 or 16 bit types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] float @_Z12test_tan_inti(
// CHECK:    [[CONVI:%.*]] = sitofp i32 %{{.*}} to float
// CHECK:    [[V3:%.*]] = call {{.*}} float @llvm.tan.f32(float [[CONVI]])
// CHECK:    ret float [[V3]]
float test_tan_int ( int p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <2 x float> @_Z13test_tan_int2Dv2_i(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <2 x float> @llvm.tan.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[V3]]
float2 test_tan_int2 ( int2 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <3 x float> @_Z13test_tan_int3Dv3_i(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <3 x float> @llvm.tan.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[V3]]
float3 test_tan_int3 ( int3 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <4 x float> @_Z13test_tan_int4Dv4_i(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <4 x float> @llvm.tan.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[V3]]
float4 test_tan_int4 ( int4 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] float @_Z13test_tan_uintj(
// CHECK:    [[CONVI:%.*]] = uitofp i32 %{{.*}} to float
// CHECK:    [[V3:%.*]] = call {{.*}} float @llvm.tan.f32(float [[CONVI]])
// CHECK:    ret float [[V3]]
float test_tan_uint ( uint p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <2 x float> @_Z14test_tan_uint2Dv2_j(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <2 x float> @llvm.tan.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[V3]]
float2 test_tan_uint2 ( uint2 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <3 x float> @_Z14test_tan_uint3Dv3_j(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <3 x float> @llvm.tan.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[V3]]
float3 test_tan_uint3 ( uint3 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <4 x float> @_Z14test_tan_uint4Dv4_j(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <4 x float> @llvm.tan.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[V3]]
float4 test_tan_uint4 ( uint4 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] float @_Z16test_tan_int64_tl(
// CHECK:    [[CONVI:%.*]] = sitofp i64 %{{.*}} to float
// CHECK:    [[V3:%.*]] = call {{.*}} float @llvm.tan.f32(float [[CONVI]])
// CHECK:    ret float [[V3]]
float test_tan_int64_t ( int64_t p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <2 x float> @_Z17test_tan_int64_t2Dv2_l(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <2 x float> @llvm.tan.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[V3]]
float2 test_tan_int64_t2 ( int64_t2 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <3 x float> @_Z17test_tan_int64_t3Dv3_l(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <3 x float> @llvm.tan.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[V3]]
float3 test_tan_int64_t3 ( int64_t3 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <4 x float> @_Z17test_tan_int64_t4Dv4_l(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <4 x float> @llvm.tan.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[V3]]
float4 test_tan_int64_t4 ( int64_t4 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] float @_Z17test_tan_uint64_tm(
// CHECK:    [[CONVI:%.*]] = uitofp i64 %{{.*}} to float
// CHECK:    [[V3:%.*]] = call {{.*}} float @llvm.tan.f32(float [[CONVI]])
// CHECK:    ret float [[V3]]
float test_tan_uint64_t ( uint64_t p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <2 x float> @_Z18test_tan_uint64_t2Dv2_m(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <2 x float> @llvm.tan.v2f32(<2 x float> [[CONVI]])
// CHECK:    ret <2 x float> [[V3]]
float2 test_tan_uint64_t2 ( uint64_t2 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <3 x float> @_Z18test_tan_uint64_t3Dv3_m(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <3 x float> @llvm.tan.v3f32(<3 x float> [[CONVI]])
// CHECK:    ret <3 x float> [[V3]]
float3 test_tan_uint64_t3 ( uint64_t3 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}

// CHECK: define [[FNATTRS]] <4 x float> @_Z18test_tan_uint64_t4Dv4_m(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V3:%.*]] = call {{.*}} <4 x float> @llvm.tan.v4f32(<4 x float> [[CONVI]])
// CHECK:    ret <4 x float> [[V3]]
float4 test_tan_uint64_t4 ( uint64_t4 p0 ) {
// expected-warning@+1 {{'tan' is deprecated: In 202x int lowering for tan is deprecated. Explicitly cast parameters to float types.}}
  return tan ( p0 );
}
