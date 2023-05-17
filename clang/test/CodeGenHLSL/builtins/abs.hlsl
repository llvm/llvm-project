// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -D__HLSL_ENABLE_16_BIT -o - | FileCheck %s --check-prefix=NO_HALF

using hlsl::abs;

// CHECK: define noundef i16 @
// CHECK: call i16 @llvm.abs.i16(
int16_t test_abs_int16_t ( int16_t p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <2 x i16> @
// CHECK: call <2 x i16> @llvm.abs.v2i16(
int16_t2 test_abs_int16_t2 ( int16_t2 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <3 x i16> @
// CHECK: call <3 x i16> @llvm.abs.v3i16(
int16_t3 test_abs_int16_t3 ( int16_t3 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <4 x i16> @
// CHECK: call <4 x i16> @llvm.abs.v4i16(
int16_t4 test_abs_int16_t4 ( int16_t4 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef half @
// CHECK: call half @llvm.fabs.f16(
// NO_HALF: define noundef float @"?test_abs_half@@YA$halff@$halff@@Z"(
// NO_HALF: call float @llvm.fabs.f32(float %0)
half test_abs_half ( half p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <2 x half> @
// CHECK: call <2 x half> @llvm.fabs.v2f16(
// NO_HALF: define noundef <2 x float> @"?test_abs_half2@@YAT?$__vector@$halff@$01@__clang@@T12@@Z"(
// NO_HALF: call <2 x float> @llvm.fabs.v2f32(
half2 test_abs_half2 ( half2 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <3 x half> @
// CHECK: call <3 x half> @llvm.fabs.v3f16(
// NO_HALF: define noundef <3 x float> @"?test_abs_half3@@YAT?$__vector@$halff@$02@__clang@@T12@@Z"(
// NO_HALF: call <3 x float> @llvm.fabs.v3f32(
half3 test_abs_half3 ( half3 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <4 x half> @
// CHECK: call <4 x half> @llvm.fabs.v4f16(
// NO_HALF: define noundef <4 x float> @"?test_abs_half4@@YAT?$__vector@$halff@$03@__clang@@T12@@Z"(
// NO_HALF: call <4 x float> @llvm.fabs.v4f32(
half4 test_abs_half4 ( half4 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef i32 @
// CHECK: call i32 @llvm.abs.i32(
// NO_HALF: define noundef i32 @"?test_abs_int@@YAHH@Z"
int test_abs_int ( int p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <2 x i32> @
// CHECK: call <2 x i32> @llvm.abs.v2i32(
int2 test_abs_int2 ( int2 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <3 x i32> @
// CHECK: call <3 x i32> @llvm.abs.v3i32(
int3 test_abs_int3 ( int3 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <4 x i32> @
// CHECK: call <4 x i32> @llvm.abs.v4i32(
int4 test_abs_int4 ( int4 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef float @
// CHECK: call float @llvm.fabs.f32(
float test_abs_float ( float p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <2 x float> @
// CHECK: call <2 x float> @llvm.fabs.v2f32(
float2 test_abs_float2 ( float2 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <3 x float> @
// CHECK: call <3 x float> @llvm.fabs.v3f32(
float3 test_abs_float3 ( float3 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <4 x float> @
// CHECK: call <4 x float> @llvm.fabs.v4f32(
float4 test_abs_float4 ( float4 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef i64 @
// CHECK: call i64 @llvm.abs.i64(
int64_t test_abs_int64_t ( int64_t p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <2 x i64> @
// CHECK: call <2 x i64> @llvm.abs.v2i64(
int64_t2 test_abs_int64_t2 ( int64_t2 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <3 x i64> @
// CHECK: call <3 x i64> @llvm.abs.v3i64(
int64_t3 test_abs_int64_t3 ( int64_t3 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <4 x i64> @
// CHECK: call <4 x i64> @llvm.abs.v4i64(
int64_t4 test_abs_int64_t4 ( int64_t4 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef double @
// CHECK: call double @llvm.fabs.f64(
double test_abs_double ( double p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <2 x double> @
// CHECK: call <2 x double> @llvm.fabs.v2f64(
double2 test_abs_double2 ( double2 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <3 x double> @
// CHECK: call <3 x double> @llvm.fabs.v3f64(
double3 test_abs_double3 ( double3 p0 ) {
  return abs ( p0 );
}
// CHECK: define noundef <4 x double> @
// CHECK: call <4 x double> @llvm.fabs.v4f64(
double4 test_abs_double4 ( double4 p0 ) {
  return abs ( p0 );
}
