// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

using hlsl::ceil;

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: call half @llvm.ceil.f16(
// NO_HALF: define noundef float @"?test_ceil_half@@YA$halff@$halff@@Z"(
// NO_HALF: call float @llvm.ceil.f32(float %0)
half test_ceil_half(half p0) { return ceil(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: call <2 x half> @llvm.ceil.v2f16(
// NO_HALF: define noundef <2 x float> @"?test_ceil_half2@@YAT?$__vector@$halff@$01@__clang@@T12@@Z"(
// NO_HALF: call <2 x float> @llvm.ceil.v2f32(
half2 test_ceil_half2(half2 p0) { return ceil(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: call <3 x half> @llvm.ceil.v3f16(
// NO_HALF: define noundef <3 x float> @"?test_ceil_half3@@YAT?$__vector@$halff@$02@__clang@@T12@@Z"(
// NO_HALF: call <3 x float> @llvm.ceil.v3f32(
half3 test_ceil_half3(half3 p0) { return ceil(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: call <4 x half> @llvm.ceil.v4f16(
// NO_HALF: define noundef <4 x float> @"?test_ceil_half4@@YAT?$__vector@$halff@$03@__clang@@T12@@Z"(
// NO_HALF: call <4 x float> @llvm.ceil.v4f32(
half4 test_ceil_half4(half4 p0) { return ceil(p0); }

// CHECK: define noundef float @
// CHECK: call float @llvm.ceil.f32(
float test_ceil_float(float p0) { return ceil(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: call <2 x float> @llvm.ceil.v2f32(
float2 test_ceil_float2(float2 p0) { return ceil(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: call <3 x float> @llvm.ceil.v3f32(
float3 test_ceil_float3(float3 p0) { return ceil(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: call <4 x float> @llvm.ceil.v4f32(
float4 test_ceil_float4(float4 p0) { return ceil(p0); }

// CHECK: define noundef double @
// CHECK: call double @llvm.ceil.f64(
double test_ceil_double(double p0) { return ceil(p0); }
// CHECK: define noundef <2 x double> @
// CHECK: call <2 x double> @llvm.ceil.v2f64(
double2 test_ceil_double2(double2 p0) { return ceil(p0); }
// CHECK: define noundef <3 x double> @
// CHECK: call <3 x double> @llvm.ceil.v3f64(
double3 test_ceil_double3(double3 p0) { return ceil(p0); }
// CHECK: define noundef <4 x double> @
// CHECK: call <4 x double> @llvm.ceil.v4f64(
double4 test_ceil_double4(double4 p0) { return ceil(p0); }
