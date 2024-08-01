// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: call half @llvm.dx.saturate.f16(
// NO_HALF: define noundef float @"?test_saturate_half
// NO_HALF: call float @llvm.dx.saturate.f32(
half test_saturate_half(half p0) { return saturate(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: call <2 x half> @llvm.dx.saturate.v2f16
// NO_HALF: define noundef <2 x float> @"?test_saturate_half2
// NO_HALF: call <2 x float> @llvm.dx.saturate.v2f32(
half2 test_saturate_half2(half2 p0) { return saturate(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: call <3 x half> @llvm.dx.saturate.v3f16
// NO_HALF: define noundef <3 x float> @"?test_saturate_half3
// NO_HALF: call <3 x float> @llvm.dx.saturate.v3f32(
half3 test_saturate_half3(half3 p0) { return saturate(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: call <4 x half> @llvm.dx.saturate.v4f16
// NO_HALF: define noundef <4 x float> @"?test_saturate_half4
// NO_HALF: call <4 x float> @llvm.dx.saturate.v4f32(
half4 test_saturate_half4(half4 p0) { return saturate(p0); }

// CHECK: define noundef float @"?test_saturate_float
// CHECK: call float @llvm.dx.saturate.f32(
float test_saturate_float(float p0) { return saturate(p0); }
// CHECK: define noundef <2 x float> @"?test_saturate_float2
// CHECK: call <2 x float> @llvm.dx.saturate.v2f32
float2 test_saturate_float2(float2 p0) { return saturate(p0); }
// CHECK: define noundef <3 x float> @"?test_saturate_float3
// CHECK: call <3 x float> @llvm.dx.saturate.v3f32
float3 test_saturate_float3(float3 p0) { return saturate(p0); }
// CHECK: define noundef <4 x float> @"?test_saturate_float4
// CHECK: call <4 x float> @llvm.dx.saturate.v4f32
float4 test_saturate_float4(float4 p0) { return saturate(p0); }

// CHECK: define noundef double @
// CHECK: call double @llvm.dx.saturate.f64(
double test_saturate_double(double p0) { return saturate(p0); }
// CHECK: define noundef <2 x double> @
// CHECK: call <2 x double> @llvm.dx.saturate.v2f64
double2 test_saturate_double2(double2 p0) { return saturate(p0); }
// CHECK: define noundef <3 x double> @
// CHECK: call <3 x double> @llvm.dx.saturate.v3f64
double3 test_saturate_double3(double3 p0) { return saturate(p0); }
// CHECK: define noundef <4 x double> @
// CHECK: call <4 x double> @llvm.dx.saturate.v4f64
double4 test_saturate_double4(double4 p0) { return saturate(p0); }
