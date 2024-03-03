// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes  -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: call half @llvm.sin.f16(
// NO_HALF: define noundef float @"?test_sin_half@@YA$halff@$halff@@Z"(
// NO_HALF: call float @llvm.sin.f32(
half test_sin_half(half p0) { return sin(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: call <2 x half> @llvm.sin.v2f16
// NO_HALF: define noundef <2 x float> @"?test_sin_half2
// NO_HALF: call <2 x float> @llvm.sin.v2f32(
half2 test_sin_half2(half2 p0) { return sin(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: call <3 x half> @llvm.sin.v3f16
// NO_HALF: define noundef <3 x float> @"?test_sin_half3
// NO_HALF: call <3 x float> @llvm.sin.v3f32(
half3 test_sin_half3(half3 p0) { return sin(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: call <4 x half> @llvm.sin.v4f16
// NO_HALF: define noundef <4 x float> @"?test_sin_half4
// NO_HALF: call <4 x float> @llvm.sin.v4f32(
half4 test_sin_half4(half4 p0) { return sin(p0); }

// CHECK: define noundef float @
// CHECK: call float @llvm.sin.f32(
float test_sin_float(float p0) { return sin(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: call <2 x float> @llvm.sin.v2f32
float2 test_sin_float2(float2 p0) { return sin(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: call <3 x float> @llvm.sin.v3f32
float3 test_sin_float3(float3 p0) { return sin(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: call <4 x float> @llvm.sin.v4f32
float4 test_sin_float4(float4 p0) { return sin(p0); }
