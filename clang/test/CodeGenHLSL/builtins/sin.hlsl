// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef half @_Z13test_sin_half
// NATIVE_HALF: call half @llvm.sin.f16(
// NO_HALF-LABEL: define noundef float @_Z13test_sin_half
// NO_HALF: call float @llvm.sin.f32(
half test_sin_half(half p0) { return sin(p0); }
// NATIVE_HALF-LABEL: define noundef <2 x half> @_Z14test_sin_half2
// NATIVE_HALF: call <2 x half> @llvm.sin.v2f16
// NO_HALF-LABEL: define noundef <2 x float> @_Z14test_sin_half2
// NO_HALF: call <2 x float> @llvm.sin.v2f32(
half2 test_sin_half2(half2 p0) { return sin(p0); }
// NATIVE_HALF-LABEL: define noundef <3 x half> @_Z14test_sin_half3
// NATIVE_HALF: call <3 x half> @llvm.sin.v3f16
// NO_HALF-LABEL: define noundef <3 x float> @_Z14test_sin_half3
// NO_HALF: call <3 x float> @llvm.sin.v3f32(
half3 test_sin_half3(half3 p0) { return sin(p0); }
// NATIVE_HALF-LABEL: define noundef <4 x half> @_Z14test_sin_half4
// NATIVE_HALF: call <4 x half> @llvm.sin.v4f16
// NO_HALF-LABEL: define noundef <4 x float> @_Z14test_sin_half4
// NO_HALF: call <4 x float> @llvm.sin.v4f32(
half4 test_sin_half4(half4 p0) { return sin(p0); }

// CHECK-LABEL: define noundef float @_Z14test_sin_float
// CHECK: call float @llvm.sin.f32(
float test_sin_float(float p0) { return sin(p0); }
// CHECK-LABEL: define noundef <2 x float> @_Z15test_sin_float2
// CHECK: call <2 x float> @llvm.sin.v2f32
float2 test_sin_float2(float2 p0) { return sin(p0); }
// CHECK-LABEL: define noundef <3 x float> @_Z15test_sin_float3
// CHECK: call <3 x float> @llvm.sin.v3f32
float3 test_sin_float3(float3 p0) { return sin(p0); }
// CHECK-LABEL: define noundef <4 x float> @_Z15test_sin_float4
// CHECK: call <4 x float> @llvm.sin.v4f32
float4 test_sin_float4(float4 p0) { return sin(p0); }
