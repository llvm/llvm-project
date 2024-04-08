// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %dx.rsqrt = call half @llvm.dx.rsqrt.f16(
// NATIVE_HALF: ret half %dx.rsqrt
// NO_HALF: define noundef float @"?test_rsqrt_half@@YA$halff@$halff@@Z"(
// NO_HALF: %dx.rsqrt = call float @llvm.dx.rsqrt.f32(
// NO_HALF: ret float %dx.rsqrt
half test_rsqrt_half(half p0) { return rsqrt(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: %dx.rsqrt = call <2 x half> @llvm.dx.rsqrt.v2f16
// NATIVE_HALF: ret <2 x half> %dx.rsqrt
// NO_HALF: define noundef <2 x float> @
// NO_HALF: %dx.rsqrt = call <2 x float> @llvm.dx.rsqrt.v2f32(
// NO_HALF: ret <2 x float> %dx.rsqrt
half2 test_rsqrt_half2(half2 p0) { return rsqrt(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: %dx.rsqrt = call <3 x half> @llvm.dx.rsqrt.v3f16
// NATIVE_HALF: ret <3 x half> %dx.rsqrt
// NO_HALF: define noundef <3 x float> @
// NO_HALF: %dx.rsqrt = call <3 x float> @llvm.dx.rsqrt.v3f32(
// NO_HALF: ret <3 x float> %dx.rsqrt
half3 test_rsqrt_half3(half3 p0) { return rsqrt(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: %dx.rsqrt = call <4 x half> @llvm.dx.rsqrt.v4f16
// NATIVE_HALF: ret <4 x half> %dx.rsqrt
// NO_HALF: define noundef <4 x float> @
// NO_HALF: %dx.rsqrt = call <4 x float> @llvm.dx.rsqrt.v4f32(
// NO_HALF: ret <4 x float> %dx.rsqrt
half4 test_rsqrt_half4(half4 p0) { return rsqrt(p0); }

// CHECK: define noundef float @
// CHECK: %dx.rsqrt = call float @llvm.dx.rsqrt.f32(
// CHECK: ret float %dx.rsqrt
float test_rsqrt_float(float p0) { return rsqrt(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: %dx.rsqrt = call <2 x float> @llvm.dx.rsqrt.v2f32
// CHECK: ret <2 x float> %dx.rsqrt
float2 test_rsqrt_float2(float2 p0) { return rsqrt(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: %dx.rsqrt = call <3 x float> @llvm.dx.rsqrt.v3f32
// CHECK: ret <3 x float> %dx.rsqrt
float3 test_rsqrt_float3(float3 p0) { return rsqrt(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: %dx.rsqrt = call <4 x float> @llvm.dx.rsqrt.v4f32
// CHECK: ret <4 x float> %dx.rsqrt
float4 test_rsqrt_float4(float4 p0) { return rsqrt(p0); }
