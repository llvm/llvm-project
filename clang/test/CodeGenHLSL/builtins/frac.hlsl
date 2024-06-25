// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %dx.frac = call half @llvm.dx.frac.f16(
// NATIVE_HALF: ret half %dx.frac
// NO_HALF: define noundef float @"?test_frac_half@@YA$halff@$halff@@Z"(
// NO_HALF: %dx.frac = call float @llvm.dx.frac.f32(
// NO_HALF: ret float %dx.frac
half test_frac_half(half p0) { return frac(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: %dx.frac = call <2 x half> @llvm.dx.frac.v2f16
// NATIVE_HALF: ret <2 x half> %dx.frac
// NO_HALF: define noundef <2 x float> @
// NO_HALF: %dx.frac = call <2 x float> @llvm.dx.frac.v2f32(
// NO_HALF: ret <2 x float> %dx.frac
half2 test_frac_half2(half2 p0) { return frac(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: %dx.frac = call <3 x half> @llvm.dx.frac.v3f16
// NATIVE_HALF: ret <3 x half> %dx.frac
// NO_HALF: define noundef <3 x float> @
// NO_HALF: %dx.frac = call <3 x float> @llvm.dx.frac.v3f32(
// NO_HALF: ret <3 x float> %dx.frac
half3 test_frac_half3(half3 p0) { return frac(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: %dx.frac = call <4 x half> @llvm.dx.frac.v4f16
// NATIVE_HALF: ret <4 x half> %dx.frac
// NO_HALF: define noundef <4 x float> @
// NO_HALF: %dx.frac = call <4 x float> @llvm.dx.frac.v4f32(
// NO_HALF: ret <4 x float> %dx.frac
half4 test_frac_half4(half4 p0) { return frac(p0); }

// CHECK: define noundef float @
// CHECK: %dx.frac = call float @llvm.dx.frac.f32(
// CHECK: ret float %dx.frac
float test_frac_float(float p0) { return frac(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: %dx.frac = call <2 x float> @llvm.dx.frac.v2f32
// CHECK: ret <2 x float> %dx.frac
float2 test_frac_float2(float2 p0) { return frac(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: %dx.frac = call <3 x float> @llvm.dx.frac.v3f32
// CHECK: ret <3 x float> %dx.frac
float3 test_frac_float3(float3 p0) { return frac(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: %dx.frac = call <4 x float> @llvm.dx.frac.v4f32
// CHECK: ret <4 x float> %dx.frac
float4 test_frac_float4(float4 p0) { return frac(p0); }
