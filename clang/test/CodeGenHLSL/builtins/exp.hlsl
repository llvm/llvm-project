// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %elt.exp = call half @llvm.exp.f16(
// NATIVE_HALF: ret half %elt.exp
// NO_HALF: define noundef float @"?test_exp_half@@YA$halff@$halff@@Z"(
// NO_HALF: %elt.exp = call float @llvm.exp.f32(
// NO_HALF: ret float %elt.exp
half test_exp_half(half p0) { return exp(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: %elt.exp = call <2 x half> @llvm.exp.v2f16
// NATIVE_HALF: ret <2 x half> %elt.exp
// NO_HALF: define noundef <2 x float> @
// NO_HALF: %elt.exp = call <2 x float> @llvm.exp.v2f32(
// NO_HALF: ret <2 x float> %elt.exp
half2 test_exp_half2(half2 p0) { return exp(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: %elt.exp = call <3 x half> @llvm.exp.v3f16
// NATIVE_HALF: ret <3 x half> %elt.exp
// NO_HALF: define noundef <3 x float> @
// NO_HALF: %elt.exp = call <3 x float> @llvm.exp.v3f32(
// NO_HALF: ret <3 x float> %elt.exp
half3 test_exp_half3(half3 p0) { return exp(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: %elt.exp = call <4 x half> @llvm.exp.v4f16
// NATIVE_HALF: ret <4 x half> %elt.exp
// NO_HALF: define noundef <4 x float> @
// NO_HALF: %elt.exp = call <4 x float> @llvm.exp.v4f32(
// NO_HALF: ret <4 x float> %elt.exp
half4 test_exp_half4(half4 p0) { return exp(p0); }

// CHECK: define noundef float @
// CHECK: %elt.exp = call float @llvm.exp.f32(
// CHECK: ret float %elt.exp
float test_exp_float(float p0) { return exp(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: %elt.exp = call <2 x float> @llvm.exp.v2f32
// CHECK: ret <2 x float> %elt.exp
float2 test_exp_float2(float2 p0) { return exp(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: %elt.exp = call <3 x float> @llvm.exp.v3f32
// CHECK: ret <3 x float> %elt.exp
float3 test_exp_float3(float3 p0) { return exp(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: %elt.exp = call <4 x float> @llvm.exp.v4f32
// CHECK: ret <4 x float> %elt.exp
float4 test_exp_float4(float4 p0) { return exp(p0); }
