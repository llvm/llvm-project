// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %elt.round = call half @llvm.round.f16(
// NATIVE_HALF: ret half %elt.round
// NO_HALF: define noundef float @"?test_round_half@@YA$halff@$halff@@Z"(
// NO_HALF: %elt.round = call float @llvm.round.f32(
// NO_HALF: ret float %elt.round
half test_round_half(half p0) { return round(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: %elt.round = call <2 x half> @llvm.round.v2f16
// NATIVE_HALF: ret <2 x half> %elt.round
// NO_HALF: define noundef <2 x float> @
// NO_HALF: %elt.round = call <2 x float> @llvm.round.v2f32(
// NO_HALF: ret <2 x float> %elt.round
half2 test_round_half2(half2 p0) { return round(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: %elt.round = call <3 x half> @llvm.round.v3f16
// NATIVE_HALF: ret <3 x half> %elt.round
// NO_HALF: define noundef <3 x float> @
// NO_HALF: %elt.round = call <3 x float> @llvm.round.v3f32(
// NO_HALF: ret <3 x float> %elt.round
half3 test_round_half3(half3 p0) { return round(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: %elt.round = call <4 x half> @llvm.round.v4f16
// NATIVE_HALF: ret <4 x half> %elt.round
// NO_HALF: define noundef <4 x float> @
// NO_HALF: %elt.round = call <4 x float> @llvm.round.v4f32(
// NO_HALF: ret <4 x float> %elt.round
half4 test_round_half4(half4 p0) { return round(p0); }

// CHECK: define noundef float @
// CHECK: %elt.round = call float @llvm.round.f32(
// CHECK: ret float %elt.round
float test_round_float(float p0) { return round(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: %elt.round = call <2 x float> @llvm.round.v2f32
// CHECK: ret <2 x float> %elt.round
float2 test_round_float2(float2 p0) { return round(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: %elt.round = call <3 x float> @llvm.round.v3f32
// CHECK: ret <3 x float> %elt.round
float3 test_round_float3(float3 p0) { return round(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: %elt.round = call <4 x float> @llvm.round.v4f32
// CHECK: ret <4 x float> %elt.round
float4 test_round_float4(float4 p0) { return round(p0); }
