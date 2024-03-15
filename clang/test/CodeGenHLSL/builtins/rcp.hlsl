// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %dx.rcp = call half @llvm.dx.rcp.f16(
// NATIVE_HALF: ret half %dx.rcp
// NO_HALF: define noundef float @"?test_rcp_half@@YA$halff@$halff@@Z"(
// NO_HALF: %dx.rcp = call float @llvm.dx.rcp.f32(
// NO_HALF: ret float %dx.rcp
half test_rcp_half(half p0) { return rcp(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: %dx.rcp = call <2 x half> @llvm.dx.rcp.v2f16
// NATIVE_HALF: ret <2 x half> %dx.rcp
// NO_HALF: define noundef <2 x float> @
// NO_HALF: %dx.rcp = call <2 x float> @llvm.dx.rcp.v2f32(
// NO_HALF: ret <2 x float> %dx.rcp
half2 test_rcp_half2(half2 p0) { return rcp(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: %dx.rcp = call <3 x half> @llvm.dx.rcp.v3f16
// NATIVE_HALF: ret <3 x half> %dx.rcp
// NO_HALF: define noundef <3 x float> @
// NO_HALF: %dx.rcp = call <3 x float> @llvm.dx.rcp.v3f32(
// NO_HALF: ret <3 x float> %dx.rcp
half3 test_rcp_half3(half3 p0) { return rcp(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: %dx.rcp = call <4 x half> @llvm.dx.rcp.v4f16
// NATIVE_HALF: ret <4 x half> %dx.rcp
// NO_HALF: define noundef <4 x float> @
// NO_HALF: %dx.rcp = call <4 x float> @llvm.dx.rcp.v4f32(
// NO_HALF: ret <4 x float> %dx.rcp
half4 test_rcp_half4(half4 p0) { return rcp(p0); }

// CHECK: define noundef float @
// CHECK: %dx.rcp = call float @llvm.dx.rcp.f32(
// CHECK: ret float %dx.rcp
float test_rcp_float(float p0) { return rcp(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: %dx.rcp = call <2 x float> @llvm.dx.rcp.v2f32
// CHECK: ret <2 x float> %dx.rcp
float2 test_rcp_float2(float2 p0) { return rcp(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: %dx.rcp = call <3 x float> @llvm.dx.rcp.v3f32
// CHECK: ret <3 x float> %dx.rcp
float3 test_rcp_float3(float3 p0) { return rcp(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: %dx.rcp = call <4 x float> @llvm.dx.rcp.v4f32
// CHECK: ret <4 x float> %dx.rcp
float4 test_rcp_float4(float4 p0) { return rcp(p0); }
