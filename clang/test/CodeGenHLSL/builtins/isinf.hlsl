// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK: define noundef i1 @
// NATIVE_HALF: %dx.isinf = call i1 @llvm.dx.isinf.f16(
// NO_HALF: %dx.isinf = call i1 @llvm.dx.isinf.f32(
// CHECK: ret i1 %dx.isinf
bool test_isinf_half(half p0) { return isinf(p0); }
// CHECK: define noundef <2 x i1> @
// NATIVE_HALF: %dx.isinf = call <2 x i1> @llvm.dx.isinf.v2f16
// NO_HALF: %dx.isinf = call <2 x i1> @llvm.dx.isinf.v2f32(
// CHECK: ret <2 x i1> %dx.isinf
bool2 test_isinf_half2(half2 p0) { return isinf(p0); }
// NATIVE_HALF: define noundef <3 x i1> @
// NATIVE_HALF: %dx.isinf = call <3 x i1> @llvm.dx.isinf.v3f16
// NO_HALF: %dx.isinf = call <3 x i1> @llvm.dx.isinf.v3f32(
// CHECK: ret <3 x i1> %dx.isinf
bool3 test_isinf_half3(half3 p0) { return isinf(p0); }
// NATIVE_HALF: define noundef <4 x i1> @
// NATIVE_HALF: %dx.isinf = call <4 x i1> @llvm.dx.isinf.v4f16
// NO_HALF: %dx.isinf = call <4 x i1> @llvm.dx.isinf.v4f32(
// CHECK: ret <4 x i1> %dx.isinf
bool4 test_isinf_half4(half4 p0) { return isinf(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.isinf = call i1 @llvm.dx.isinf.f32(
// CHECK: ret i1 %dx.isinf
bool test_isinf_float(float p0) { return isinf(p0); }
// CHECK: define noundef <2 x i1> @
// CHECK: %dx.isinf = call <2 x i1> @llvm.dx.isinf.v2f32
// CHECK: ret <2 x i1> %dx.isinf
bool2 test_isinf_float2(float2 p0) { return isinf(p0); }
// CHECK: define noundef <3 x i1> @
// CHECK: %dx.isinf = call <3 x i1> @llvm.dx.isinf.v3f32
// CHECK: ret <3 x i1> %dx.isinf
bool3 test_isinf_float3(float3 p0) { return isinf(p0); }
// CHECK: define noundef <4 x i1> @
// CHECK: %dx.isinf = call <4 x i1> @llvm.dx.isinf.v4f32
// CHECK: ret <4 x i1> %dx.isinf
bool4 test_isinf_float4(float4 p0) { return isinf(p0); }
