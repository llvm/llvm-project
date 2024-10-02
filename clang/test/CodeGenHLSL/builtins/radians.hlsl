// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %{{.*}} = call half @llvm.dx.radians.f16(
// NATIVE_HALF: ret half %{{.*}}
// NO_HALF: define noundef float @"?test_radians_half@@YA$halff@$halff@@Z"(
// NO_HALF: %{{.*}} = call float @llvm.dx.radians.f32(
// NO_HALF: ret float %{{.*}}
half test_radians_half(half p0) { return radians(p0); }
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: %{{.*}} = call <2 x half> @llvm.dx.radians.v2f16
// NATIVE_HALF: ret <2 x half> %{{.*}}
// NO_HALF: define noundef <2 x float> @
// NO_HALF: %{{.*}} = call <2 x float> @llvm.dx.radians.v2f32(
// NO_HALF: ret <2 x float> %{{.*}}
half2 test_radians_half2(half2 p0) { return radians(p0); }
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: %{{.*}} = call <3 x half> @llvm.dx.radians.v3f16
// NATIVE_HALF: ret <3 x half> %{{.*}}
// NO_HALF: define noundef <3 x float> @
// NO_HALF: %{{.*}} = call <3 x float> @llvm.dx.radians.v3f32(
// NO_HALF: ret <3 x float> %{{.*}}
half3 test_radians_half3(half3 p0) { return radians(p0); }
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: %{{.*}} = call <4 x half> @llvm.dx.radians.v4f16
// NATIVE_HALF: ret <4 x half> %{{.*}}
// NO_HALF: define noundef <4 x float> @
// NO_HALF: %{{.*}} = call <4 x float> @llvm.dx.radians.v4f32(
// NO_HALF: ret <4 x float> %{{.*}}
half4 test_radians_half4(half4 p0) { return radians(p0); }

// CHECK: define noundef float @
// CHECK: %{{.*}} = call float @llvm.dx.radians.f32(
// CHECK: ret float %{{.*}}
float test_radians_float(float p0) { return radians(p0); }
// CHECK: define noundef <2 x float> @
// CHECK: %{{.*}} = call <2 x float> @llvm.dx.radians.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_radians_float2(float2 p0) { return radians(p0); }
// CHECK: define noundef <3 x float> @
// CHECK: %{{.*}} = call <3 x float> @llvm.dx.radians.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_radians_float3(float3 p0) { return radians(p0); }
// CHECK: define noundef <4 x float> @
// CHECK: %{{.*}} = call <4 x float> @llvm.dx.radians.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_radians_float4(float4 p0) { return radians(p0); }

