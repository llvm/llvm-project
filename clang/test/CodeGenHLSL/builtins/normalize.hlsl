// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: call half @llvm.dx.normalize.f16(half
// NO_HALF: call float @llvm.dx.normalize.f32(float
// NATIVE_HALF: ret half
// NO_HALF: ret float
half test_normalize_half(half p0)
{
    return normalize(p0);
}
// NATIVE_HALF: define noundef <2 x half> @
// NATIVE_HALF: %hlsl.normalize = call <2 x half> @llvm.dx.normalize.v2f16
// NO_HALF: %hlsl.normalize = call <2 x float> @llvm.dx.normalize.v2f32(
// NATIVE_HALF: ret <2 x half> %hlsl.normalize
// NO_HALF: ret <2 x float> %hlsl.normalize
half2 test_normalize_half2(half2 p0)
{
    return normalize(p0);
}
// NATIVE_HALF: define noundef <3 x half> @
// NATIVE_HALF: %hlsl.normalize = call <3 x half> @llvm.dx.normalize.v3f16
// NO_HALF: %hlsl.normalize = call <3 x float> @llvm.dx.normalize.v3f32(
// NATIVE_HALF: ret <3 x half> %hlsl.normalize
// NO_HALF: ret <3 x float> %hlsl.normalize
half3 test_normalize_half3(half3 p0)
{
    return normalize(p0);
}
// NATIVE_HALF: define noundef <4 x half> @
// NATIVE_HALF: %hlsl.normalize = call <4 x half> @llvm.dx.normalize.v4f16
// NO_HALF: %hlsl.normalize = call <4 x float> @llvm.dx.normalize.v4f32(
// NATIVE_HALF: ret <4 x half> %hlsl.normalize
// NO_HALF: ret <4 x float> %hlsl.normalize
half4 test_normalize_half4(half4 p0)
{
    return normalize(p0);
}

// CHECK: define noundef float @
// CHECK: call float @llvm.dx.normalize.f32(float
// CHECK: ret float
float test_normalize_float(float p0)
{
    return normalize(p0);
}
// CHECK: define noundef <2 x float> @
// CHECK: %hlsl.normalize = call <2 x float> @llvm.dx.normalize.v2f32(
// CHECK: ret <2 x float> %hlsl.normalize
float2 test_normalize_float2(float2 p0)
{
    return normalize(p0);
}
// CHECK: define noundef <3 x float> @
// CHECK: %hlsl.normalize = call <3 x float> @llvm.dx.normalize.v3f32(
// CHECK: ret <3 x float> %hlsl.normalize
float3 test_normalize_float3(float3 p0)
{
    return normalize(p0);
}
// CHECK: define noundef <4 x float> @
// CHECK: %hlsl.normalize = call <4 x float> @llvm.dx.normalize.v4f32(
// CHECK: ret <4 x float> %hlsl.normalize
float4 test_length_float4(float4 p0)
{
    return normalize(p0);
}
