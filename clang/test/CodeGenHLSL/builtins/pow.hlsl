// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -D__HLSL_ENABLE_16_BIT -o - | FileCheck %s --check-prefix=NO_HALF

// CHECK: define noundef half @
// CHECK: call half @llvm.pow.f16(
// NO_HALF: define noundef float @"?test_pow_half@@YA$halff@$halff@0@Z"(
// NO_HALF: call float @llvm.pow.f32(
half test_pow_half(half p0, half p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <2 x half> @"?test_pow_half2@@YAT?$__vector@$f16@$01@__clang@@T12@0@Z"(
// CHECK: call <2 x half> @llvm.pow.v2f16
// NO_HALF: define noundef <2 x float> @"?test_pow_float2@@YAT?$__vector@M$01@__clang@@T12@0@Z"(
// NO_HALF: call <2 x float> @llvm.pow.v2f32(
half2 test_pow_half2(half2 p0, half2 p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <3 x half> @"?test_pow_half3@@YAT?$__vector@$f16@$02@__clang@@T12@0@Z"(
// CHECK: call <3 x half> @llvm.pow.v3f16
// NO_HALF: define noundef <3 x float> @"?test_pow_float3@@YAT?$__vector@M$02@__clang@@T12@0@Z"(
// NO_HALF: call <3 x float> @llvm.pow.v3f32(
half3 test_pow_half3(half3 p0, half3 p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <4 x half> @"?test_pow_half4@@YAT?$__vector@$f16@$03@__clang@@T12@0@Z"(
// CHECK: call <4 x half> @llvm.pow.v4f16
// NO_HALF: define noundef <4 x float> @"?test_pow_float4@@YAT?$__vector@M$03@__clang@@T12@0@Z"(
// NO_HALF: call <4 x float> @llvm.pow.v4f32(
half4 test_pow_half4(half4 p0, half4 p1)
{
    return pow(p0, p1);
}

// CHECK: define noundef float @"?test_pow_float@@YAMMM@Z"(
// CHECK: call float @llvm.pow.f32(
float test_pow_float(float p0, float p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <2 x float> @"?test_pow_float2@@YAT?$__vector@M$01@__clang@@T12@0@Z"(
// CHECK: call <2 x float> @llvm.pow.v2f32
float2 test_pow_float2(float2 p0, float2 p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <3 x float> @"?test_pow_float3@@YAT?$__vector@M$02@__clang@@T12@0@Z"(
// CHECK: call <3 x float> @llvm.pow.v3f32
float3 test_pow_float3(float3 p0, float3 p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <4 x float> @"?test_pow_float4@@YAT?$__vector@M$03@__clang@@T12@0@Z"(
// CHECK: call <4 x float> @llvm.pow.v4f32
float4 test_pow_float4(float4 p0, float4 p1)
{
    return pow(p0, p1);
}

// CHECK: define noundef double @"?test_pow_double@@YANNN@Z"(
// CHECK: call double @llvm.pow.f64(
double test_pow_double(double p0, double p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <2 x double> @"?test_pow_double2@@YAT?$__vector@N$01@__clang@@T12@0@Z"(
// CHECK: call <2 x double> @llvm.pow.v2f64
double2 test_pow_double2(double2 p0, double2 p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <3 x double> @"?test_pow_double3@@YAT?$__vector@N$02@__clang@@T12@0@Z"(
// CHECK: call <3 x double> @llvm.pow.v3f64
double3 test_pow_double3(double3 p0, double3 p1)
{
    return pow(p0, p1);
}
// CHECK: define noundef <4 x double> @"?test_pow_double4@@YAT?$__vector@N$03@__clang@@T12@0@Z"(
// CHECK: call <4 x double> @llvm.pow.v4f64
double4 test_pow_double4(double4 p0, double4 p1)
{
    return pow(p0, p1);
}
