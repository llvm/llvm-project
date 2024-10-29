// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF: define noundef half @
// NATIVE_HALF: call half @llvm.fabs.f16(half
// NO_HALF: call float @llvm.fabs.f32(float
// NATIVE_HALF: ret half
// NO_HALF: ret float
half test_length_half(half p0)
{
  return length(p0);
}
// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %hlsl.length = call half @llvm.dx.length.v2f16
// NO_HALF: %hlsl.length = call float @llvm.dx.length.v2f32(
// NATIVE_HALF: ret half %hlsl.length
// NO_HALF: ret float %hlsl.length
half test_length_half2(half2 p0)
{
  return length(p0);
}
// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %hlsl.length = call half @llvm.dx.length.v3f16
// NO_HALF: %hlsl.length = call float @llvm.dx.length.v3f32(
// NATIVE_HALF: ret half %hlsl.length
// NO_HALF: ret float %hlsl.length
half test_length_half3(half3 p0)
{
  return length(p0);
}
// NATIVE_HALF: define noundef half @
// NATIVE_HALF: %hlsl.length = call half @llvm.dx.length.v4f16
// NO_HALF: %hlsl.length = call float @llvm.dx.length.v4f32(
// NATIVE_HALF: ret half %hlsl.length
// NO_HALF: ret float %hlsl.length
half test_length_half4(half4 p0)
{
  return length(p0);
}

// CHECK: define noundef float @
// CHECK: call float @llvm.fabs.f32(float
// CHECK: ret float
float test_length_float(float p0)
{
  return length(p0);
}
// CHECK: define noundef float @
// CHECK: %hlsl.length = call float @llvm.dx.length.v2f32(
// CHECK: ret float %hlsl.length
float test_length_float2(float2 p0)
{
  return length(p0);
}
// CHECK: define noundef float @
// CHECK: %hlsl.length = call float @llvm.dx.length.v3f32(
// CHECK: ret float %hlsl.length
float test_length_float3(float3 p0)
{
  return length(p0);
}
// CHECK: define noundef float @
// CHECK: %hlsl.length = call float @llvm.dx.length.v4f32(
// CHECK: ret float %hlsl.length
float test_length_float4(float4 p0)
{
  return length(p0);
}
