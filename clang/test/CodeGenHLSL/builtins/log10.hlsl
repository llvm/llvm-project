// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -D__HLSL_ENABLE_16_BIT -o - | FileCheck %s --check-prefix=NO_HALF

// CHECK: define noundef half @
// CHECK: call half @llvm.log10.f16(
// NO_HALF: define noundef float @"?test_log10_half@@YA$halff@$halff@@Z"(
// NO_HALF: call float @llvm.log10.f32(
half test_log10_half ( half p0 ) {
  return log10 ( p0 );
}
// CHECK: define noundef <2 x half> @
// CHECK: call <2 x half> @llvm.log10.v2f16
// NO_HALF: define noundef <2 x float> @"?test_log10_float2@@YAT?$__vector@M$01@__clang@@T12@@Z"(
// NO_HALF: call <2 x float> @llvm.log10.v2f32(
half2 test_log10_half2 ( half2 p0 ) {
  return log10 ( p0 );
}
// CHECK: define noundef <3 x half> @
// CHECK: call <3 x half> @llvm.log10.v3f16
// NO_HALF: define noundef <3 x float> @"?test_log10_float3@@YAT?$__vector@M$02@__clang@@T12@@Z"(
// NO_HALF: call <3 x float> @llvm.log10.v3f32(
half3 test_log10_half3 ( half3 p0 ) {
  return log10 ( p0 );
}
// CHECK: define noundef <4 x half> @
// CHECK: call <4 x half> @llvm.log10.v4f16
// NO_HALF: define noundef <4 x float> @"?test_log10_float4@@YAT?$__vector@M$03@__clang@@T12@@Z"(
// NO_HALF: call <4 x float> @llvm.log10.v4f32(
half4 test_log10_half4 ( half4 p0 ) {
  return log10 ( p0 );
}

// CHECK: define noundef float @
// CHECK: call float @llvm.log10.f32(
float test_log10_float ( float p0 ) {
  return log10 ( p0 );
}
// CHECK: define noundef <2 x float> @
// CHECK: call <2 x float> @llvm.log10.v2f32
float2 test_log10_float2 ( float2 p0 ) {
  return log10 ( p0 );
}
// CHECK: define noundef <3 x float> @
// CHECK: call <3 x float> @llvm.log10.v3f32
float3 test_log10_float3 ( float3 p0 ) {
  return log10 ( p0 );
}
// CHECK: define noundef <4 x float> @
// CHECK: call <4 x float> @llvm.log10.v4f32
float4 test_log10_float4 ( float4 p0 ) {
  return log10 ( p0 );
}
