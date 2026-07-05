// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_atan2_half1x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.atan2.v2f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
half1x2 test_atan2_half1x2 (half1x2 p0, half1x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half1x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.atan2.v3f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
half1x3 test_atan2_half1x3 (half1x3 p0, half1x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half1x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.atan2.v4f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
half1x4 test_atan2_half1x4 (half1x4 p0, half1x4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half2x1
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.atan2.v2f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
half2x1 test_atan2_half2x1 (half2x1 p0, half2x1 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half2x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.atan2.v4f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
half2x2 test_atan2_half2x2 (half2x2 p0, half2x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half2x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.atan2.v6f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.atan2.v6f32
half2x3 test_atan2_half2x3 (half2x3 p0, half2x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half2x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <8 x half> @llvm.atan2.v8f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.atan2.v8f32
half2x4 test_atan2_half2x4 (half2x4 p0, half2x4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half3x1
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.atan2.v3f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
half3x1 test_atan2_half3x1 (half3x1 p0, half3x1 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half3x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.atan2.v6f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.atan2.v6f32
half3x2 test_atan2_half3x2 (half3x2 p0, half3x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half3x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <9 x half> @llvm.atan2.v9f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <9 x float> @llvm.atan2.v9f32
half3x3 test_atan2_half3x3 (half3x3 p0, half3x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half3x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.atan2.v12f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.atan2.v12f32
half3x4 test_atan2_half3x4 (half3x4 p0, half3x4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half4x1
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.atan2.v4f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
half4x1 test_atan2_half4x1 (half4x1 p0, half4x1 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half4x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <8 x half> @llvm.atan2.v8f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.atan2.v8f32
half4x2 test_atan2_half4x2 (half4x2 p0, half4x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half4x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.atan2.v12f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.atan2.v12f32
half4x3 test_atan2_half4x3 (half4x3 p0, half4x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half4x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.atan2.v16f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.atan2.v16f32
half4x4 test_atan2_half4x4 (half4x4 p0, half4x4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float1x2
// CHECK:  call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float1x2 test_atan2_float1x2 (float1x2 p0, float1x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float1x3
// CHECK:  call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float1x3 test_atan2_float1x3 (float1x3 p0, float1x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float1x4
// CHECK:  call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float1x4 test_atan2_float1x4 (float1x4 p0, float1x4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float2x1
// CHECK:  call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float2x1 test_atan2_float2x1 (float2x1 p0, float2x1 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float2x2
// CHECK:  call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float2x2 test_atan2_float2x2 (float2x2 p0, float2x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float2x3
// CHECK:  call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.atan2.v6f32
float2x3 test_atan2_float2x3 (float2x3 p0, float2x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float2x4
// CHECK:  call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.atan2.v8f32
float2x4 test_atan2_float2x4 (float2x4 p0, float2x4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float3x1
// CHECK:  call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float3x1 test_atan2_float3x1 (float3x1 p0, float3x1 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float3x2
// CHECK:  call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.atan2.v6f32
float3x2 test_atan2_float3x2 (float3x2 p0, float3x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float3x3
// CHECK:  call reassoc nnan ninf nsz arcp afn <9 x float> @llvm.atan2.v9f32
float3x3 test_atan2_float3x3 (float3x3 p0, float3x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float3x4
// CHECK:  call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.atan2.v12f32
float3x4 test_atan2_float3x4 (float3x4 p0, float3x4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float4x1
// CHECK:  call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float4x1 test_atan2_float4x1 (float4x1 p0, float4x1 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float4x2
// CHECK:  call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.atan2.v8f32
float4x2 test_atan2_float4x2 (float4x2 p0, float4x2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float4x3
// CHECK:  call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.atan2.v12f32
float4x3 test_atan2_float4x3 (float4x3 p0, float4x3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float4x4
// CHECK:  call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.atan2.v16f32
float4x4 test_atan2_float4x4 (float4x4 p0, float4x4 p1) {
  return atan2(p0, p1);
}
