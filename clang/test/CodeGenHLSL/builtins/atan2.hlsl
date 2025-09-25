// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_atan2_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.atan2.f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.atan2.f32
half test_atan2_half (half p0, half p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.atan2.v2f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
half2 test_atan2_half2 (half2 p0, half2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.atan2.v3f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
half3 test_atan2_half3 (half3 p0, half3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.atan2.v4f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
half4 test_atan2_half4 (half4 p0, half4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.atan2.f32
float test_atan2_float (float p0, float p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float2 test_atan2_float2 (float2 p0, float2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float3 test_atan2_float3 (float3 p0, float3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float4 test_atan2_float4 (float4 p0, float4 p1) {
  return atan2(p0, p1);
}
