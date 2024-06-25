// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_tan_half
// NATIVE_HALF: call half @llvm.tan.f16
// NO_HALF: call float @llvm.tan.f32
half test_tan_half ( half p0 ) {
  return tan ( p0 );
}

// CHECK-LABEL: test_tan_half2
// NATIVE_HALF: call <2 x half> @llvm.tan.v2f16
// NO_HALF: call <2 x float> @llvm.tan.v2f32
half2 test_tan_half2 ( half2 p0 ) {
  return tan ( p0 );
}

// CHECK-LABEL: test_tan_half3
// NATIVE_HALF: call <3 x half> @llvm.tan.v3f16
// NO_HALF: call <3 x float> @llvm.tan.v3f32
half3 test_tan_half3 ( half3 p0 ) {
  return tan ( p0 );
}

// CHECK-LABEL: test_tan_half4
// NATIVE_HALF: call <4 x half> @llvm.tan.v4f16
// NO_HALF: call <4 x float> @llvm.tan.v4f32
half4 test_tan_half4 ( half4 p0 ) {
  return tan ( p0 );
}

// CHECK-LABEL: test_tan_float
// CHECK: call float @llvm.tan.f32
float test_tan_float ( float p0 ) {
  return tan ( p0 );
}

// CHECK-LABEL: test_tan_float2
// CHECK: call <2 x float> @llvm.tan.v2f32
float2 test_tan_float2 ( float2 p0 ) {
  return tan ( p0 );
}

// CHECK-LABEL: test_tan_float3
// CHECK: call <3 x float> @llvm.tan.v3f32
float3 test_tan_float3 ( float3 p0 ) {
  return tan ( p0 );
}

// CHECK-LABEL: test_tan_float4
// CHECK: call <4 x float> @llvm.tan.v4f32
float4 test_tan_float4 ( float4 p0 ) {
  return tan ( p0 );
}
