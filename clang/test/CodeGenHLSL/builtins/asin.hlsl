// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_asin_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.asin.f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.asin.f32
half test_asin_half ( half p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.asin.v2f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.asin.v2f32
half2 test_asin_half2 ( half2 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.asin.v3f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.asin.v3f32
half3 test_asin_half3 ( half3 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.asin.v4f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.asin.v4f32
half4 test_asin_half4 ( half4 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.asin.f32
float test_asin_float ( float p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.asin.v2f32
float2 test_asin_float2 ( float2 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.asin.v3f32
float3 test_asin_float3 ( float3 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.asin.v4f32
float4 test_asin_float4 ( float4 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_double
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.asin.f32
float test_asin_double ( double p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.asin.v2f32
float2 test_asin_double2 ( double2 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.asin.v3f32
float3 test_asin_double3 ( double3 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.asin.v4f32
float4 test_asin_double4 ( double4 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.asin.f32
float test_asin_int ( int p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.asin.v2f32
float2 test_asin_int2 ( int2 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.asin.v3f32
float3 test_asin_int3 ( int3 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.asin.v4f32
float4 test_asin_int4 ( int4 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.asin.f32
float test_asin_uint ( uint p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.asin.v2f32
float2 test_asin_uint2 ( uint2 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.asin.v3f32
float3 test_asin_uint3 ( uint3 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.asin.v4f32
float4 test_asin_uint4 ( uint4 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.asin.f32
float test_asin_int64_t ( int64_t p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.asin.v2f32
float2 test_asin_int64_t2 ( int64_t2 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.asin.v3f32
float3 test_asin_int64_t3 ( int64_t3 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_int64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.asin.v4f32
float4 test_asin_int64_t4 ( int64_t4 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.asin.f32
float test_asin_uint64_t ( uint64_t p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.asin.v2f32
float2 test_asin_uint64_t2 ( uint64_t2 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.asin.v3f32
float3 test_asin_uint64_t3 ( uint64_t3 p0 ) {
  return asin ( p0 );
}

// CHECK-LABEL: test_asin_uint64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.asin.v4f32
float4 test_asin_uint64_t4 ( uint64_t4 p0 ) {
  return asin ( p0 );
}
