// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_sinh_half
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn half @llvm.sinh.f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn float @llvm.sinh.f32
half test_sinh_half ( half p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_half2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.sinh.v2f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sinh.v2f32
half2 test_sinh_half2 ( half2 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_half3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.sinh.v3f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sinh.v3f32
half3 test_sinh_half3 ( half3 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_half4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.sinh.v4f16
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sinh.v4f32
half4 test_sinh_half4 ( half4 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_float
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.sinh.f32
float test_sinh_float ( float p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_float2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sinh.v2f32
float2 test_sinh_float2 ( float2 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_float3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sinh.v3f32
float3 test_sinh_float3 ( float3 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_float4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sinh.v4f32
float4 test_sinh_float4 ( float4 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_double
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.sinh.f32
float test_sinh_double ( double p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sinh.v2f32
float2 test_sinh_double2 ( double2 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sinh.v3f32
float3 test_sinh_double3 ( double3 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sinh.v4f32
float4 test_sinh_double4 ( double4 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.sinh.f32
float test_sinh_int ( int p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sinh.v2f32
float2 test_sinh_int2 ( int2 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sinh.v3f32
float3 test_sinh_int3 ( int3 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sinh.v4f32
float4 test_sinh_int4 ( int4 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.sinh.f32
float test_sinh_uint ( uint p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sinh.v2f32
float2 test_sinh_uint2 ( uint2 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sinh.v3f32
float3 test_sinh_uint3 ( uint3 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sinh.v4f32
float4 test_sinh_uint4 ( uint4 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.sinh.f32
float test_sinh_int64_t ( int64_t p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sinh.v2f32
float2 test_sinh_int64_t2 ( int64_t2 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sinh.v3f32
float3 test_sinh_int64_t3 ( int64_t3 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_int64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sinh.v4f32
float4 test_sinh_int64_t4 ( int64_t4 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.sinh.f32
float test_sinh_uint64_t ( uint64_t p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sinh.v2f32
float2 test_sinh_uint64_t2 ( uint64_t2 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sinh.v3f32
float3 test_sinh_uint64_t3 ( uint64_t3 p0 ) {
  return sinh ( p0 );
}

// CHECK-LABEL: test_sinh_uint64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sinh.v4f32
float4 test_sinh_uint64_t4 ( uint64_t4 p0 ) {
  return sinh ( p0 );
}
