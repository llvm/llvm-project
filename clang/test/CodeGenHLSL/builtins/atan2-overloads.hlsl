// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: test_atan2_double
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.atan2.f32
float test_atan2_double (double p0, double p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float2 test_atan2_double2 (double2 p0, double2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float3 test_atan2_double3 (double3 p0, double3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float4 test_atan2_double4 (double4 p0, double4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.atan2.f32
float test_atan2_int (int p0, int p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float2 test_atan2_int2 (int2 p0, int2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float3 test_atan2_int3 (int3 p0, int3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float4 test_atan2_int4 (int4 p0, int4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.atan2.f32
float test_atan2_uint (uint p0, uint p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float2 test_atan2_uint2 (uint2 p0, uint2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float3 test_atan2_uint3 (uint3 p0, uint3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float4 test_atan2_uint4 (uint4 p0, uint4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.atan2.f32
float test_atan2_int64_t (int64_t p0, int64_t p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float2 test_atan2_int64_t2 (int64_t2 p0, int64_t2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float3 test_atan2_int64_t3 (int64_t3 p0, int64_t3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_int64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float4 test_atan2_int64_t4 (int64_t4 p0, int64_t4 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.atan2.f32
float test_atan2_uint64_t (uint64_t p0, uint64_t p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.atan2.v2f32
float2 test_atan2_uint64_t2 (uint64_t2 p0, uint64_t2 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.atan2.v3f32
float3 test_atan2_uint64_t3 (uint64_t3 p0, uint64_t3 p1) {
  return atan2(p0, p1);
}

// CHECK-LABEL: test_atan2_uint64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.atan2.v4f32
float4 test_atan2_uint64_t4 (uint64_t4 p0, uint64_t4 p1) {
  return atan2(p0, p1);
}
