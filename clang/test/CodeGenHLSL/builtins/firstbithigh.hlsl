// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN: -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s -DTARGET=spv

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-LABEL: test_firstbithigh_ushort
// CHECK: call i32 @llvm.[[TARGET]].firstbituhigh.i16
uint test_firstbithigh_ushort(uint16_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ushort2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbituhigh.v2i16
uint2 test_firstbithigh_ushort2(uint16_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ushort3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbituhigh.v3i16
uint3 test_firstbithigh_ushort3(uint16_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ushort4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbituhigh.v4i16
uint4 test_firstbithigh_ushort4(uint16_t4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short
// CHECK: call i32 @llvm.[[TARGET]].firstbitshigh.i16
uint test_firstbithigh_short(int16_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitshigh.v2i16
uint2 test_firstbithigh_short2(int16_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitshigh.v3i16
uint3 test_firstbithigh_short3(int16_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitshigh.v4i16
uint4 test_firstbithigh_short4(int16_t4 p0) {
  return firstbithigh(p0);
}
#endif // __HLSL_ENABLE_16_BIT

// CHECK-LABEL: test_firstbithigh_uint
// CHECK: call i32 @llvm.[[TARGET]].firstbituhigh.i32
uint test_firstbithigh_uint(uint p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_uint2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbituhigh.v2i32
uint2 test_firstbithigh_uint2(uint2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_uint3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbituhigh.v3i32
uint3 test_firstbithigh_uint3(uint3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_uint4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbituhigh.v4i32
uint4 test_firstbithigh_uint4(uint4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong
// CHECK: call i32 @llvm.[[TARGET]].firstbituhigh.i64
uint test_firstbithigh_ulong(uint64_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbituhigh.v2i64
uint2 test_firstbithigh_ulong2(uint64_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbituhigh.v3i64
uint3 test_firstbithigh_ulong3(uint64_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbituhigh.v4i64
uint4 test_firstbithigh_ulong4(uint64_t4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int
// CHECK: call i32 @llvm.[[TARGET]].firstbitshigh.i32
uint test_firstbithigh_int(int p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitshigh.v2i32
uint2 test_firstbithigh_int2(int2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitshigh.v3i32
uint3 test_firstbithigh_int3(int3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitshigh.v4i32
uint4 test_firstbithigh_int4(int4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long
// CHECK: call i32 @llvm.[[TARGET]].firstbitshigh.i64
uint test_firstbithigh_long(int64_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitshigh.v2i64
uint2 test_firstbithigh_long2(int64_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitshigh.v3i64
uint3 test_firstbithigh_long3(int64_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitshigh.v4i64
uint4 test_firstbithigh_long4(int64_t4 p0) {
  return firstbithigh(p0);
}
