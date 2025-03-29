// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN: -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s -DTARGET=spv

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-LABEL: test_firstbitlow_ushort
// CHECK: call i32 @llvm.[[TARGET]].firstbitlow.i16
uint test_firstbitlow_ushort(uint16_t p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_ushort2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitlow.v2i16
uint2 test_firstbitlow_ushort2(uint16_t2 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_ushort3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitlow.v3i16
uint3 test_firstbitlow_ushort3(uint16_t3 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_ushort4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitlow.v4i16
uint4 test_firstbitlow_ushort4(uint16_t4 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_short
// CHECK: call i32 @llvm.[[TARGET]].firstbitlow.i16
uint test_firstbitlow_short(int16_t p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_short2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitlow.v2i16
uint2 test_firstbitlow_short2(int16_t2 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_short3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitlow.v3i16
uint3 test_firstbitlow_short3(int16_t3 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_short4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitlow.v4i16
uint4 test_firstbitlow_short4(int16_t4 p0) {
  return firstbitlow(p0);
}
#endif // __HLSL_ENABLE_16_BIT

// CHECK-LABEL: test_firstbitlow_uint
// CHECK: call i32 @llvm.[[TARGET]].firstbitlow.i32
uint test_firstbitlow_uint(uint p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_uint2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitlow.v2i32
uint2 test_firstbitlow_uint2(uint2 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_uint3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitlow.v3i32
uint3 test_firstbitlow_uint3(uint3 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_uint4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitlow.v4i32
uint4 test_firstbitlow_uint4(uint4 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_ulong
// CHECK: call i32 @llvm.[[TARGET]].firstbitlow.i64
uint test_firstbitlow_ulong(uint64_t p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_ulong2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitlow.v2i64
uint2 test_firstbitlow_ulong2(uint64_t2 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_ulong3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitlow.v3i64
uint3 test_firstbitlow_ulong3(uint64_t3 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_ulong4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitlow.v4i64
uint4 test_firstbitlow_ulong4(uint64_t4 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_int
// CHECK: call i32 @llvm.[[TARGET]].firstbitlow.i32
uint test_firstbitlow_int(int p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_int2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitlow.v2i32
uint2 test_firstbitlow_int2(int2 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_int3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitlow.v3i32
uint3 test_firstbitlow_int3(int3 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_int4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitlow.v4i32
uint4 test_firstbitlow_int4(int4 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_long
// CHECK: call i32 @llvm.[[TARGET]].firstbitlow.i64
uint test_firstbitlow_long(int64_t p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_long2
// CHECK: call <2 x i32> @llvm.[[TARGET]].firstbitlow.v2i64
uint2 test_firstbitlow_long2(int64_t2 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_long3
// CHECK: call <3 x i32> @llvm.[[TARGET]].firstbitlow.v3i64
uint3 test_firstbitlow_long3(int64_t3 p0) {
  return firstbitlow(p0);
}

// CHECK-LABEL: test_firstbitlow_long4
// CHECK: call <4 x i32> @llvm.[[TARGET]].firstbitlow.v4i64
uint4 test_firstbitlow_long4(int64_t4 p0) {
  return firstbitlow(p0);
}
