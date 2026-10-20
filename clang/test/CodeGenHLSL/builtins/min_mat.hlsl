// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF-LABEL: test_min_short1x2
// NATIVE_HALF: call <2 x i16> @llvm.smin.v2i16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}})
int16_t1x2 test_min_short1x2(int16_t1x2 p0, int16_t1x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short1x3
// NATIVE_HALF: call <3 x i16> @llvm.smin.v3i16(<3 x i16> %{{.*}}, <3 x i16> %{{.*}})
int16_t1x3 test_min_short1x3(int16_t1x3 p0, int16_t1x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short1x4
// NATIVE_HALF: call <4 x i16> @llvm.smin.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}})
int16_t1x4 test_min_short1x4(int16_t1x4 p0, int16_t1x4 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short2x1
// NATIVE_HALF: call <2 x i16> @llvm.smin.v2i16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}})
int16_t2x1 test_min_short2x1(int16_t2x1 p0, int16_t2x1 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short2x2
// NATIVE_HALF: call <4 x i16> @llvm.smin.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}})
int16_t2x2 test_min_short2x2(int16_t2x2 p0, int16_t2x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short2x3
// NATIVE_HALF: call <6 x i16> @llvm.smin.v6i16(<6 x i16> %{{.*}}, <6 x i16> %{{.*}})
int16_t2x3 test_min_short2x3(int16_t2x3 p0, int16_t2x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short2x4
// NATIVE_HALF: call <8 x i16> @llvm.smin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
int16_t2x4 test_min_short2x4(int16_t2x4 p0, int16_t2x4 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short3x1
// NATIVE_HALF: call <3 x i16> @llvm.smin.v3i16(<3 x i16> %{{.*}}, <3 x i16> %{{.*}})
int16_t3x1 test_min_short3x1(int16_t3x1 p0, int16_t3x1 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short3x2
// NATIVE_HALF: call <6 x i16> @llvm.smin.v6i16(<6 x i16> %{{.*}}, <6 x i16> %{{.*}})
int16_t3x2 test_min_short3x2(int16_t3x2 p0, int16_t3x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short3x3
// NATIVE_HALF: call <9 x i16> @llvm.smin.v9i16(<9 x i16> %{{.*}}, <9 x i16> %{{.*}})
int16_t3x3 test_min_short3x3(int16_t3x3 p0, int16_t3x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short3x4
// NATIVE_HALF: call <12 x i16> @llvm.smin.v12i16(<12 x i16> %{{.*}}, <12 x i16> %{{.*}})
int16_t3x4 test_min_short3x4(int16_t3x4 p0, int16_t3x4 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short4x1
// NATIVE_HALF: call <4 x i16> @llvm.smin.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}})
int16_t4x1 test_min_short4x1(int16_t4x1 p0, int16_t4x1 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short4x2
// NATIVE_HALF: call <8 x i16> @llvm.smin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
int16_t4x2 test_min_short4x2(int16_t4x2 p0, int16_t4x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short4x3
// NATIVE_HALF: call <12 x i16> @llvm.smin.v12i16(<12 x i16> %{{.*}}, <12 x i16> %{{.*}})
int16_t4x3 test_min_short4x3(int16_t4x3 p0, int16_t4x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_short4x4
// NATIVE_HALF: call <16 x i16> @llvm.smin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
int16_t4x4 test_min_short4x4(int16_t4x4 p0, int16_t4x4 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort1x2
// NATIVE_HALF: call <2 x i16> @llvm.umin.v2i16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}})
uint16_t1x2 test_min_ushort1x2(uint16_t1x2 p0, uint16_t1x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort1x3
// NATIVE_HALF: call <3 x i16> @llvm.umin.v3i16(<3 x i16> %{{.*}}, <3 x i16> %{{.*}})
uint16_t1x3 test_min_ushort1x3(uint16_t1x3 p0, uint16_t1x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort1x4
// NATIVE_HALF: call <4 x i16> @llvm.umin.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}})
uint16_t1x4 test_min_ushort1x4(uint16_t1x4 p0, uint16_t1x4 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort2x1
// NATIVE_HALF: call <2 x i16> @llvm.umin.v2i16(<2 x i16> %{{.*}}, <2 x i16> %{{.*}})
uint16_t2x1 test_min_ushort2x1(uint16_t2x1 p0, uint16_t2x1 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort2x2
// NATIVE_HALF: call <4 x i16> @llvm.umin.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}})
uint16_t2x2 test_min_ushort2x2(uint16_t2x2 p0, uint16_t2x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort2x3
// NATIVE_HALF: call <6 x i16> @llvm.umin.v6i16(<6 x i16> %{{.*}}, <6 x i16> %{{.*}})
uint16_t2x3 test_min_ushort2x3(uint16_t2x3 p0, uint16_t2x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort2x4
// NATIVE_HALF: call <8 x i16> @llvm.umin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
uint16_t2x4 test_min_ushort2x4(uint16_t2x4 p0, uint16_t2x4 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort3x1
// NATIVE_HALF: call <3 x i16> @llvm.umin.v3i16(<3 x i16> %{{.*}}, <3 x i16> %{{.*}})
uint16_t3x1 test_min_ushort3x1(uint16_t3x1 p0, uint16_t3x1 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort3x2
// NATIVE_HALF: call <6 x i16> @llvm.umin.v6i16(<6 x i16> %{{.*}}, <6 x i16> %{{.*}})
uint16_t3x2 test_min_ushort3x2(uint16_t3x2 p0, uint16_t3x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort3x3
// NATIVE_HALF: call <9 x i16> @llvm.umin.v9i16(<9 x i16> %{{.*}}, <9 x i16> %{{.*}})
uint16_t3x3 test_min_ushort3x3(uint16_t3x3 p0, uint16_t3x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort3x4
// NATIVE_HALF: call <12 x i16> @llvm.umin.v12i16(<12 x i16> %{{.*}}, <12 x i16> %{{.*}})
uint16_t3x4 test_min_ushort3x4(uint16_t3x4 p0, uint16_t3x4 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort4x1
// NATIVE_HALF: call <4 x i16> @llvm.umin.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}})
uint16_t4x1 test_min_ushort4x1(uint16_t4x1 p0, uint16_t4x1 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort4x2
// NATIVE_HALF: call <8 x i16> @llvm.umin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
uint16_t4x2 test_min_ushort4x2(uint16_t4x2 p0, uint16_t4x2 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort4x3
// NATIVE_HALF: call <12 x i16> @llvm.umin.v12i16(<12 x i16> %{{.*}}, <12 x i16> %{{.*}})
uint16_t4x3 test_min_ushort4x3(uint16_t4x3 p0, uint16_t4x3 p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: test_min_ushort4x4
// NATIVE_HALF: call <16 x i16> @llvm.umin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
uint16_t4x4 test_min_ushort4x4(uint16_t4x4 p0, uint16_t4x4 p1) { return min(p0, p1); }

#endif

// CHECK-LABEL: test_min_int1x2
// CHECK: call <2 x i32> @llvm.smin.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}})
int1x2 test_min_int1x2(int1x2 p0, int1x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int1x3
// CHECK: call <3 x i32> @llvm.smin.v3i32(<3 x i32> %{{.*}}, <3 x i32> %{{.*}})
int1x3 test_min_int1x3(int1x3 p0, int1x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int1x4
// CHECK: call <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
int1x4 test_min_int1x4(int1x4 p0, int1x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int2x1
// CHECK: call <2 x i32> @llvm.smin.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}})
int2x1 test_min_int2x1(int2x1 p0, int2x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int2x2
// CHECK: call <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
int2x2 test_min_int2x2(int2x2 p0, int2x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int2x3
// CHECK: call <6 x i32> @llvm.smin.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}})
int2x3 test_min_int2x3(int2x3 p0, int2x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int2x4
// CHECK: call <8 x i32> @llvm.smin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
int2x4 test_min_int2x4(int2x4 p0, int2x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int3x1
// CHECK: call <3 x i32> @llvm.smin.v3i32(<3 x i32> %{{.*}}, <3 x i32> %{{.*}})
int3x1 test_min_int3x1(int3x1 p0, int3x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int3x2
// CHECK: call <6 x i32> @llvm.smin.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}})
int3x2 test_min_int3x2(int3x2 p0, int3x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int3x3
// CHECK: call <9 x i32> @llvm.smin.v9i32(<9 x i32> %{{.*}}, <9 x i32> %{{.*}})
int3x3 test_min_int3x3(int3x3 p0, int3x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int3x4
// CHECK: call <12 x i32> @llvm.smin.v12i32(<12 x i32> %{{.*}}, <12 x i32> %{{.*}})
int3x4 test_min_int3x4(int3x4 p0, int3x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int4x1
// CHECK: call <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
int4x1 test_min_int4x1(int4x1 p0, int4x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int4x2
// CHECK: call <8 x i32> @llvm.smin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
int4x2 test_min_int4x2(int4x2 p0, int4x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int4x3
// CHECK: call <12 x i32> @llvm.smin.v12i32(<12 x i32> %{{.*}}, <12 x i32> %{{.*}})
int4x3 test_min_int4x3(int4x3 p0, int4x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_int4x4
// CHECK: call <16 x i32> @llvm.smin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
int4x4 test_min_int4x4(int4x4 p0, int4x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint1x2
// CHECK: call <2 x i32> @llvm.umin.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}})
uint1x2 test_min_uint1x2(uint1x2 p0, uint1x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint1x3
// CHECK: call <3 x i32> @llvm.umin.v3i32(<3 x i32> %{{.*}}, <3 x i32> %{{.*}})
uint1x3 test_min_uint1x3(uint1x3 p0, uint1x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint1x4
// CHECK: call <4 x i32> @llvm.umin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
uint1x4 test_min_uint1x4(uint1x4 p0, uint1x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint2x1
// CHECK: call <2 x i32> @llvm.umin.v2i32(<2 x i32> %{{.*}}, <2 x i32> %{{.*}})
uint2x1 test_min_uint2x1(uint2x1 p0, uint2x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint2x2
// CHECK: call <4 x i32> @llvm.umin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
uint2x2 test_min_uint2x2(uint2x2 p0, uint2x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint2x3
// CHECK: call <6 x i32> @llvm.umin.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}})
uint2x3 test_min_uint2x3(uint2x3 p0, uint2x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint2x4
// CHECK: call <8 x i32> @llvm.umin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
uint2x4 test_min_uint2x4(uint2x4 p0, uint2x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint3x1
// CHECK: call <3 x i32> @llvm.umin.v3i32(<3 x i32> %{{.*}}, <3 x i32> %{{.*}})
uint3x1 test_min_uint3x1(uint3x1 p0, uint3x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint3x2
// CHECK: call <6 x i32> @llvm.umin.v6i32(<6 x i32> %{{.*}}, <6 x i32> %{{.*}})
uint3x2 test_min_uint3x2(uint3x2 p0, uint3x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint3x3
// CHECK: call <9 x i32> @llvm.umin.v9i32(<9 x i32> %{{.*}}, <9 x i32> %{{.*}})
uint3x3 test_min_uint3x3(uint3x3 p0, uint3x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint3x4
// CHECK: call <12 x i32> @llvm.umin.v12i32(<12 x i32> %{{.*}}, <12 x i32> %{{.*}})
uint3x4 test_min_uint3x4(uint3x4 p0, uint3x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint4x1
// CHECK: call <4 x i32> @llvm.umin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
uint4x1 test_min_uint4x1(uint4x1 p0, uint4x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint4x2
// CHECK: call <8 x i32> @llvm.umin.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
uint4x2 test_min_uint4x2(uint4x2 p0, uint4x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint4x3
// CHECK: call <12 x i32> @llvm.umin.v12i32(<12 x i32> %{{.*}}, <12 x i32> %{{.*}})
uint4x3 test_min_uint4x3(uint4x3 p0, uint4x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_uint4x4
// CHECK: call <16 x i32> @llvm.umin.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
uint4x4 test_min_uint4x4(uint4x4 p0, uint4x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long1x2
// CHECK: call <2 x i64> @llvm.smin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
int64_t1x2 test_min_long1x2(int64_t1x2 p0, int64_t1x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long1x3
// CHECK: call <3 x i64> @llvm.smin.v3i64(<3 x i64> %{{.*}}, <3 x i64> %{{.*}})
int64_t1x3 test_min_long1x3(int64_t1x3 p0, int64_t1x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long1x4
// CHECK: call <4 x i64> @llvm.smin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
int64_t1x4 test_min_long1x4(int64_t1x4 p0, int64_t1x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long2x1
// CHECK: call <2 x i64> @llvm.smin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
int64_t2x1 test_min_long2x1(int64_t2x1 p0, int64_t2x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long2x2
// CHECK: call <4 x i64> @llvm.smin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
int64_t2x2 test_min_long2x2(int64_t2x2 p0, int64_t2x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long2x3
// CHECK: call <6 x i64> @llvm.smin.v6i64(<6 x i64> %{{.*}}, <6 x i64> %{{.*}})
int64_t2x3 test_min_long2x3(int64_t2x3 p0, int64_t2x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long2x4
// CHECK: call <8 x i64> @llvm.smin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
int64_t2x4 test_min_long2x4(int64_t2x4 p0, int64_t2x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long3x1
// CHECK: call <3 x i64> @llvm.smin.v3i64(<3 x i64> %{{.*}}, <3 x i64> %{{.*}})
int64_t3x1 test_min_long3x1(int64_t3x1 p0, int64_t3x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long3x2
// CHECK: call <6 x i64> @llvm.smin.v6i64(<6 x i64> %{{.*}}, <6 x i64> %{{.*}})
int64_t3x2 test_min_long3x2(int64_t3x2 p0, int64_t3x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long3x3
// CHECK: call <9 x i64> @llvm.smin.v9i64(<9 x i64> %{{.*}}, <9 x i64> %{{.*}})
int64_t3x3 test_min_long3x3(int64_t3x3 p0, int64_t3x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long3x4
// CHECK: call <12 x i64> @llvm.smin.v12i64(<12 x i64> %{{.*}}, <12 x i64> %{{.*}})
int64_t3x4 test_min_long3x4(int64_t3x4 p0, int64_t3x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long4x1
// CHECK: call <4 x i64> @llvm.smin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
int64_t4x1 test_min_long4x1(int64_t4x1 p0, int64_t4x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long4x2
// CHECK: call <8 x i64> @llvm.smin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
int64_t4x2 test_min_long4x2(int64_t4x2 p0, int64_t4x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long4x3
// CHECK: call <12 x i64> @llvm.smin.v12i64(<12 x i64> %{{.*}}, <12 x i64> %{{.*}})
int64_t4x3 test_min_long4x3(int64_t4x3 p0, int64_t4x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_long4x4
// CHECK: call <16 x i64> @llvm.smin.v16i64(<16 x i64> %{{.*}}, <16 x i64> %{{.*}})
int64_t4x4 test_min_long4x4(int64_t4x4 p0, int64_t4x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong1x2
// CHECK: call <2 x i64> @llvm.umin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
uint64_t1x2 test_min_ulong1x2(uint64_t1x2 p0, uint64_t1x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong1x3
// CHECK: call <3 x i64> @llvm.umin.v3i64(<3 x i64> %{{.*}}, <3 x i64> %{{.*}})
uint64_t1x3 test_min_ulong1x3(uint64_t1x3 p0, uint64_t1x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong1x4
// CHECK: call <4 x i64> @llvm.umin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
uint64_t1x4 test_min_ulong1x4(uint64_t1x4 p0, uint64_t1x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong2x1
// CHECK: call <2 x i64> @llvm.umin.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
uint64_t2x1 test_min_ulong2x1(uint64_t2x1 p0, uint64_t2x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong2x2
// CHECK: call <4 x i64> @llvm.umin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
uint64_t2x2 test_min_ulong2x2(uint64_t2x2 p0, uint64_t2x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong2x3
// CHECK: call <6 x i64> @llvm.umin.v6i64(<6 x i64> %{{.*}}, <6 x i64> %{{.*}})
uint64_t2x3 test_min_ulong2x3(uint64_t2x3 p0, uint64_t2x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong2x4
// CHECK: call <8 x i64> @llvm.umin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
uint64_t2x4 test_min_ulong2x4(uint64_t2x4 p0, uint64_t2x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong3x1
// CHECK: call <3 x i64> @llvm.umin.v3i64(<3 x i64> %{{.*}}, <3 x i64> %{{.*}})
uint64_t3x1 test_min_ulong3x1(uint64_t3x1 p0, uint64_t3x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong3x2
// CHECK: call <6 x i64> @llvm.umin.v6i64(<6 x i64> %{{.*}}, <6 x i64> %{{.*}})
uint64_t3x2 test_min_ulong3x2(uint64_t3x2 p0, uint64_t3x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong3x3
// CHECK: call <9 x i64> @llvm.umin.v9i64(<9 x i64> %{{.*}}, <9 x i64> %{{.*}})
uint64_t3x3 test_min_ulong3x3(uint64_t3x3 p0, uint64_t3x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong3x4
// CHECK: call <12 x i64> @llvm.umin.v12i64(<12 x i64> %{{.*}}, <12 x i64> %{{.*}})
uint64_t3x4 test_min_ulong3x4(uint64_t3x4 p0, uint64_t3x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong4x1
// CHECK: call <4 x i64> @llvm.umin.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
uint64_t4x1 test_min_ulong4x1(uint64_t4x1 p0, uint64_t4x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong4x2
// CHECK: call <8 x i64> @llvm.umin.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
uint64_t4x2 test_min_ulong4x2(uint64_t4x2 p0, uint64_t4x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong4x3
// CHECK: call <12 x i64> @llvm.umin.v12i64(<12 x i64> %{{.*}}, <12 x i64> %{{.*}})
uint64_t4x3 test_min_ulong4x3(uint64_t4x3 p0, uint64_t4x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_ulong4x4
// CHECK: call <16 x i64> @llvm.umin.v16i64(<16 x i64> %{{.*}}, <16 x i64> %{{.*}})
uint64_t4x4 test_min_ulong4x4(uint64_t4x4 p0, uint64_t4x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half1x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.minnum.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.minnum.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
half1x2 test_min_half1x2(half1x2 p0, half1x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half1x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.minnum.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.minnum.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
half1x3 test_min_half1x3(half1x3 p0, half1x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half1x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.minnum.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
half1x4 test_min_half1x4(half1x4 p0, half1x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half2x1
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.minnum.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.minnum.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
half2x1 test_min_half2x1(half2x1 p0, half2x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half2x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.minnum.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
half2x2 test_min_half2x2(half2x2 p0, half2x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half2x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.minnum.v6f16(<6 x half> %{{.*}}, <6 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.minnum.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}})
half2x3 test_min_half2x3(half2x3 p0, half2x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half2x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <8 x half> @llvm.minnum.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.minnum.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}})
half2x4 test_min_half2x4(half2x4 p0, half2x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half3x1
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.minnum.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.minnum.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
half3x1 test_min_half3x1(half3x1 p0, half3x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half3x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.minnum.v6f16(<6 x half> %{{.*}}, <6 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.minnum.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}})
half3x2 test_min_half3x2(half3x2 p0, half3x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half3x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <9 x half> @llvm.minnum.v9f16(<9 x half> %{{.*}}, <9 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <9 x float> @llvm.minnum.v9f32(<9 x float> %{{.*}}, <9 x float> %{{.*}})
half3x3 test_min_half3x3(half3x3 p0, half3x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half3x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.minnum.v12f16(<12 x half> %{{.*}}, <12 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.minnum.v12f32(<12 x float> %{{.*}}, <12 x float> %{{.*}})
half3x4 test_min_half3x4(half3x4 p0, half3x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half4x1
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.minnum.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
half4x1 test_min_half4x1(half4x1 p0, half4x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half4x2
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <8 x half> @llvm.minnum.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.minnum.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}})
half4x2 test_min_half4x2(half4x2 p0, half4x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half4x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.minnum.v12f16(<12 x half> %{{.*}}, <12 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.minnum.v12f32(<12 x float> %{{.*}}, <12 x float> %{{.*}})
half4x3 test_min_half4x3(half4x3 p0, half4x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_half4x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.minnum.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.minnum.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}})
half4x4 test_min_half4x4(half4x4 p0, half4x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float1x2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.minnum.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
float1x2 test_min_float1x2(float1x2 p0, float1x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float1x3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.minnum.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
float1x3 test_min_float1x3(float1x3 p0, float1x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float1x4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
float1x4 test_min_float1x4(float1x4 p0, float1x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float2x1
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.minnum.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
float2x1 test_min_float2x1(float2x1 p0, float2x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float2x2
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
float2x2 test_min_float2x2(float2x2 p0, float2x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float2x3
// CHECK: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.minnum.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}})
float2x3 test_min_float2x3(float2x3 p0, float2x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float2x4
// CHECK: call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.minnum.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}})
float2x4 test_min_float2x4(float2x4 p0, float2x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float3x1
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.minnum.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
float3x1 test_min_float3x1(float3x1 p0, float3x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float3x2
// CHECK: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.minnum.v6f32(<6 x float> %{{.*}}, <6 x float> %{{.*}})
float3x2 test_min_float3x2(float3x2 p0, float3x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float3x3
// CHECK: call reassoc nnan ninf nsz arcp afn <9 x float> @llvm.minnum.v9f32(<9 x float> %{{.*}}, <9 x float> %{{.*}})
float3x3 test_min_float3x3(float3x3 p0, float3x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float3x4
// CHECK: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.minnum.v12f32(<12 x float> %{{.*}}, <12 x float> %{{.*}})
float3x4 test_min_float3x4(float3x4 p0, float3x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float4x1
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
float4x1 test_min_float4x1(float4x1 p0, float4x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float4x2
// CHECK: call reassoc nnan ninf nsz arcp afn <8 x float> @llvm.minnum.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}})
float4x2 test_min_float4x2(float4x2 p0, float4x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float4x3
// CHECK: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.minnum.v12f32(<12 x float> %{{.*}}, <12 x float> %{{.*}})
float4x3 test_min_float4x3(float4x3 p0, float4x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_float4x4
// CHECK: call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.minnum.v16f32(<16 x float> %{{.*}}, <16 x float> %{{.*}})
float4x4 test_min_float4x4(float4x4 p0, float4x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double1x2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.minnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})
double1x2 test_min_double1x2(double1x2 p0, double1x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double1x3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.minnum.v3f64(<3 x double> %{{.*}}, <3 x double> %{{.*}})
double1x3 test_min_double1x3(double1x3 p0, double1x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double1x4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.minnum.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}})
double1x4 test_min_double1x4(double1x4 p0, double1x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double2x1
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.minnum.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}})
double2x1 test_min_double2x1(double2x1 p0, double2x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double2x2
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.minnum.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}})
double2x2 test_min_double2x2(double2x2 p0, double2x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double2x3
// CHECK: call reassoc nnan ninf nsz arcp afn <6 x double> @llvm.minnum.v6f64(<6 x double> %{{.*}}, <6 x double> %{{.*}})
double2x3 test_min_double2x3(double2x3 p0, double2x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double2x4
// CHECK: call reassoc nnan ninf nsz arcp afn <8 x double> @llvm.minnum.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}})
double2x4 test_min_double2x4(double2x4 p0, double2x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double3x1
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.minnum.v3f64(<3 x double> %{{.*}}, <3 x double> %{{.*}})
double3x1 test_min_double3x1(double3x1 p0, double3x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double3x2
// CHECK: call reassoc nnan ninf nsz arcp afn <6 x double> @llvm.minnum.v6f64(<6 x double> %{{.*}}, <6 x double> %{{.*}})
double3x2 test_min_double3x2(double3x2 p0, double3x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double3x3
// CHECK: call reassoc nnan ninf nsz arcp afn <9 x double> @llvm.minnum.v9f64(<9 x double> %{{.*}}, <9 x double> %{{.*}})
double3x3 test_min_double3x3(double3x3 p0, double3x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double3x4
// CHECK: call reassoc nnan ninf nsz arcp afn <12 x double> @llvm.minnum.v12f64(<12 x double> %{{.*}}, <12 x double> %{{.*}})
double3x4 test_min_double3x4(double3x4 p0, double3x4 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double4x1
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.minnum.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}})
double4x1 test_min_double4x1(double4x1 p0, double4x1 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double4x2
// CHECK: call reassoc nnan ninf nsz arcp afn <8 x double> @llvm.minnum.v8f64(<8 x double> %{{.*}}, <8 x double> %{{.*}})
double4x2 test_min_double4x2(double4x2 p0, double4x2 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double4x3
// CHECK: call reassoc nnan ninf nsz arcp afn <12 x double> @llvm.minnum.v12f64(<12 x double> %{{.*}}, <12 x double> %{{.*}})
double4x3 test_min_double4x3(double4x3 p0, double4x3 p1) { return min(p0, p1); }

// CHECK-LABEL: test_min_double4x4
// CHECK: call reassoc nnan ninf nsz arcp afn <16 x double> @llvm.minnum.v16f64(<16 x double> %{{.*}}, <16 x double> %{{.*}})
double4x4 test_min_double4x4(double4x4 p0, double4x4 p1) { return min(p0, p1); }

