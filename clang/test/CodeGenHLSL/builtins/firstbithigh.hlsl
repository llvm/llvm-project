// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -fnative-int16-type -emit-llvm -O1 -o - | FileCheck %s -DTARGET=dx \
// RUN:   --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -fnative-int16-type -emit-llvm -O1 -o - | FileCheck %s -DTARGET=spv

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-LABEL: test_firstbithigh_ushort
// CHECK: [[FBH:%.*]] = tail call {{.*}}i32 @llvm.[[TARGET]].firstbituhigh.i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub i32 15, [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq i32 [[FBH]], -1
// DXCHECK-NEXT: select i1 %cmp.i, i32 -1, i32 [[SUB]]
// CHECK-NEXT: ret i32
uint test_firstbithigh_ushort(uint16_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ushort2
// CHECK: [[FBH:%.*]] = tail call {{.*}}<2 x i32> @llvm.[[TARGET]].firstbituhigh.v2i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub <2 x i32> splat (i32 15), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <2 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <2 x i1> %cmp.i, <2 x i32> splat (i32 -1), <2 x i32> [[SUB]]
// CHECK-NEXT: ret <2 x i32>
uint2 test_firstbithigh_ushort2(uint16_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ushort3
// CHECK: [[FBH:%.*]] = tail call {{.*}}<3 x i32> @llvm.[[TARGET]].firstbituhigh.v3i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub <3 x i32> splat (i32 15), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <3 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <3 x i1> %cmp.i, <3 x i32> splat (i32 -1), <3 x i32> [[SUB]]
// CHECK-NEXT: ret <3 x i32>
uint3 test_firstbithigh_ushort3(uint16_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ushort4
// CHECK: [[FBH:%.*]] = tail call {{.*}}<4 x i32> @llvm.[[TARGET]].firstbituhigh.v4i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub <4 x i32> splat (i32 15), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <4 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <4 x i1> %cmp.i, <4 x i32> splat (i32 -1), <4 x i32> [[SUB]]
// CHECK-NEXT: ret <4 x i32>
uint4 test_firstbithigh_ushort4(uint16_t4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short
// CHECK: [[FBH:%.*]] = tail call {{.*}}i32 @llvm.[[TARGET]].firstbitshigh.i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub i32 15, [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq i32 [[FBH]], -1
// DXCHECK-NEXT: select i1 %cmp.i, i32 -1, i32 [[SUB]]
// CHECK-NEXT: ret i32
uint test_firstbithigh_short(int16_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short2
// CHECK: [[FBH:%.*]] = tail call {{.*}}<2 x i32> @llvm.[[TARGET]].firstbitshigh.v2i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub <2 x i32> splat (i32 15), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <2 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <2 x i1> %cmp.i, <2 x i32> splat (i32 -1), <2 x i32> [[SUB]]
// CHECK-NEXT: ret <2 x i32>
uint2 test_firstbithigh_short2(int16_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short3
// CHECK: [[FBH:%.*]] = tail call {{.*}}<3 x i32> @llvm.[[TARGET]].firstbitshigh.v3i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub <3 x i32> splat (i32 15), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <3 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <3 x i1> %cmp.i, <3 x i32> splat (i32 -1), <3 x i32> [[SUB]]
// CHECK-NEXT: ret <3 x i32>
uint3 test_firstbithigh_short3(int16_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_short4
// CHECK: [[FBH:%.*]] = tail call {{.*}}<4 x i32> @llvm.[[TARGET]].firstbitshigh.v4i16
// DXCHECK-NEXT: [[SUB:%.*]] = sub <4 x i32> splat (i32 15), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <4 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <4 x i1> %cmp.i, <4 x i32> splat (i32 -1), <4 x i32> [[SUB]]
// CHECK-NEXT: ret <4 x i32>
uint4 test_firstbithigh_short4(int16_t4 p0) {
  return firstbithigh(p0);
}
#endif // __HLSL_ENABLE_16_BIT

// CHECK-LABEL: test_firstbithigh_uint
// CHECK: [[FBH:%.*]] = tail call {{.*}}i32 @llvm.[[TARGET]].firstbituhigh.i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub i32 31, [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq i32 [[FBH]], -1
// DXCHECK-NEXT: select i1 %cmp.i, i32 -1, i32 [[SUB]]
// CHECK-NEXT: ret i32
uint test_firstbithigh_uint(uint p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_uint2
// CHECK: [[FBH:%.*]] = tail call {{.*}}<2 x i32> @llvm.[[TARGET]].firstbituhigh.v2i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub <2 x i32> splat (i32 31), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <2 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <2 x i1> %cmp.i, <2 x i32> splat (i32 -1), <2 x i32> [[SUB]]
// CHECK-NEXT: ret <2 x i32>
uint2 test_firstbithigh_uint2(uint2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_uint3
// CHECK: [[FBH:%.*]] = tail call {{.*}}<3 x i32> @llvm.[[TARGET]].firstbituhigh.v3i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub <3 x i32> splat (i32 31), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <3 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <3 x i1> %cmp.i, <3 x i32> splat (i32 -1), <3 x i32> [[SUB]]
// CHECK-NEXT: ret <3 x i32>
uint3 test_firstbithigh_uint3(uint3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_uint4
// CHECK: [[FBH:%.*]] = tail call {{.*}}<4 x i32> @llvm.[[TARGET]].firstbituhigh.v4i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub <4 x i32> splat (i32 31), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <4 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <4 x i1> %cmp.i, <4 x i32> splat (i32 -1), <4 x i32> [[SUB]]
// CHECK-NEXT: ret <4 x i32>
uint4 test_firstbithigh_uint4(uint4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong
// CHECK: [[FBH:%.*]] = tail call {{.*}}i32 @llvm.[[TARGET]].firstbituhigh.i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub i32 63, [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq i32 [[FBH]], -1
// DXCHECK-NEXT: select i1 %cmp.i, i32 -1, i32 [[SUB]]
// CHECK-NEXT: ret i32
uint test_firstbithigh_ulong(uint64_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong2
// CHECK: [[FBH:%.*]] = tail call {{.*}}<2 x i32> @llvm.[[TARGET]].firstbituhigh.v2i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub <2 x i32> splat (i32 63), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <2 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <2 x i1> %cmp.i, <2 x i32> splat (i32 -1), <2 x i32> [[SUB]]
// CHECK-NEXT: ret <2 x i32>
uint2 test_firstbithigh_ulong2(uint64_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong3
// CHECK: [[FBH:%.*]] = tail call {{.*}}<3 x i32> @llvm.[[TARGET]].firstbituhigh.v3i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub <3 x i32> splat (i32 63), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <3 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <3 x i1> %cmp.i, <3 x i32> splat (i32 -1), <3 x i32> [[SUB]]
// CHECK-NEXT: ret <3 x i32>
uint3 test_firstbithigh_ulong3(uint64_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_ulong4
// CHECK: [[FBH:%.*]] = tail call {{.*}}<4 x i32> @llvm.[[TARGET]].firstbituhigh.v4i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub <4 x i32> splat (i32 63), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <4 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <4 x i1> %cmp.i, <4 x i32> splat (i32 -1), <4 x i32> [[SUB]]
// CHECK-NEXT: ret <4 x i32>
uint4 test_firstbithigh_ulong4(uint64_t4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int
// CHECK: [[FBH:%.*]] = tail call {{.*}}i32 @llvm.[[TARGET]].firstbitshigh.i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub i32 31, [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq i32 [[FBH]], -1
// DXCHECK-NEXT: select i1 %cmp.i, i32 -1, i32 [[SUB]]
// CHECK-NEXT: ret i32
uint test_firstbithigh_int(int p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int2
// CHECK: [[FBH:%.*]] = tail call {{.*}}<2 x i32> @llvm.[[TARGET]].firstbitshigh.v2i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub <2 x i32> splat (i32 31), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <2 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <2 x i1> %cmp.i, <2 x i32> splat (i32 -1), <2 x i32> [[SUB]]
// CHECK-NEXT: ret <2 x i32>
uint2 test_firstbithigh_int2(int2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int3
// CHECK: [[FBH:%.*]] = tail call {{.*}}<3 x i32> @llvm.[[TARGET]].firstbitshigh.v3i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub <3 x i32> splat (i32 31), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <3 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <3 x i1> %cmp.i, <3 x i32> splat (i32 -1), <3 x i32> [[SUB]]
// CHECK-NEXT: ret <3 x i32>
uint3 test_firstbithigh_int3(int3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_int4
// CHECK: [[FBH:%.*]] = tail call {{.*}}<4 x i32> @llvm.[[TARGET]].firstbitshigh.v4i32
// DXCHECK-NEXT: [[SUB:%.*]] = sub <4 x i32> splat (i32 31), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <4 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <4 x i1> %cmp.i, <4 x i32> splat (i32 -1), <4 x i32> [[SUB]]
// CHECK-NEXT: ret <4 x i32>
uint4 test_firstbithigh_int4(int4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long
// CHECK: [[FBH:%.*]] = tail call {{.*}}i32 @llvm.[[TARGET]].firstbitshigh.i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub i32 63, [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq i32 [[FBH]], -1
// DXCHECK-NEXT: select i1 %cmp.i, i32 -1, i32 [[SUB]]
// CHECK-NEXT: ret i32
uint test_firstbithigh_long(int64_t p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long2
// CHECK: [[FBH:%.*]] = tail call {{.*}}<2 x i32> @llvm.[[TARGET]].firstbitshigh.v2i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub <2 x i32> splat (i32 63), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <2 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <2 x i1> %cmp.i, <2 x i32> splat (i32 -1), <2 x i32> [[SUB]]
// CHECK-NEXT: ret <2 x i32>
uint2 test_firstbithigh_long2(int64_t2 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long3
// CHECK: [[FBH:%.*]] = tail call {{.*}}<3 x i32> @llvm.[[TARGET]].firstbitshigh.v3i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub <3 x i32> splat (i32 63), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <3 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <3 x i1> %cmp.i, <3 x i32> splat (i32 -1), <3 x i32> [[SUB]]
// CHECK-NEXT: ret <3 x i32>
uint3 test_firstbithigh_long3(int64_t3 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_long4
// CHECK: [[FBH:%.*]] = tail call {{.*}}<4 x i32> @llvm.[[TARGET]].firstbitshigh.v4i64
// DXCHECK-NEXT: [[SUB:%.*]] = sub <4 x i32> splat (i32 63), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <4 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <4 x i1> %cmp.i, <4 x i32> splat (i32 -1), <4 x i32> [[SUB]]
// CHECK-NEXT: ret <4 x i32>
uint4 test_firstbithigh_long4(int64_t4 p0) {
  return firstbithigh(p0);
}

// CHECK-LABEL: test_firstbithigh_upcast
// CHECK: [[FBH:%.*]] = tail call {{.*}}<4 x i32> @llvm.[[TARGET]].firstbituhigh.v4i32(<4 x i32> %{{.*}})
// DXCHECK-NEXT: [[SUB:%.*]] = sub <4 x i32> splat (i32 31), [[FBH]]
// DXCHECK-NEXT: [[ICMP:%.*]] = icmp eq <4 x i32> [[FBH]], splat (i32 -1)
// DXCHECK-NEXT: select <4 x i1> %cmp.i, <4 x i32> splat (i32 -1), <4 x i32> [[SUB]]
// CHECK-NEXT: [[ZEXT:%.*]] = zext <4 x i32> {{.*}} to <4 x i64>
// CHECK-NEXT: ret <4 x i64> [[ZEXT]]
uint64_t4 test_firstbithigh_upcast(uint4 p0) {
  return firstbithigh(p0);
}
