// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -fnative-half-type -fnative-int16-type \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DINTRINSIC=dx.wave.multi.prefix.bit.and
// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -fnative-half-type -fnative-int16-type \
// RUN:   -triple spirv-pc-vulkan-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DINTRINSIC=spv.subgroup.multi.prefix.bit.and

// CHECK-LABEL: test_uint
uint test_uint(uint value, uint4 mask) {
  // CHECK: %[[TOKEN:.*]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[MASK:.*]] = load <4 x i32>
  // CHECK: %[[M0:.*]] = extractelement <4 x i32> %[[MASK]], i64 0
  // CHECK: %[[M1:.*]] = extractelement <4 x i32> %[[MASK]], i64 1
  // CHECK: %[[M2:.*]] = extractelement <4 x i32> %[[MASK]], i64 2
  // CHECK: %[[M3:.*]] = extractelement <4 x i32> %[[MASK]], i64 3
  // CHECK: %[[RET:.*]] = call i32 @llvm.[[INTRINSIC]].i32(i32 %{{.*}}, i32 %[[M0]], i32 %[[M1]], i32 %[[M2]], i32 %[[M3]]) [ "convergencectrl"(token %[[TOKEN]]) ]
  // CHECK: ret i32 %[[RET]]
  return WaveMultiPrefixBitAnd(value, mask);
}

// CHECK: declare i32 @llvm.[[INTRINSIC]].i32(i32, i32, i32, i32, i32) #[[ATTR:[0-9]+]]

// CHECK-LABEL: test_int
int test_int(int value) {
  // CHECK: %[[RET:.*]] = call i32 @llvm.[[INTRINSIC]].i32(i32 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret i32 %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_int16
int16_t test_int16(int16_t value) {
  // CHECK: %[[RET:.*]] = call i16 @llvm.[[INTRINSIC]].i16(i16 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret i16 %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint16
uint16_t test_uint16(uint16_t value) {
  // CHECK: %[[RET:.*]] = call i16 @llvm.[[INTRINSIC]].i16(i16 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret i16 %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_int64
int64_t test_int64(int64_t value) {
  // CHECK: %[[RET:.*]] = call i64 @llvm.[[INTRINSIC]].i64(i64 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret i64 %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint64
uint64_t test_uint64(uint64_t value) {
  // CHECK: %[[RET:.*]] = call i64 @llvm.[[INTRINSIC]].i64(i64 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret i64 %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint2
uint2 test_uint2(uint2 value) {
  // CHECK: %[[RET:.*]] = call <2 x i32> @llvm.[[INTRINSIC]].v2i32(<2 x i32> %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret <2 x i32> %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_int3
int3 test_int3(int3 value) {
  // CHECK: %[[RET:.*]] = call <3 x i32> @llvm.[[INTRINSIC]].v3i32(<3 x i32> %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret <3 x i32> %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint64_t4
uint64_t4 test_uint64_t4(uint64_t4 value) {
  // CHECK: %[[RET:.*]] = call <4 x i64> @llvm.[[INTRINSIC]].v4i64(<4 x i64> %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret <4 x i64> %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint5
vector<uint, 5> test_uint5(vector<uint, 5> value) {
  // CHECK: %[[RET:.*]] = call <5 x i32> @llvm.[[INTRINSIC]].v5i32(<5 x i32> %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret <5 x i32> %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_int2x3
int2x3 test_int2x3(int2x3 value) {
  // CHECK: %[[RET:.*]] = call <6 x i32> @llvm.[[INTRINSIC]].v6i32(<6 x i32> %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  // CHECK: ret <6 x i32> %[[RET]]
  return WaveMultiPrefixBitAnd(value, uint4(1, 2, 4, 8));
}

// CHECK: attributes #[[ATTR]] = { {{.*}}convergent {{.*}} }
