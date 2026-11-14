// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -fnative-half-type -fnative-int16-type \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DINTRINSIC=dx.wave.multi.prefix.product \
// RUN:   -DUINTRINSIC=dx.wave.multi.prefix.uproduct
// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -fnative-half-type -fnative-int16-type \
// RUN:   -triple spirv-pc-vulkan-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DINTRINSIC=spv.subgroup.multi.prefix.product \
// RUN:   -DUINTRINSIC=spv.subgroup.multi.prefix.product

// CHECK-LABEL: test_float
float test_float(float value, uint4 mask) {
  // CHECK: %[[TOKEN:.*]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[MASK:.*]] = load <4 x i32>
  // CHECK: %[[M0:.*]] = extractelement <4 x i32> %[[MASK]], i64 0
  // CHECK: %[[M1:.*]] = extractelement <4 x i32> %[[MASK]], i64 1
  // CHECK: %[[M2:.*]] = extractelement <4 x i32> %[[MASK]], i64 2
  // CHECK: %[[M3:.*]] = extractelement <4 x i32> %[[MASK]], i64 3
  // CHECK: %[[RET:.*]] = call {{.*}}float @llvm.[[INTRINSIC]].f32(float %{{.*}}, i32 %[[M0]], i32 %[[M1]], i32 %[[M2]], i32 %[[M3]]) [ "convergencectrl"(token %[[TOKEN]]) ]
  // CHECK: ret float %[[RET]]
  return WaveMultiPrefixProduct(value, mask);
}

// CHECK: declare float @llvm.[[INTRINSIC]].f32(float, i32, i32, i32, i32) #[[ATTR:[0-9]+]]

// CHECK-LABEL: test_half
float16_t test_half(float16_t value) {
  // CHECK: call {{.*}}half @llvm.[[INTRINSIC]].f16(half %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_double
double test_double(double value) {
  // CHECK: call {{.*}}double @llvm.[[INTRINSIC]].f64(double %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_int
int test_int(int value) {
  // CHECK: call i32 @llvm.[[INTRINSIC]].i32(i32 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return __builtin_hlsl_wave_multi_prefix_product(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_int16
int16_t test_int16(int16_t value) {
  // CHECK: call i16 @llvm.[[INTRINSIC]].i16(i16 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_int64
int64_t test_int64(int64_t value) {
  // CHECK: call i64 @llvm.[[INTRINSIC]].i64(i64 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint
uint test_uint(uint value) {
  // CHECK: call i32 @llvm.[[UINTRINSIC]].i32(i32 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint16
uint16_t test_uint16(uint16_t value) {
  // CHECK: call i16 @llvm.[[UINTRINSIC]].i16(i16 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint64
uint64_t test_uint64(uint64_t value) {
  // CHECK: call i64 @llvm.[[UINTRINSIC]].i64(i64 %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_uint2
uint2 test_uint2(uint2 value, uint4 mask) {
  // CHECK: call <2 x i32> @llvm.[[UINTRINSIC]].v2i32(<2 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return WaveMultiPrefixProduct(value, mask);
}

// CHECK-LABEL: test_float4
float4 test_float4(float4 value, uint4 mask) {
  // CHECK: call {{.*}}<4 x float> @llvm.[[INTRINSIC]].v4f32(<4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return WaveMultiPrefixProduct(value, mask);
}

// CHECK-LABEL: test_uint_matrix
uint2x2 test_uint_matrix(uint2x2 value, uint4 mask) {
  // CHECK: call <4 x i32> @llvm.[[UINTRINSIC]].v4i32(<4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return WaveMultiPrefixProduct(value, mask);
}

// CHECK-LABEL: test_float_matrix
float2x2 test_float_matrix(float2x2 value, uint4 mask) {
  // CHECK: call {{.*}}<4 x float> @llvm.[[INTRINSIC]].v4f32(<4 x float> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return WaveMultiPrefixProduct(value, mask);
}

// CHECK-LABEL: test_uint5
vector<uint, 5> test_uint5(vector<uint, 5> value) {
  // CHECK: call <5 x i32> @llvm.[[UINTRINSIC]].v5i32(<5 x i32> %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_float5
vector<float, 5> test_float5(vector<float, 5> value) {
  // CHECK: call {{.*}}<5 x float> @llvm.[[INTRINSIC]].v5f32(<5 x float> %{{.*}}, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixProduct(value, uint4(1, 2, 4, 8));
}

// CHECK: attributes #[[ATTR]] = { convergent {{.*}}memory(none) }
