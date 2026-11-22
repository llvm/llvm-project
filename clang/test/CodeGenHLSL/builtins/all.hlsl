// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library \
// RUN:   %s -fnative-half-type -fnative-int16-type -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_half{{.*}}(half
// CHECK: [[CMP:%.*]] = fcmp {{.*}} une half {{.*}}, 0.000000e+00
// CHECK: ret i1 [[CMP]]
bool test_all_half(half value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_half2{{.*}}(<2 x half>
// CHECK: [[CMP:%.*]] = fcmp {{.*}} une <2 x half> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v2i1(<2 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_half2(half2 value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_int16_t3{{.*}}(<3 x i16>
// CHECK: [[CMP:%.*]] = icmp ne <3 x i16> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v3i1(<3 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_int16_t3(int16_t3 value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_float4{{.*}}(<4 x float>
// CHECK: [[CMP:%.*]] = fcmp {{.*}} une <4 x float> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v4i1(<4 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_float4(float4 value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_float17{{.*}}(<17 x float>
// CHECK: [[CMP:%.*]] = fcmp {{.*}} une <17 x float> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v17i1(<17 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_float17(vector<float, 17> value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_double2{{.*}}(<2 x double>
// CHECK: [[CMP:%.*]] = fcmp {{.*}} une <2 x double> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v2i1(<2 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_double2(double2 value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_int{{.*}}(i32
// CHECK: [[CMP:%.*]] = icmp ne i32 {{.*}}, 0
// CHECK: ret i1 [[CMP]]
bool test_all_int(int value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_uint3{{.*}}(<3 x i32>
// CHECK: [[CMP:%.*]] = icmp ne <3 x i32> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v3i1(<3 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_uint3(uint3 value) { return all(value); }

// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_int64_t4{{.*}}(<4 x i64>
// CHECK: [[CMP:%.*]] = icmp ne <4 x i64> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v4i1(<4 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_int64_t4(int64_t4 value) { return all(value); }

// NOTE: Bools are i32s in HLSL so thats why you see the zext ops in the IR.
// CHECK-LABEL: define hidden noundef i1 @{{.*}}test_all_bool4{{.*}}(<4 x i1>
// CHECK-SAME: noundef [[INPUT:%.*]])
// CHECK: [[INPUT_EXT:%.*]] = zext <4 x i1> [[INPUT]] to <4 x i32>
// CHECK-NEXT: store <4 x i32> [[INPUT_EXT]], ptr [[INPUT_ADDR:%.*]]
// CHECK-NEXT: [[INPUT_RAW:%.*]] = load <4 x i32>, ptr [[INPUT_ADDR]]
// CHECK-NEXT: [[WRAPPER_INPUT:%.*]] = icmp ne <4 x i32> [[INPUT_RAW]], zeroinitializer
// CHECK-NEXT: [[WRAPPER_EXT:%.*]] = zext <4 x i1> [[WRAPPER_INPUT]] to <4 x i32>
// CHECK-NEXT: store <4 x i32> [[WRAPPER_EXT]], ptr [[WRAPPER_ADDR:%.*]]
// CHECK-NEXT: [[WRAPPER_RAW:%.*]] = load <4 x i32>, ptr [[WRAPPER_ADDR]]
// CHECK-NEXT: [[HELPER_INPUT:%.*]] = icmp ne <4 x i32> [[WRAPPER_RAW]], zeroinitializer
// CHECK-NEXT: [[HELPER_EXT:%.*]] = zext <4 x i1> [[HELPER_INPUT]] to <4 x i32>
// CHECK-NEXT: store <4 x i32> [[HELPER_EXT]], ptr [[HELPER_ADDR:%.*]]
// CHECK-NEXT: [[HELPER_RAW:%.*]] = load <4 x i32>, ptr [[HELPER_ADDR]]
// CHECK-NEXT: [[REDUCE_INPUT:%.*]] = icmp ne <4 x i32> [[HELPER_RAW]], zeroinitializer
// CHECK: [[REDUCE:%.*]] = call noundef i1 @llvm.vector.reduce.and.v4i1(<4 x i1> [[REDUCE_INPUT]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_bool4(bool4 value) { return all(value); }
