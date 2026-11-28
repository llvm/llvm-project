// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_all_half2x3
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} une <6 x half> {{.*}}, zeroinitializer
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} une <6 x float> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call {{.*}}i1 @llvm.vector.reduce.and.v6i1(<6 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_half2x3(half2x3 value) { return all(value); }

// CHECK-LABEL: test_all_half3x4
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} une <12 x half> {{.*}}, zeroinitializer
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} une <12 x float> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call {{.*}}i1 @llvm.vector.reduce.and.v12i1(<12 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_half3x4(half3x4 value) { return all(value); }

// CHECK-LABEL: test_all_half4x4
// NATIVE_HALF: [[CMP:%.*]] = fcmp {{.*}} une <16 x half> {{.*}}, zeroinitializer
// NO_HALF: [[CMP:%.*]] = fcmp {{.*}} une <16 x float> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call {{.*}}i1 @llvm.vector.reduce.and.v16i1(<16 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_half4x4(half4x4 value) { return all(value); }

// CHECK-LABEL: test_all_float2x3
// CHECK: [[CMP:%.*]] = fcmp {{.*}} une <6 x float> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call {{.*}}i1 @llvm.vector.reduce.and.v6i1(<6 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_float2x3(float2x3 value) { return all(value); }

// CHECK-LABEL: test_all_double3x4
// CHECK: [[CMP:%.*]] = fcmp {{.*}} une <12 x double> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call {{.*}}i1 @llvm.vector.reduce.and.v12i1(<12 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_double3x4(double3x4 value) { return all(value); }

// CHECK-LABEL: test_all_int4x4
// CHECK: [[CMP:%.*]] = icmp ne <16 x i32> {{.*}}, zeroinitializer
// CHECK: [[REDUCE:%.*]] = call {{.*}}i1 @llvm.vector.reduce.and.v16i1(<16 x i1> [[CMP]])
// CHECK: ret i1 [[REDUCE]]
bool test_all_int4x4(int4x4 value) { return all(value); }

// CHECK-LABEL: test_all_bool2x3
// CHECK: [[REDUCE:%.*]] = call {{.*}}i1 @llvm.vector.reduce.and.v6i1(<6 x i1> %{{.*}})
// CHECK: ret i1 [[REDUCE]]
bool test_all_bool2x3(bool2x3 value) { return all(value); }
