// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s

// Test lowering of asdouble expansion to shuffle/bitcast and splat when required

// CHECK-LABEL: test_uint
double test_uint(uint low, uint high) {
  // CHECK: %[[LOW_INSERT:.*]] = insertelement <1 x i32>
  // CHECK: %[[LOW_SHUFFLE:.*]] = shufflevector <1 x i32> %[[LOW_INSERT]], {{.*}} zeroinitializer
  // CHECK: %[[HIGH_INSERT:.*]] = insertelement <1 x i32>
  // CHECK: %[[HIGH_SHUFFLE:.*]] = shufflevector <1 x i32> %[[HIGH_INSERT]], {{.*}} zeroinitializer

  // CHECK:      %[[SHUFFLE0:.*]] = shufflevector <1 x i32> %[[LOW_SHUFFLE]], <1 x i32> %[[HIGH_SHUFFLE]],
  // CHECK-SAME: {{.*}} <i32 0, i32 1>
  // CHECK:      bitcast <2 x i32> %[[SHUFFLE0]] to double
  return asdouble(low, high);
}

// CHECK-LABEL: test_vuint
double3 test_vuint(uint3 low, uint3 high) {
  // CHECK:      %[[SHUFFLE1:.*]] = shufflevector
  // CHECK-SAME: {{.*}} <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
  // CHECK:      bitcast <6 x i32> %[[SHUFFLE1]] to <3 x double>
  return asdouble(low, high);
}
