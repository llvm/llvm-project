// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPV

// Test lowering of asdouble expansion to shuffle/bitcast and splat when required

// CHECK-LABEL: test_uint
double test_uint(uint low, uint high) {
  // CHECK-SPV: %[[LOW_INSERT:.*]] = insertelement <1 x i32>
  // CHECK-SPV: %[[LOW_SHUFFLE:.*]] = shufflevector <1 x i32> %[[LOW_INSERT]], {{.*}} zeroinitializer
  // CHECK-SPV: %[[HIGH_INSERT:.*]] = insertelement <1 x i32>
  // CHECK-SPV: %[[HIGH_SHUFFLE:.*]] = shufflevector <1 x i32> %[[HIGH_INSERT]], {{.*}} zeroinitializer

  // CHECK-SPV:      %[[SHUFFLE0:.*]] = shufflevector <1 x i32> %[[LOW_SHUFFLE]], <1 x i32> %[[HIGH_SHUFFLE]],
  // CHECK-SPV-SAME: {{.*}} <i32 0, i32 1>
  // CHECK-SPV:      bitcast <2 x i32> %[[SHUFFLE0]] to double

  // CHECK-DXIL: call reassoc nnan ninf nsz arcp afn double @llvm.dx.asdouble.i32
  return asdouble(low, high);
}

// CHECK-DXIL: declare double @llvm.dx.asdouble.i32

// CHECK-LABEL: test_vuint
double3 test_vuint(uint3 low, uint3 high) {
  // CHECK-SPV:      %[[SHUFFLE1:.*]] = shufflevector
  // CHECK-SPV-SAME: {{.*}} <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
  // CHECK-SPV:      bitcast <6 x i32> %[[SHUFFLE1]] to <3 x double>

  // CHECK-DXIL: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.dx.asdouble.v3i32
  return asdouble(low, high);
}

// CHECK-DXIL: declare <3 x double> @llvm.dx.asdouble.v3i32
