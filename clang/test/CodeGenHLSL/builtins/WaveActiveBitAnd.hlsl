// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK -DCALL="call"

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK -DCALL="call spir_func"

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_uint
uint test_uint(uint expr) {
  // DXCHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF:dx]].wave.reduce.and.i32([[TY]] %[[#]])
  // SPVCHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF:spv]].wave.reduce.and.i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint2
uint2 test_uint2(uint2 expr) {
  // CHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF]].wave.reduce.and.v2i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.v2i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint3
uint3 test_uint3(uint3 expr) {
  // CHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF]].wave.reduce.and.v3i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.v3i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint4
uint4 test_uint4(uint4 expr) {
  // CHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF]].wave.reduce.and.v4i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.v4i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t
uint64_t test_uint64_t(uint64_t expr) {
  // CHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF]].wave.reduce.and.i64([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.i64([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t2
uint64_t2 test_uint64_t2(uint64_t2 expr) {
  // CHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF]].wave.reduce.and.v2i64([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.v2i64([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t3
uint64_t3 test_uint64_t3(uint64_t3 expr) {
  // CHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF]].wave.reduce.and.v3i64([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.v3i64([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t4
uint64_t4 test_uint64_t4(uint64_t4 expr) {
  // CHECK:  %[[RET:.*]] = [[CALL]] [[TY:.*]] @llvm.[[ICF]].wave.reduce.and.v4i64([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitAnd(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.and.v4i64([[TY]]) #[[#attr:]]
