// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_uint
uint test_uint(uint expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.reduce.or.i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.reduce.or.i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.reduce.or.i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.reduce.or.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t
uint64_t test_uint64_t(uint64_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.reduce.or.i64([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.reduce.or.i64([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.reduce.or.i64([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.reduce.or.i64([[TY]]) #[[#attr:]]
