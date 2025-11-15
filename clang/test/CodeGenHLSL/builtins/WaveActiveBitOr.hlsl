// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// DXCHECK: define hidden [[FN_TYPE:]]noundef i1 @
// SPVCHECK: define hidden [[FN_TYPE:spir_func ]]noundef i1 @

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_uint
uint test_uint(uint expr) {
  // DXCHECK:  %[[RET:.*]] = call i1 [[TY:.*]] @llvm.[[ICF:dx]].wave.reduce.or.i32([[TY]] %[[#]])
  // SPVCHECK:  %[[RET:.*]] = call i1 [[TY:.*]] @llvm.[[ICF:spv]].wave.reduce.or.i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK: declare [[TY]] @llvm.[[ICF]].wave.reduce.or.i32([[TY]]) #[[#attr:]]
