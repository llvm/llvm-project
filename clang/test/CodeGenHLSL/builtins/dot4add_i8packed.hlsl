// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DTARGET=spv

// Test basic lowering to runtime function call.

// CHECK-LABEL: test
int test(uint a, uint b, int c) {
  // CHECK:  %[[RET:.*]] = call [[TY:i32]] @llvm.[[TARGET]].dot4add.i8packed([[TY]] %[[#]], [[TY]] %[[#]], [[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return dot4add_i8packed(a, b, c);
}

// CHECK: declare [[TY]] @llvm.[[TARGET]].dot4add.i8packed([[TY]], [[TY]], [[TY]])
