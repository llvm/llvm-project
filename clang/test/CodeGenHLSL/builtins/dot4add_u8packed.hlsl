// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.4-compute %s -emit-llvm -o - | FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan-compute %s -emit-llvm -o - | FileCheck %s -DTARGET=spv

// Test basic lowering to runtime function call.

// CHECK-LABEL: test
int test(uint x, uint y, int acc) {
  // CHECK:    [[X_ADDR:%.*]] = alloca i32, align 4
  // CHECK:    [[Y_ADDR:%.*]] = alloca i32, align 4
  // CHECK:    [[ACC_ADDR:%.*]] = alloca i32, align 4
  // CHECK:    store i32 %x, ptr [[X_ADDR]], align 4
  // CHECK:    store i32 %y, ptr [[Y_ADDR]], align 4
  // CHECK:    store i32 %acc, ptr [[ACC_ADDR]], align 4
  // CHECK:    [[X0:%.*]] = load i32, ptr [[X_ADDR]], align 4
  // CHECK:    [[Y0:%.*]] = load i32, ptr [[Y_ADDR]], align 4
  // CHECK:    [[ACC0:%.*]] = load i32, ptr [[ACC_ADDR]], align 4
  // CHECK:    call i32 @llvm.[[TARGET]].dot4add.u8packed(i32 [[ACC0]], i32 [[X0]], i32 [[Y0]])
  return dot4add_u8packed(x, y, acc);
}

[numthreads(1,1,1)]
void main() {
  test(0, 0, 0);
}
