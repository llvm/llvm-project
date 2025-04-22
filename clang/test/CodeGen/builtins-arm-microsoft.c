// RUN: %clang_cc1 -triple thumbv7-windows -fms-compatibility -emit-llvm -o - %s \
// RUN:     | FileCheck %s -check-prefix CHECK-MSVC
// RUN: %clang_cc1 -Wno-implicit-function-declaration -triple armv7-eabi -emit-llvm %s -o - \
// RUN:     | FileCheck %s -check-prefix CHECK-EABI
// REQUIRES: arm-registered-target

#include <arm_acle.h>

void test_yield_intrinsic() {
  __yield();
}

// CHECK-MSVC: call void @llvm.arm.hint(i32 1)
// CHECK-EABI: call void @llvm.arm.hint(i32 1)

void wfe() {
  __wfe();
}

// CHECK-MSVC: call {{.*}} @llvm.arm.hint(i32 2)
// CHECK-EABI: call {{.*}} @llvm.arm.hint(i32 2)

void wfi() {
  __wfi();
}

// CHECK-MSVC: call {{.*}} @llvm.arm.hint(i32 3)
// CHECK-EABI: call {{.*}} @llvm.arm.hint(i32 3)

void sev() {
  __sev();
}

// CHECK-MSVC: call {{.*}} @llvm.arm.hint(i32 4)
// CHECK-EABI: call {{.*}} @llvm.arm.hint(i32 4)

void sevl() {
  __sevl();
}

// CHECK-MSVC: call {{.*}} @llvm.arm.hint(i32 5)
// CHECK-EABI: call {{.*}} @llvm.arm.hint(i32 5)

