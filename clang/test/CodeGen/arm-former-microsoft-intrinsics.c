// RUN: %clang_cc1 -triple thumbv7-windows -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s

// RUN: %clang_cc1 -triple armv7-eabi -Werror -emit-llvm -o - %s \
// RUN:    | FileCheck %s
#include <arm_acle.h>
void check__dmb(void) {
  __dmb(0);
}

// CHECK: @llvm.arm.dmb(i32 0)

void check__dsb(void) {
  __dsb(0);
}

// CHECK: @llvm.arm.dsb(i32 0)

void check__isb(void) {
  __isb(0);
}

// CHECK: @llvm.arm.isb(i32 0)

void check__yield(void) {
  __yield();
}

// CHECK: @llvm.arm.hint(i32 1)

void check__wfe(void) {
  __wfe();
}

// CHECK: @llvm.arm.hint(i32 2)

void check__wfi(void) {
  __wfi();
}

// CHECK: @llvm.arm.hint(i32 3)

void check__sev(void) {
  __sev();
}

// CHECK: @llvm.arm.hint(i32 4)

void check__sevl(void) {
  __sevl();
}

// CHECK: @llvm.arm.hint(i32 5)
