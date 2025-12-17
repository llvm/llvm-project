// RUN: %clang_cc1 -triple arm64-windows -Wno-implicit-function-declaration -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s

// RUN: %clang_cc1 -triple arm64-linux -Werror -emit-llvm -o - %s 2>&1 \
// RUN:    | FileCheck %s

// RUN: %clang_cc1 -triple arm64-darwin -Wno-implicit-function-declaration -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s

#include <arm_acle.h>

void check__dmb(void) {
  __dmb(0);
}

// CHECK: @llvm.aarch64.dmb(i32 0)

void check__dsb(void) {
  __dsb(0);
}

// CHECK: @llvm.aarch64.dsb(i32 0)

void check__isb(void) {
  __isb(0);
}

// CHECK: @llvm.aarch64.isb(i32 0)

void check__yield(void) {
  __yield();
}

// CHECK: @llvm.aarch64.hint(i32 1)

void check__wfe(void) {
  __wfe();
}

// CHECK: @llvm.aarch64.hint(i32 2)

void check__wfi(void) {
  __wfi();
}

// CHECK: @llvm.aarch64.hint(i32 3)

void check__sev(void) {
  __sev();
}

// CHECK: @llvm.aarch64.hint(i32 4)

void check__sevl(void) {
  __sevl();
}

// CHECK: @llvm.aarch64.hint(i32 5)

