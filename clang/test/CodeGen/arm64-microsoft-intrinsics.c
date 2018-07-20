// RUN: %clang_cc1 -triple arm64-windows -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-MSVC

// RUN: not %clang_cc1 -triple arm64-linux -Werror -S -o /dev/null %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-LINUX

void check__dmb(void) {
  __dmb(0);
}

// CHECK-MSVC: @llvm.aarch64.dmb(i32 0)
// CHECK-LINUX: error: implicit declaration of function '__dmb'

void check__dsb(void) {
  __dsb(0);
}

// CHECK-MSVC: @llvm.aarch64.dsb(i32 0)
// CHECK-LINUX: error: implicit declaration of function '__dsb'

void check__isb(void) {
  __isb(0);
}

// CHECK-MSVC: @llvm.aarch64.isb(i32 0)
// CHECK-LINUX: error: implicit declaration of function '__isb'

void check__yield(void) {
  __yield();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 1)
// CHECK-LINUX: error: implicit declaration of function '__yield'

void check__wfe(void) {
  __wfe();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 2)
// CHECK-LINUX: error: implicit declaration of function '__wfe'

void check__wfi(void) {
  __wfi();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 3)
// CHECK-LINUX: error: implicit declaration of function '__wfi'

void check__sev(void) {
  __sev();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 4)
// CHECK-LINUX: error: implicit declaration of function '__sev'

void check__sevl(void) {
  __sevl();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 5)
// CHECK-LINUX: error: implicit declaration of function '__sevl'
