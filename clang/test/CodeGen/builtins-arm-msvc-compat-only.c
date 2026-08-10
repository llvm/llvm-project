// RUN: %clang_cc1 -triple thumbv7-windows -fms-extensions -emit-llvm -o - %s \
// RUN:     | FileCheck %s -check-prefix CHECK-MSVC
// RUN: not %clang_cc1 -triple armv7-eabi -emit-llvm %s -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefix CHECK-EABI
// REQUIRES: arm-registered-target

void emit() {
  __emit(0xdefe);
}

// CHECK-MSVC: call void asm sideeffect ".inst.n 0xDEFE", ""()
// CHECK-EABI: error: call to undeclared function '__emit'

void emit_truncated() {
  __emit(0x11110000); // movs r0, r0
}

// CHECK-MSVC: call void asm sideeffect ".inst.n 0x0", ""()

