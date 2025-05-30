// RUN: %clang_cc1 -triple thumbv7-windows -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-MSVC

// RUN: not %clang_cc1 -triple armv7-eabi -Werror -S -o /dev/null %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EABI

void check__dmb(void) {
  __dmb(0);
}

// CHECK-MSVC: @llvm.arm.dmb(i32 0)
// CHECK-EABI: error: call to undeclared function '__dmb'

void check__dsb(void) {
  __dsb(0);
}

// CHECK-MSVC: @llvm.arm.dsb(i32 0)
// CHECK-EABI: error: call to undeclared function '__dsb'

void check__isb(void) {
  __isb(0);
}

// CHECK-MSVC: @llvm.arm.isb(i32 0)
// CHECK-EABI: error: call to undeclared function '__isb'

__INT64_TYPE__ check__ldrexd(void) {
  __INT64_TYPE__ i64;
  return __ldrexd(&i64);
}

// CHECK-MSVC: @llvm.arm.ldrexd(ptr {{.*}})
// CHECK-EABI: error: call to undeclared function '__ldrexd'

unsigned int check_MoveFromCoprocessor(void) {
  return _MoveFromCoprocessor(0, 0, 0, 0, 0);
}

// CHECK-MSVC: @llvm.arm.mrc(i32 0, i32 0, i32 0, i32 0, i32 0)
// CHECK-EABI: error: call to undeclared function '_MoveFromCoprocessor'

unsigned int check_MoveFromCoprocessor2(void) {
  return _MoveFromCoprocessor2(0, 0, 0, 0, 0);
}

// CHECK-MSVC: @llvm.arm.mrc2(i32 0, i32 0, i32 0, i32 0, i32 0)
// CHECK-EABI: error: call to undeclared function '_MoveFromCoprocessor2'

void check_MoveToCoprocessor(unsigned int value) {
  _MoveToCoprocessor(value, 10, 7, 1, 0, 0);
}

// CHECK-MSVC: @llvm.arm.mcr(i32 10, i32 7, i32 %{{[^,]*}}, i32 1, i32 0, i32 0)
// CHECK-EABI: error: call to undeclared function '_MoveToCoprocessor'

void check_MoveToCoprocessor2(unsigned int value) {
  _MoveToCoprocessor2(value, 10, 7, 1, 0, 0);
}

// CHECK-MSVC: @llvm.arm.mcr2(i32 10, i32 7, i32 %{{[^,]*}}, i32 1, i32 0, i32 0)
// CHECK-EABI: error: call to undeclared function '_MoveToCoprocessor2'

