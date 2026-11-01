// RUN: %clang_cc1 -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// Every inline assembly statement clobbers the Y register, since code written
// for GCC can write it without listing it.

// CHECK-LABEL: @none
// CHECK: call void asm sideeffect "nop", "~{y}"()
void none() {
  __asm__ __volatile__("nop");
}

// CHECK-LABEL: @icc
// CHECK: call void asm sideeffect "nop", "~{icc},~{y}"()
void icc() {
  __asm__ __volatile__("nop" ::: "icc");
}

// CHECK-LABEL: @fcc
// CHECK: call void asm sideeffect "nop", "~{fcc0},~{fcc1},~{fcc2},~{fcc3},~{y}"()
void fcc() {
  __asm__ __volatile__("nop" ::: "fcc0", "fcc1", "fcc2", "fcc3");
}
