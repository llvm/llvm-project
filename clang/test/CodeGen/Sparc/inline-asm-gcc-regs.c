// RUN: %clang_cc1 -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: @icc
// CHECK: call void asm sideeffect "nop", "~{icc}"()
void icc() {
  __asm__ __volatile__("nop" ::: "icc");
}

// CHECK-LABEL: @fcc
// CHECK: call void asm sideeffect "nop", "~{fcc0},~{fcc1},~{fcc2},~{fcc3}"()
void fcc() {
  __asm__ __volatile__("nop" ::: "fcc0", "fcc1", "fcc2", "fcc3");
}
