// RUN: %clang_cc1 -triple riscv32 -emit-llvm -fcf-protection=branch -mcf-branch-label-scheme=func-sig %s -o - | FileCheck %s
// cfi-unchecked is for software CFI and should not disable zicfilp-related functionality.
typedef void (*CB)(int) __attribute__((cfi_unchecked_callee));
int test_unchecked_indirect(CB f, int x) {
  f(x);
  return 0;
}
// CHECK: call void %{{.*}}(i32 noundef %{{.*}}), !riscv_lpad_label [[LABEL:![0-9]+]]
// CHECK: [[LABEL]] = !{i32 {{[1-9][0-9]*}}}