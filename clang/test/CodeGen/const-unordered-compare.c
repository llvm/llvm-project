// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// Checks folding of an unordered comparison
int nan_ne_check(void) {
  // CHECK: ret i32 1
  return (__builtin_nanf("") != __builtin_nanf("")) ? 1 : 0;
}
