// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.DumpTraversal %s | FileCheck %s

void inline_callee(int i);

// CHECK: --BEGIN FUNCTION--
void inline_caller() {
  // CHECK: --BEGIN FUNCTION--
  // CHECK: --BEGIN FUNCTION--
  // CHECK: --BEGIN FUNCTION--
  inline_callee(3);
  // CHECK: --END FUNCTION--
  // CHECK: --END FUNCTION--
  // CHECK: --END FUNCTION--
}
// CHECK: --END FUNCTION--

void inline_callee(int i) {
  if (i <= 1)
    return;

  inline_callee(i - 1);
}
