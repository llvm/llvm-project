// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpTraversal %s | FileCheck %s

void inline_callee(int i);

// CHECK: --BEGIN FUNCTION--
void inline_caller(void) {
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
