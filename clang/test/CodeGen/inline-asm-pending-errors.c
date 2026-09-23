// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 %s 2>&1 | FileCheck %s

// PR225036: When inline assembly contains both a parse-time directive error
// (e.g., non-absolute subsection number in .text) and a layout-time directive error
// (e.g., non-absolute fill expression in .zero), MCAssembler must flush pending
// errors rather than asserting PendingErrors.empty() in Finish().

void test_pending_errors(int b) {
  // CHECK: error: cannot evaluate subsection number
  // CHECK: error: expected assembly-time absolute expression
  asm(".text a\n.zero %0" : : "r"(b));
}
