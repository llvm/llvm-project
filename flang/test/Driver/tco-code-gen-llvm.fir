// Test the tco's -code-gen-llvm option.

// RUN: tco -code-gen-llvm %s 2>&1 | FileCheck %s

// Check that FIR is translated into LLVM IR, and that
// there is no any FIR output.

// CHECK-NOT: func.func
// CHECK: define void @_QPfoo
// CHECK-NOT: func.func
func.func @_QPfoo() {
  return
}
