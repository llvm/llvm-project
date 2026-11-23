// Test that saving and restoring the stack pointer, and reaching an
// unreachable terminator, do not modify or reference memory.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "stack_save_restore"
// CHECK-DAG: save -> x#0: NoModRef
// CHECK-DAG: restore -> x#0: NoModRef
func.func @stack_save_restore() {
  %x = fir.alloca f32 {test.ptr = "x"}
  %sp = llvm.intr.stacksave {test.ptr = "save"} : !llvm.ptr
  llvm.intr.stackrestore %sp {test.ptr = "restore"} : !llvm.ptr
  return
}

// -----

// CHECK-LABEL: Testing : "unreachable"
// CHECK: unreachable -> x#0: NoModRef
func.func @unreachable() {
  %x = fir.alloca f32 {test.ptr = "x"}
  fir.unreachable {test.ptr = "unreachable"}
}
