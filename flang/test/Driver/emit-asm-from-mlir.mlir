// Verify that the driver can consume MLIR/FIR files.

// RUN: %flang_fc1 -S %s -o - | FileCheck %s

// CHECK-LABEL: foo:
// CHECK: ret
func.func @foo() {
  return
}
