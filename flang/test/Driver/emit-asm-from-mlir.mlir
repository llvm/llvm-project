// REQUIRES: aarch64-registered-target || x86-registered-target || arm-registered-target
// Above restricts the test to those architectures that match "ret" to return
// from a function.
// Verify that the driver can consume MLIR/FIR files.

// RUN: %flang_fc1 -S %s -o - | FileCheck %s

// CHECK-LABEL: foo:
// CHECK: ret
func.func @foo() {
  return
}
