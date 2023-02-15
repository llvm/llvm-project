// REQUIRES: target=aarch64{{.*}} || target=x86{{.*}} || target=arm{{.*}}
// Above restricts the test to those architectures that match "ret" to return
// from a function.
// Verify that the driver can consume MLIR/FIR files.

// RUN: %flang_fc1 -S %s -o - | FileCheck %s

// CHECK-LABEL: foo:
// CHECK: ret
func.func @foo() {
  return
}
