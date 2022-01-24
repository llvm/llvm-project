// RUN: tco --target=x86_64-unknown-linux-gnu %s | FileCheck %s

// Check that tco is working with a basic test.

func @_QQmain() {
  return
}

// CHECK: ; ModuleID = 'FIRModule'
// CHECK-LABEL: define void @_QQmain()
// CHECK:       ret void
