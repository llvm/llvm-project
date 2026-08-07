// exec_if lowers to the predicated-goto + join diamond.
// RUN: inter-translate --xemachine-to-iga %s | FileCheck %s

// CHECK: (~f0.0) goto (32|M0)  L1  L1
// CHECK: goto (32|M0)  L1  L2
// CHECK: L1:
// CHECK: join (32|M0)  L2
// CHECK: L2:
// CHECK: join (32|M0)  L3
// CHECK: L3:
func.func @k(%f: !xemachine.arf<f, 2, 0>) {
  xemachine.exec_if %f : !xemachine.arf<f, 2, 0> {
    xemachine.yield
  } otherwise {
    xemachine.yield
  }
  return
}
