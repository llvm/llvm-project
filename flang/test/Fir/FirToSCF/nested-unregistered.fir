// Test conversions of nested ops, including a fir.do_loop nested inside an
// OpenACC kernels region.
//
// RUN: fir-opt %s --fir-to-scf | FileCheck %s

// CHECK-LABEL: func.func @nested_region
// CHECK-NOT:   fir.do_loop
// CHECK:       scf.for
// CHECK:       acc.kernels
// CHECK:       scf.for
// CHECK:       acc.terminator

func.func @nested_region() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  fir.do_loop %arg0 = %c1 to %c2 step %c1 {
    acc.kernels {
      fir.do_loop %arg1 = %c1 to %c2 step %c1 {
      }
      acc.terminator
    }
  }
  return
}


