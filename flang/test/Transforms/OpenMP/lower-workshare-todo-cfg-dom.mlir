// RUN: fir-opt --lower-workshare --allow-unregistered-dialect %s 2>&1 | FileCheck %s

// CHECK: warning: omp workshare with unstructured control flow is currently unsupported and will be serialized.

// CHECK: omp.parallel
// CHECK-NEXT: omp.single

// TODO Check that the definition of %r dominates its use post-transform
func.func @wsfunc() {
  %a = fir.alloca i32
  omp.parallel {
    omp.workshare {
    ^bb1:
      %c1 = arith.constant 1 : i32
      cf.br ^bb3(%c1: i32)
    ^bb2:
      "test.test2"(%r) : (i32) -> ()
      omp.terminator
    ^bb3(%arg1: i32):
      %r = "test.test2"(%arg1) : (i32) -> i32
      cf.br ^bb2
    }
    omp.terminator
  }
  return
}
