// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Check that we do not emit an omp.single for the constant operation

func.func @foo() {
  omp.workshare {
    %c1 = arith.constant 1 : index
    omp.workshare.loop_wrapper {
      omp.loop_nest (%arg1) : index = (%c1) to (%c1) inclusive step (%c1) {
        "test.test0"() : () -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// CHECK-NOT: omp.single
