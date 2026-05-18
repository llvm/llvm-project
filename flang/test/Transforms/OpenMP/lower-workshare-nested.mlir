// RUN: fir-opt --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Checks that the nested loop_wrapper gets parallelized
func.func @wsfunc(%cond : i1) {
  omp.workshare {
    %c1 = arith.constant 1 : index
    %c42 = arith.constant 42 : index
    fir.if %cond {
      omp.workshare.loop_wrapper {
        omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
          "test.test1"() : () -> ()
          omp.yield
        }
      }
    }
    omp.terminator
  }
  return
}

// CHECK:     fir.if
// CHECK:       omp.wsloop nowait
