// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Check that we correctly handle nowait

// CHECK-LABEL:   func.func @nonowait
func.func @nonowait(%arg0: !fir.ref<!fir.array<42xi32>>) {
  // CHECK: omp.barrier
  omp.workshare {
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL:   func.func @nowait
func.func @nowait(%arg0: !fir.ref<!fir.array<42xi32>>) {
  // CHECK-NOT: omp.barrier
  omp.workshare nowait {
    omp.terminator
  }
  return
}

// -----

// Check that nowait is not propagated into a region which is nested in
// something that is not itself the last piece of work of the omp.workshare
// region, or that may run more than once.

// CHECK-LABEL:   func.func @no_nowait_in_nested_conditional
func.func @no_nowait_in_nested_conditional(%arg0: !fir.ref<i32>, %cond: i1) {
  omp.parallel {
    omp.workshare {
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      fir.do_loop %i = %c1 to %c10 step %c1 {
        fir.if %cond {
          omp.workshare.loop_wrapper {
            omp.loop_nest (%j) : index = (%c1) to (%c10) inclusive step (%c1) {
              "test.inner"(%j) : (index) -> ()
              omp.yield
            }
          }
          "test.side_effect"(%arg0) : (!fir.ref<i32>) -> ()
        }
      }
      "test.after_loop"(%arg0) : (!fir.ref<i32>) -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       fir.do_loop
// CHECK:         fir.if
// CHECK:           omp.wsloop {
// CHECK-NOT:       nowait
// CHECK:           omp.single {
// CHECK:             "test.side_effect"
// CHECK:             omp.terminator
// CHECK-NEXT:      }
// The work after the loop is the last one, so it may use nowait and rely on
// the barrier at the end of the omp.workshare region.
// CHECK:       omp.single nowait {
// CHECK:         "test.after_loop"
// CHECK:         omp.terminator
// CHECK-NEXT:  }
// CHECK-NEXT:  omp.barrier
