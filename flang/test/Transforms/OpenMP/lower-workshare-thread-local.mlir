// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Tests for thread-local memory handling in workshare lowering (#143330):
// 1. Thread-local variables (from fir.alloca in omp.parallel or from OpenMP
//    private/reduction clauses) should be parallelized, not wrapped in omp.single
// 2. nowait should not be added to omp.single when inside loop-like operations
//    that contain omp.workshare.loop_wrapper


// Check that fir.alloca inside omp.parallel creates thread-local memory,
// and stores to it should be parallelized (not wrapped in omp.single).

// CHECK-LABEL: func.func @thread_local_alloca_store
func.func @thread_local_alloca_store() {
  omp.parallel {
    // The alloca is inside omp.parallel, so it's thread-local
    %alloca = fir.alloca i32
    omp.workshare {
      %c1 = arith.constant 1 : i32
      // This store should NOT be in omp.single because %alloca is thread-local
      fir.store %c1 to %alloca : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[ALLOCA:.*]] = fir.alloca i32
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    fir.store %[[C1]] to %[[ALLOCA]] : !fir.ref<i32>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }


// Check that memory accessed through fir.declare is also recognized as thread-local
// when the underlying alloca is in the parallel region.

// CHECK-LABEL: func.func @thread_local_with_declare
func.func @thread_local_with_declare() {
  omp.parallel {
    %alloca = fir.alloca i32
    %declare = fir.declare %alloca {uniq_name = "local_var"} : (!fir.ref<i32>) -> !fir.ref<i32>
    omp.workshare {
      %c42 = arith.constant 42 : i32
      // Store through declare should still be recognized as thread-local
      fir.store %c42 to %declare : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[ALLOCA:.*]] = fir.alloca i32
// CHECK-NEXT:    %[[DECLARE:.*]] = fir.declare %[[ALLOCA]]
// CHECK-NEXT:    %[[C42:.*]] = arith.constant 42 : i32
// CHECK-NEXT:    fir.store %[[C42]] to %[[DECLARE]] : !fir.ref<i32>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }


// Check that private clause block arguments are recognized as thread-local.

omp.private {type = private} @x_private : i32

// CHECK-LABEL: func.func @private_clause_thread_local
func.func @private_clause_thread_local(%arg0: !fir.ref<i32>) {
  omp.parallel private(@x_private %arg0 -> %priv_arg : !fir.ref<i32>) {
    omp.workshare {
      %c10 = arith.constant 10 : i32
      // Store to private variable should NOT be in omp.single
      fir.store %c10 to %priv_arg : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel private(@x_private %{{.*}} -> %[[PRIV_ARG:.*]] : !fir.ref<i32>) {
// CHECK-NEXT:    %[[C10:.*]] = arith.constant 10 : i32
// CHECK-NEXT:    fir.store %[[C10]] to %[[PRIV_ARG]] : !fir.ref<i32>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }


// Check that reduction clause block arguments are recognized as thread-local.

omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %c0 = arith.constant 0 : i32
  omp.yield(%c0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = arith.addi %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}

// CHECK-LABEL: func.func @reduction_clause_thread_local
func.func @reduction_clause_thread_local(%arg0: !fir.ref<i32>) {
  omp.parallel reduction(@add_reduction_i32 %arg0 -> %red_arg : !fir.ref<i32>) {
    omp.workshare {
      %c5 = arith.constant 5 : i32
      // Store to reduction variable should NOT be in omp.single
      fir.store %c5 to %red_arg : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel reduction(@add_reduction_i32 %{{.*}} -> %[[RED_ARG:.*]] : !fir.ref<i32>) {
// CHECK-NEXT:    %[[C5:.*]] = arith.constant 5 : i32
// CHECK-NEXT:    fir.store %[[C5]] to %[[RED_ARG]] : !fir.ref<i32>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }


// Check that nowait is NOT added to omp.single when inside fir.do_loop
// that contains omp.workshare.loop_wrapper. This prevents race conditions
// when multiple threads execute different loop iterations concurrently.
// The workshare.loop_wrapper triggers recursive parallelization of the loop body.

// CHECK-LABEL: func.func @no_nowait_in_loop_with_workshare_wrapper
func.func @no_nowait_in_loop_with_workshare_wrapper(%arg0: !fir.ref<i32>) {
  omp.parallel {
    omp.workshare {
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      fir.do_loop %i = %c1 to %c10 step %c1 {
        // This side-effecting op will be wrapped in omp.single without nowait
        "test.side_effect"(%arg0) : (!fir.ref<i32>) -> ()
        // The workshare.loop_wrapper triggers recursive processing of the loop
        omp.workshare.loop_wrapper {
          omp.loop_nest (%j) : index = (%c1) to (%c10) inclusive step (%c1) {
            "test.inner"() : () -> ()
            omp.yield
          }
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// The omp.single inside the loop should NOT have nowait
// CHECK:       omp.parallel {
// CHECK:         fir.do_loop
// CHECK:           omp.single {
// CHECK:             "test.side_effect"
// CHECK:             omp.terminator
// CHECK-NEXT:      }
// CHECK:           omp.wsloop {
// CHECK:         }
// CHECK:         omp.barrier
// CHECK:       }


// Check that thread-local store inside a loop with workshare.loop_wrapper
// is correctly parallelized (not wrapped in omp.single).

// CHECK-LABEL: func.func @thread_local_store_in_loop_with_wrapper
func.func @thread_local_store_in_loop_with_wrapper() {
  omp.parallel {
    %alloca = fir.alloca i32
    omp.workshare {
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      fir.do_loop %i = %c1 to %c10 step %c1 {
        %c99 = arith.constant 99 : i32
        // Store to thread-local alloca should NOT be in omp.single
        fir.store %c99 to %alloca : !fir.ref<i32>
        omp.workshare.loop_wrapper {
          omp.loop_nest (%j) : index = (%c1) to (%c10) inclusive step (%c1) {
            "test.inner"() : () -> ()
            omp.yield
          }
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[ALLOCA:.*]] = fir.alloca i32
// CHECK:         fir.do_loop
// The store should be outside omp.single
// CHECK:           %[[C99:.*]] = arith.constant 99 : i32
// CHECK-NEXT:      fir.store %[[C99]] to %[[ALLOCA]] : !fir.ref<i32>
// CHECK:           omp.wsloop {
// CHECK:         }
// CHECK:         omp.barrier
// CHECK:       }


// Check that non-thread-local memory is still wrapped in omp.single.
// This is the baseline case to ensure we haven't broken normal behavior.

// CHECK-LABEL: func.func @non_thread_local_needs_single
func.func @non_thread_local_needs_single(%arg0: !fir.ref<i32>) {
  omp.parallel {
    omp.workshare {
      %c1 = arith.constant 1 : i32
      // arg0 is shared memory, store must be in omp.single
      fir.store %c1 to %arg0 : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel {
// CHECK-NEXT:    omp.single nowait {
// CHECK-NEXT:      %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      fir.store %[[C1]] to %{{.*}} : !fir.ref<i32>
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }


// Check that loads from thread-local alloca combined with stores are parallelized.

// CHECK-LABEL: func.func @thread_local_load_and_store
func.func @thread_local_load_and_store() {
  omp.parallel {
    %alloca = fir.alloca i32
    omp.workshare {
      %c1 = arith.constant 1 : i32
      fir.store %c1 to %alloca : !fir.ref<i32>
      %val = fir.load %alloca : !fir.ref<i32>
      fir.store %val to %alloca : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// All operations on thread-local memory should be outside omp.single

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[ALLOCA:.*]] = fir.alloca i32
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    fir.store %[[C1]] to %[[ALLOCA]] : !fir.ref<i32>
// CHECK-NEXT:    %[[VAL:.*]] = fir.load %[[ALLOCA]] : !fir.ref<i32>
// CHECK-NEXT:    fir.store %[[VAL]] to %[[ALLOCA]] : !fir.ref<i32>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }
