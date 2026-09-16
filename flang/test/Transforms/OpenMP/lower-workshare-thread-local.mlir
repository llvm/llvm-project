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


// Check that hlfir.assign to a private clause variable through hlfir.declare
// is recognized as thread-local. The alias analysis marks private items as
// SourceKind::Allocate with the declare result as source value, so we need
// to trace through the declare to find the private block argument.

// CHECK-LABEL: func.func @hlfir_assign_private_clause
func.func @hlfir_assign_private_clause(%arg0: !fir.ref<i32>) {
  omp.parallel private(@x_private %arg0 -> %priv_arg : !fir.ref<i32>) {
    %decl:2 = hlfir.declare %priv_arg {uniq_name = "x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    omp.workshare {
      %c1 = arith.constant 1 : i32
      // hlfir.assign to private variable should NOT be in omp.single
      hlfir.assign %c1 to %decl#0 : i32, !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel private(@x_private %{{.*}} -> %[[PRIV_ARG:.*]] : !fir.ref<i32>) {
// CHECK-NEXT:    %[[DECL:.*]]:2 = hlfir.declare %[[PRIV_ARG]]
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    hlfir.assign %[[C1]] to %[[DECL]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }


// -----

// Check that hlfir.assign from a shared variable to a private variable is
// NOT parallelized. The assign has a Read effect on the shared RHS, which
// is not a write to thread-local memory, so isSafeToParallelize returns false.

omp.private {type = private} @y_private : i32

// CHECK-LABEL: func.func @hlfir_assign_shared_to_private
func.func @hlfir_assign_shared_to_private(%arg0: !fir.ref<i32>, %shared: !fir.ref<i32>) {
  omp.parallel private(@y_private %arg0 -> %priv_arg : !fir.ref<i32>) {
    %decl:2 = hlfir.declare %priv_arg {uniq_name = "x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    omp.workshare {
      // hlfir.assign with a shared RHS variable should stay in omp.single
      hlfir.assign %shared to %decl#0 : !fir.ref<i32>, !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel private(@y_private %{{.*}} -> %[[PRIV_ARG:.*]] : !fir.ref<i32>) {
// CHECK-NEXT:    %[[DECL:.*]]:2 = hlfir.declare %[[PRIV_ARG]]
// CHECK-NEXT:    omp.single nowait {
// CHECK:           hlfir.assign %{{.*}} to %[[DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>
// CHECK:           omp.terminator
// CHECK-NEXT:    }
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


// Check that stores to thread-local alloca are parallelized, but loads stay in
// omp.single. Only writes to thread-local memory are safe to parallelize;
// reads must remain in the single to preserve synchronization barriers.

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

// The store to thread-local memory is parallelized (outside the single),
// but the load remains inside the single to maintain synchronization. The
// store which depends on that load can only run on the thread executing the
// single, so the thread-local memory it updates is broadcast with copyprivate
// to keep the copies of the other threads in sync.

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[ALLOCA:.*]] = fir.alloca i32
// CHECK:         omp.single copyprivate(%[[ALLOCA]] -> @_workshare_copy_i32 : !fir.ref<i32>) {
// CHECK:           fir.store {{.*}} to %[[ALLOCA]] : !fir.ref<i32>
// CHECK:           fir.load %[[ALLOCA]] : !fir.ref<i32>
// CHECK:           fir.store {{.*}} to %[[ALLOCA]] : !fir.ref<i32>
// CHECK:           omp.terminator
// CHECK-NEXT:    }
// CHECK:         fir.store {{.*}} to %[[ALLOCA]] : !fir.ref<i32>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }


// Check the forall-in-workshare pattern: a sequential loop (fir.do_loop) with
// thread-local index stores, shared memory loads, and a workshare.loop_wrapper.
// This models the lowered IR for:
//   !$omp workshare
//   forall (i=1:n) a(:,i) = a(:,i) + 1
//   !$omp end workshare
//
// Loads from thread-local allocas must remain inside omp.single so that the
// single's barrier preserves thread synchronization.
// If reads were also safe-to-parallelize, the entire SingleRegion before the
// workshare.loop_wrapper could become fully parallelized, eliminating the
// omp.single and its implicit barrier. This caused race conditions on shared
// runtime data structures in forall-workshare patterns (see issue #143330).

// CHECK-LABEL: func.func @forall_pattern_in_workshare
func.func @forall_pattern_in_workshare(%shared: !fir.ref<i32>) {
  omp.parallel {
    %idx_alloca = fir.alloca i32 {bindc_name = "i", pinned}
    omp.workshare {
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      fir.do_loop %iv = %c1 to %c10 step %c1 {
        // Store loop index to thread-local alloca (must be parallelized)
        %iv_i32 = fir.convert %iv : (index) -> i32
        fir.store %iv_i32 to %idx_alloca : !fir.ref<i32>
        // Load from shared memory (must stay in omp.single)
        %shared_val = fir.load %shared : !fir.ref<i32>
        // Load from thread-local alloca (must stay in omp.single to
        // prevent SingleRegion elimination and barrier loss)
        %idx_val = fir.load %idx_alloca : !fir.ref<i32>
        // Side-effecting op using both values (must stay in omp.single)
        "test.side_effect"(%shared_val, %idx_val) : (i32, i32) -> ()
        // Workshared loop
        omp.workshare.loop_wrapper {
          omp.loop_nest (%j) : index = (%c1) to (%c10) inclusive step (%c1) {
            "test.inner"(%j) : (index) -> ()
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

// The thread-local load must remain inside omp.single along with the shared
// load and side-effecting op (preserving the single's barrier). The
// thread-local store is also cloned inside the single, but a parallel copy
// is placed after it so all threads update their own alloca.
// CHECK:       omp.parallel {
// CHECK:         %[[IDX:.*]] = fir.alloca i32 {bindc_name = "i", pinned}
// CHECK:         fir.do_loop
// The single contains the shared load, thread-local load, and side effect.
// The thread-local store is also cloned inside (harmless, one thread runs it).
// CHECK:           omp.single {
// CHECK:             fir.store {{.*}} to %[[IDX]] : !fir.ref<i32>
// CHECK:             fir.load %{{.*}} : !fir.ref<i32>
// CHECK:             fir.load %[[IDX]] : !fir.ref<i32>
// CHECK:             "test.side_effect"
// CHECK:             omp.terminator
// CHECK-NEXT:      }
// The parallelized copy of the store runs after the single's barrier,
// ensuring all threads update their own thread-local index alloca.
// CHECK:           fir.store {{.*}} to %[[IDX]] : !fir.ref<i32>
// CHECK:           omp.wsloop {
// CHECK:         }
// CHECK:         omp.barrier
// CHECK:       }

// Check the FORALL fetch-counter pattern: a thread-local counter which is
// read, incremented and written back from inside an omp.single.
//
//   !$omp workshare
//   forall (i=1:1)
//     forall (j=1:3)
//       a(:,i,j) = a(:,i,j) + 1
//     end forall
//   end forall
//   !$omp end workshare
//
// The increment can only be computed on the thread executing the omp.single,
// so the counter must be broadcast with copyprivate. Otherwise the threads
// which did not execute the omp.single keep a stale counter and fetch the
// wrong element on the following iterations. See issue #209942.

// CHECK-LABEL: func.func @forall_fetch_counter_in_workshare
func.func @forall_fetch_counter_in_workshare(%stack: !fir.ref<i32>) {
  omp.parallel {
    %counter = fir.alloca i64 {pinned}
    omp.workshare {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      fir.store %c0_i64 to %counter : !fir.ref<i64>
      fir.do_loop %iv = %c1 to %c3 step %c1 {
        %idx = fir.load %counter : !fir.ref<i64>
        %next = arith.addi %idx, %c1_i64 : i64
        fir.store %next to %counter : !fir.ref<i64>
        "test.fetch"(%stack, %idx) : (!fir.ref<i32>, i64) -> ()
        omp.workshare.loop_wrapper {
          omp.loop_nest (%j) : index = (%c1) to (%c3) inclusive step (%c1) {
            "test.inner"(%j) : (index) -> ()
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
// CHECK:         %[[COUNTER:.*]] = fir.alloca i64 {pinned}
// The reset of the counter is a write to thread-local memory whose operands
// are all available, so it is parallelized and all threads run it.
// CHECK:         fir.store %{{.*}} to %[[COUNTER]] : !fir.ref<i64>
// CHECK:         fir.do_loop
// CHECK:           omp.single copyprivate(%[[COUNTER]] -> @_workshare_copy_i64 : !fir.ref<i64>) {
// CHECK:             %[[IDX:.*]] = fir.load %[[COUNTER]] : !fir.ref<i64>
// CHECK:             %[[NEXT:.*]] = arith.addi %[[IDX]], %{{.*}} : i64
// CHECK:             fir.store %[[NEXT]] to %[[COUNTER]] : !fir.ref<i64>
// CHECK:             "test.fetch"
// CHECK:             omp.terminator
// CHECK-NEXT:      }
// The increment must not be repeated outside the single.
// CHECK-NOT:       fir.store {{.*}} to %[[COUNTER]]
// CHECK:           omp.wsloop {


// -----

// Check that a thread-local location written from within an omp.single but
// never read back by the team is NOT broadcasted with copyprivate. The store
// is not safe to parallelize on its own here because its value comes from a
// shared load that must stay in the omp.single, so it ends up executed by a
// single thread.

// CHECK-LABEL: func.func @write_only_thread_local_not_broadcast
func.func @write_only_thread_local_not_broadcast(%shared: !fir.ref<i32>) {
  omp.parallel {
    %tl = fir.alloca i32
    omp.workshare {
      %v = fir.load %shared : !fir.ref<i32>
      fir.store %v to %tl : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[TL:.*]] = fir.alloca i32
// The single carries no copyprivate: %[[TL]] is never read by the team.
// CHECK:         omp.single nowait {
// CHECK-NOT:       copyprivate
// CHECK:           fir.store %{{.*}} to %[[TL]] : !fir.ref<i32>
// CHECK:           omp.terminator
// CHECK-NEXT:    }
// CHECK-NEXT:    omp.barrier

// -----

// Same write from within an omp.single, but now the location is read back by
// the whole team after the omp.workshare region: it must be broadcasted so the
// threads which did not run the single do not observe a stale value.

// CHECK-LABEL: func.func @written_then_read_thread_local_is_broadcast
func.func @written_then_read_thread_local_is_broadcast(%shared: !fir.ref<i32>, %sink: !fir.ref<i32>) {
  omp.parallel {
    %tl = fir.alloca i32
    omp.workshare {
      %v = fir.load %shared : !fir.ref<i32>
      fir.store %v to %tl : !fir.ref<i32>
      omp.terminator
    }
    %r = fir.load %tl : !fir.ref<i32>
    fir.store %r to %sink : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[TL:.*]] = fir.alloca i32
// CHECK:         omp.single copyprivate(%[[TL]] -> @_workshare_copy_i32 : !fir.ref<i32>) {
// CHECK:           fir.store %{{.*}} to %[[TL]] : !fir.ref<i32>
// CHECK:           omp.terminator
// CHECK-NEXT:    }
// CHECK:         fir.load %[[TL]] : !fir.ref<i32>


// -----

// Check that the read which keeps a thread-local write live for broadcasting
// is still recognized when it goes through a fir.declare of the allocation
// (looking through declares/converts, as flang lowering routinely inserts
// them). Matching the raw load/store value instead would miss this read and
// drop the required broadcast.

// CHECK-LABEL: func.func @broadcast_when_read_through_declare
func.func @broadcast_when_read_through_declare(%shared: !fir.ref<i32>, %sink: !fir.ref<i32>) {
  omp.parallel {
    %tl = fir.alloca i32
    %d = fir.declare %tl {uniq_name = "tl"} : (!fir.ref<i32>) -> !fir.ref<i32>
    omp.workshare {
      %v = fir.load %shared : !fir.ref<i32>
      fir.store %v to %tl : !fir.ref<i32>
      omp.terminator
    }
    %r = fir.load %d : !fir.ref<i32>
    fir.store %r to %sink : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK:       %[[TL:.*]] = fir.alloca i32
// The broadcast copies the underlying allocation, not the fir.declare.
// CHECK:       omp.single copyprivate(%[[TL]] -> @_workshare_copy_i32 : !fir.ref<i32>) {

// -----

// Same, but now the write from within the omp.single goes through a
// fir.declare of the allocation while the read is direct.

// CHECK-LABEL: func.func @broadcast_when_written_through_declare
func.func @broadcast_when_written_through_declare(%shared: !fir.ref<i32>, %sink: !fir.ref<i32>) {
  omp.parallel {
    %tl = fir.alloca i32
    %d = fir.declare %tl {uniq_name = "tl"} : (!fir.ref<i32>) -> !fir.ref<i32>
    omp.workshare {
      %v = fir.load %shared : !fir.ref<i32>
      fir.store %v to %d : !fir.ref<i32>
      omp.terminator
    }
    %r = fir.load %tl : !fir.ref<i32>
    fir.store %r to %sink : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK:       %[[TL:.*]] = fir.alloca i32
// CHECK:       omp.single copyprivate(%[[TL]] -> @_workshare_copy_i32 : !fir.ref<i32>) {

// -----

// Check that a write performed from within an omp.single through an
// hlfir.declare of a privatizing clause block argument is still broadcasted
// when the location is read back through the block argument itself. The alias
// analysis reports the declare result for the write but the block argument for
// the read; getOpenMPThreadLocalSource canonicalizes both to the block
// argument so the broadcast is not dropped.

omp.private {type = private} @z_private : i32

// CHECK-LABEL: func.func @broadcast_private_write_through_declare_read_direct
func.func @broadcast_private_write_through_declare_read_direct(
    %arg0: !fir.ref<i32>, %shared: !fir.ref<i32>, %sink: !fir.ref<i32>) {
  omp.parallel private(@z_private %arg0 -> %priv_arg : !fir.ref<i32>) {
    %decl:2 = hlfir.declare %priv_arg {uniq_name = "z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    omp.workshare {
      // The value comes from a shared load, so the store stays in the single.
      %v = fir.load %shared : !fir.ref<i32>
      fir.store %v to %decl#0 : !fir.ref<i32>
      omp.terminator
    }
    // Read back the private variable directly through the block argument.
    %r = fir.load %priv_arg : !fir.ref<i32>
    fir.store %r to %sink : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK:       omp.parallel private(@z_private %{{.*}} -> %[[PRIV:.*]] : !fir.ref<i32>) {
// The broadcast copies the private block argument, matching the read.
// CHECK:         omp.single copyprivate(%[[PRIV]] -> @_workshare_copy_i32 : !fir.ref<i32>) {

// -----

// Check that a thread-local location written from within an omp.single but
// never read back through a visible load is still broadcasted when the team may
// run an opaque call: the callee might read the location, and that read cannot
// be attributed to a specific location. Contrast with
// @write_only_thread_local_not_broadcast, which has no call and is left with a
// plain omp.single.

func.func private @opaque(!fir.ref<i32>)

// CHECK-LABEL: func.func @opaque_call_forces_broadcast
func.func @opaque_call_forces_broadcast(%shared: !fir.ref<i32>) {
  omp.parallel {
    %tl = fir.alloca i32
    omp.workshare {
      %v = fir.load %shared : !fir.ref<i32>
      fir.store %v to %tl : !fir.ref<i32>
      omp.terminator
    }
    // The callee might read %tl, so the write must be broadcasted.
    fir.call @opaque(%tl) : (!fir.ref<i32>) -> ()
    omp.terminator
  }
  return
}

// CHECK:       %[[TL:.*]] = fir.alloca i32
// CHECK:         omp.single copyprivate(%[[TL]] -> @_workshare_copy_i32 : !fir.ref<i32>) {
// CHECK:           fir.store %{{.*}} to %[[TL]] : !fir.ref<i32>
