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
// but the load remains inside the single to maintain synchronization.

// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[ALLOCA:.*]] = fir.alloca i32
// CHECK:         omp.single nowait {
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


// Check that a store through a pointer dereference is NOT considered
// thread-local, even if the pointer descriptor itself is in a thread-local
// alloca. This models the "parallel workshare firstprivate(P)" case where P
// is a Fortran POINTER: each thread gets its own copy of the descriptor, but
// P(i) accesses shared target data through the pointer.

// CHECK-LABEL: func.func @pointer_deref_not_thread_local
func.func @pointer_deref_not_thread_local() {
  omp.parallel {
    // Thread-local alloca for the pointer descriptor (models firstprivate)
    %desc = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
    %decl = fir.declare %desc {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "p"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    omp.workshare {
      // Load the pointer box and access the target data via array_coor.
      // Even though %desc is thread-local, the target data is shared.
      %box = fir.load %decl : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
      %c0 = arith.constant 0 : index
      %dims:3 = fir.box_dims %box, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
      %shift = fir.shift %dims#0 : (index) -> !fir.shift<1>
      %c1_i64 = arith.constant 1 : i64
      %elem = fir.array_coor %box(%shift) %c1_i64 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, i64) -> !fir.ref<i32>
      %c42 = arith.constant 42 : i32
      // This store goes to shared target data (through pointer deref),
      // so it MUST be in omp.single, not parallelized.
      fir.store %c42 to %elem : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// The store through the pointer dereference must be inside omp.single.
// CHECK:       omp.parallel {
// CHECK:         fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
// CHECK:         omp.single
// CHECK:           fir.load
// CHECK:           fir.box_dims
// CHECK:           fir.array_coor
// CHECK:           fir.store
// CHECK:           omp.terminator
// CHECK-NEXT:    }
// CHECK:         omp.barrier
// CHECK:       }


// Check that a direct store to the pointer descriptor alloca (not through
// the pointer target) IS still recognized as thread-local.

// CHECK-LABEL: func.func @pointer_descriptor_store_is_thread_local
func.func @pointer_descriptor_store_is_thread_local() {
  omp.parallel {
    %desc = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
    omp.workshare {
      %null = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
      %c0 = arith.constant 0 : index
      %shape = fir.shape %c0 : (index) -> !fir.shape<1>
      %box = fir.embox %null(%shape) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
      // This store updates the descriptor itself (thread-local alloca),
      // NOT the pointer target, so it should be parallelized.
      fir.store %box to %desc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// The store to the descriptor alloca is thread-local and should NOT be in omp.single.
// CHECK:       omp.parallel {
// CHECK-NEXT:    %[[DESC:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
// CHECK:         fir.zero_bits
// CHECK:         fir.shape
// CHECK:         fir.embox
// CHECK:         fir.store {{.*}} to %[[DESC]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK-NEXT:    omp.barrier
// CHECK-NEXT:    omp.terminator
// CHECK-NEXT:  }

// -----

// Test for "parallel workshare firstprivate(z)" where z is an array.
// Check that z is broadcast to all private values of the threads.

// The copy function uses _FortranAAssign to broadcast array data.
// CHECK-LABEL: func.func private @_workshare_copy_data_box_Uxi32(
// CHECK: fir.call @_FortranAAssign
// CHECK: return

// CHECK-LABEL: func.func @dynamic_alloca_firstprivate_array(
// CHECK-SAME: %[[N:.*]]: index,
// CHECK-SAME: %[[SRC:.*]]: !fir.ref<!fir.array<?xi32>>,
// CHECK-SAME: %[[DST:.*]]: !fir.ref<!fir.array<?xi32>>)
func.func @dynamic_alloca_firstprivate_array(%n: index, %src: !fir.ref<!fir.array<?xi32>>, %dst: !fir.ref<!fir.array<?xi32>>) {
  omp.parallel {
    omp.workshare {
      // Dynamic alloca for the firstprivate array copy
      %z = fir.alloca !fir.array<?xi32>, %n {bindc_name = "z", pinned}
      %shape = fir.shape %n : (index) -> !fir.shape<1>
      %decl = fir.declare %z(%shape) {uniq_name = "z"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xi32>>
      // A side-effecting op that initializes the firstprivate copy
      "test.init"(%decl, %src) : (!fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>) -> ()
      // Workshared loop that reads from the firstprivate array
      %c1 = arith.constant 1 : index
      omp.workshare.loop_wrapper {
        omp.loop_nest (%i) : index = (%c1) to (%n) inclusive step (%c1) {
          %elem = fir.array_coor %decl(%shape) %i : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
          %val = fir.load %elem : !fir.ref<i32>
          %dst_elem = fir.array_coor %dst(%shape) %i : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
          fir.store %val to %dst_elem : !fir.ref<i32>
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK: omp.parallel {

// The dynamic alloca is hoisted so each thread gets its own allocation.
// CHECK: %[[ARRAY:.*]] = fir.alloca !fir.array<?xi32>, %[[N]]
// CHECK-SAME: bindc_name = "z"
// CHECK-SAME: pinned

// A box slot is used for copyprivate to broadcast the array data.
// CHECK: %[[BOX_SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK: %[[SHAPE0:.*]] = fir.shape %[[N]]
// CHECK: %[[BOX:.*]] = fir.embox %[[ARRAY]](%[[SHAPE0]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK: fir.store %[[BOX]] to %[[BOX_SLOT]] : !fir.ref<!fir.box<!fir.array<?xi32>>>

// Initialization happens only inside the single block.
// CHECK: omp.single copyprivate(%[[BOX_SLOT]] -> @_workshare_copy_data_box_Uxi32 : !fir.ref<!fir.box<!fir.array<?xi32>>>) {
// CHECK: %[[SHAPE1:.*]] = fir.shape %[[N]]
// CHECK: %[[DECL1:.*]] = fir.declare %[[ARRAY]](%[[SHAPE1]])
// CHECK: "test.init"(%[[DECL1]], %[[SRC]])
// CHECK: omp.terminator
// CHECK: }

// The workshared loop uses the hoisted per-thread array.
// CHECK: %[[SHAPE2:.*]] = fir.shape %[[N]]
// CHECK: %[[DECL2:.*]] = fir.declare %[[ARRAY]](%[[SHAPE2]])
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: omp.wsloop nowait
// CHECK: omp.loop_nest (%[[I:.*]]) : index = (%[[C1]]) to (%[[N]]) inclusive step (%[[C1]])
// CHECK: %[[ELEM:.*]] = fir.array_coor %[[DECL2]](%[[SHAPE2]]) %[[I]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK: %[[VAL:.*]] = fir.load %[[ELEM]] : !fir.ref<i32>
// CHECK: %[[DST_ELEM:.*]] = fir.array_coor %[[DST]](%[[SHAPE2]]) %[[I]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK: fir.store %[[VAL]] to %[[DST_ELEM]] : !fir.ref<i32>
// CHECK: omp.yield

// CHECK: omp.barrier
// CHECK: omp.terminator
// CHECK: return
