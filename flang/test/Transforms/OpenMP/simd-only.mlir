// RUN: fir-opt --split-input-file --verify-diagnostics --omp-simd-only %s | FileCheck %s

// Check that simd operations are not removed and rewritten, but all the other OpenMP ops are.

// CHECK-LABEL: func.func @simd
omp.private {type = private} @_QFEi_private_i32 : i32
func.func @simd(%arg0: i32, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c100000_i32 = arith.constant 100000 : i32
  // CHECK: omp.simd private
  omp.simd private(@_QFEi_private_i32 %arg2 -> %arg3 : !fir.ref<i32>) {
    // CHECK: omp.loop_nest
    omp.loop_nest (%arg4) : i32 = (%c1_i32) to (%c100000_i32) inclusive step (%c1_i32) {
      // CHECK: fir.store
      fir.store %arg0 to %arg1 : !fir.ref<i32>
      // CHECK: omp.yield
      omp.yield
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @simd_composite
func.func @simd_composite(%arg0: i32, %arg1: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c100000_i32 = arith.constant 100000 : i32
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK-NOT: omp.wsloop
    omp.wsloop {
      // CHECK: omp.simd
      omp.simd {
        // CHECK: omp.loop_nest
        omp.loop_nest (%arg3) : i32 = (%c1_i32) to (%c100000_i32) inclusive step (%c1_i32) {
          // CHECK: fir.store
          fir.store %arg0 to %arg1 : !fir.ref<i32>
          // CHECK: omp.yield
          omp.yield
        }
      // CHECK-NOT: {omp.composite}
      } {omp.composite}
    } {omp.composite}
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @parallel
omp.private {type = private} @_QFEi_private_i32 : i32
func.func @parallel(%arg0: i32, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c100000_i32 = arith.constant 100000 : i32
  // CHECK-NOT: omp.parallel
  omp.parallel private(@_QFEi_private_i32 %arg1 -> %arg3 : !fir.ref<i32>) {
    // CHECK: fir.convert
    %15 = fir.convert %c1_i32 : (i32) -> index
    // CHECK: fir.convert
    %16 = fir.convert %c100000_i32 : (i32) -> index
    // CHECK: fir.do_loop
    %18:2 = fir.do_loop %arg4 = %15 to %16 step %c1 iter_args(%arg2 = %arg0) -> (index, i32) {
      // CHECK: fir.store
      fir.store %arg0 to %arg1 : !fir.ref<i32>
      // CHECK-NOT: omp.barrier
      omp.barrier
      fir.result %arg4, %arg2 : index, i32
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
    }
  return
}

// -----

// CHECK-LABEL: func.func @do
func.func @do(%arg5: i32, %arg6: !fir.ref<i32>) {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  // CHECK: %[[C100:.*]] = fir.convert %c100_i32 : (i32) -> index
  %c100_i32 = arith.constant 100 : i32
  // CHECK-NOT: omp.wsloop
  omp.wsloop {
    // CHECK-NOT: omp.loop_nest
    // CHECK: fir.do_loop %[[IVAR:.*]] = %[[C1]] to %[[C100]] step %[[C1]]
    omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c100_i32) inclusive step (%c1_i32) {
      // CHECK: fir.store
      fir.store %arg5 to %arg6 : !fir.ref<i32>
      // CHECK-NOT: omp.yield
      omp.yield
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @do_nested
func.func @do_nested(%arg5: i32, %arg6: !fir.ref<i32>) {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  %c200_i32 = arith.constant 200 : i32
  // CHECK-NOT: omp.wsloop
  omp.wsloop {
    // CHECK: %[[C200:.*]] = fir.convert %c200_i32 : (i32) -> index
    // CHECK-NOT: omp.loop_nest
    // CHECK: fir.do_loop %[[IVAR_1:.*]] = %[[C1]] to %[[C200]] step %[[C1]]
    // CHECK: %[[C100:.*]] = fir.convert %c100_i32 : (i32) -> index
    // CHECK: fir.do_loop %[[IVAR_2:.*]] = %[[C1]] to %[[C100]] step %[[C1]]
    omp.loop_nest (%arg2, %arg3) : i32 = (%c1_i32, %c1_i32) to (%c200_i32, %c100_i32) inclusive step (%c1_i32, %c1_i32) {
      // CHECK: fir.store
      fir.store %arg5 to %arg6 : !fir.ref<i32>
      // CHECK-NOT: omp.yield
      omp.yield
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @single
func.func @single(%arg0: i32, %arg1: !fir.ref<i32>) {
  // CHECK-NOT: omp.single
  omp.single {
    // CHECK: fir.store
    fir.store %arg0 to %arg1 : !fir.ref<i32>
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @target_map(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @target_map(%arg5: i32, %arg6: !fir.ref<i32>) {
  // CHECK-NOT: omp.map.info
  %3 = omp.map.info var_ptr(%arg6 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32>
  // CHECK-NOT: omp.target
  omp.target map_entries(%3 -> %arg0 : !fir.ref<i32>) {
    // CHECK: arith.constant
    %c1_i32 = arith.constant 1 : i32
    // CHECK: fir.store %c1_i32 to %[[ARG_1]]
    fir.store %c1_i32 to %arg0 : !fir.ref<i32>
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @task(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
omp.private {type = private} @_QFEi_private_i32 : i32
func.func @task(%arg5: i32, %arg6: !fir.ref<i32>) {
  // CHECK-NOT: omp.task
  omp.task private(@_QFEi_private_i32 %arg6 -> %arg2 : !fir.ref<i32>) {
    // CHECK: fir.store %[[ARG_0]] to %[[ARG_1]]
    fir.store %arg5 to %arg2 : !fir.ref<i32>
    // CHECK-NOT: omp.flush
    omp.flush
    // CHECK-NOT: omp.taskyield
    omp.taskyield
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @teams
func.func @teams(%arg0: i32, %arg1: !fir.ref<i32>) {
  // CHECK-NOT: omp.teams
  omp.teams {
    // CHECK: fir.store
    fir.store %arg0 to %arg1 : !fir.ref<i32>
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @distribute
func.func @distribute(%arg0: i32, %arg1: i32, %arg2: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  // CHECK-NOT: omp.teams
  omp.teams {
    // CHECK-NOT: omp.distribute
    omp.distribute {
      // CHECK-NOT: omp.loop_nest
      // CHECK: fir.do_loop
      omp.loop_nest (%arg5) : i32 = (%arg0) to (%arg1) inclusive step (%c1_i32) {
        // CHECK: fir.store
        fir.store %arg0 to %arg2 : !fir.ref<i32>
        // CHECK-NOT: omp.yield
        omp.yield
      }
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @threadprivate(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @threadprivate(%arg0: i32, %arg1: !fir.ref<i32>) {
  // CHECK-NOT: omp.threadprivate
  %1 = omp.threadprivate %arg1 : !fir.ref<i32> -> !fir.ref<i32>
  // CHECK: fir.store %[[ARG_0]] to %[[ARG_1]]
  fir.store %arg0 to %1 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL: func.func @taskloop(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @taskloop(%funcArg0: i32, %funcArg1: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c10_i32 = arith.constant 10 : i32
  // CHECK-NOT: omp.taskloop
  omp.taskloop grainsize(%c2_i32: i32) {
    // CHECK-NOT: omp.loop_nest
    // CHECK: fir.do_loop
    omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK: fir.store %[[ARG_0]] to %[[ARG_1]]
      fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
      // CHECK-NOT: omp.yield
      omp.yield
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @target_update_enter_data_map_info(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @target_update_enter_data_map_info(%funcArg0: i32, %funcArg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  // CHECK-NOT: omp.map.bounds
  %1 = omp.map.bounds lower_bound(%c1 : index) upper_bound(%c1 : index) extent(%c1 : index) stride(%c1 : index) start_idx(%c1 : index)
  // CHECK-NOT: omp.map.info
  %13 = omp.map.info var_ptr(%funcArg1 : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) bounds(%1) -> !fir.ref<i32>
  // CHECK-NOT: omp.target_enter_data
  omp.target_enter_data map_entries(%13 : !fir.ref<i32>)
  // CHECK-NOT: omp.target
  omp.target map_entries(%13 -> %arg3 : !fir.ref<i32>) {
    %c1_i32 = arith.constant 1 : i32
    // CHECK: fir.store %c1_i32 to %[[ARG_1]]
    fir.store %c1_i32 to %arg3 : !fir.ref<i32>
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  // CHECK-NOT: omp.map.info
  %18 = omp.map.info var_ptr(%funcArg1 : !fir.ref<i32>, i32) map_clauses(from) capture(ByRef) bounds(%1) -> !fir.ref<i32>
  // CHECK-NOT: omp.target_update
  omp.target_update map_entries(%18 : !fir.ref<i32>)
  // CHECK-NOT: omp.target_exit_data
  omp.target_exit_data map_entries(%18 : !fir.ref<i32>)
  return
}

// -----

// CHECK-LABEL: func.func @target_data(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @target_data(%funcArg0: i32, %funcArg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  // CHECK-NOT: omp.map.bounds
  %3 = omp.map.bounds lower_bound(%c1 : index) upper_bound(%c1 : index) extent(%c1 : index) stride(%c1 : index) start_idx(%c1 : index)
  // CHECK-NOT: omp.map.info
  %4 = omp.map.info var_ptr(%funcArg1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) bounds(%3) -> !fir.ref<i32>
  // CHECK-NOT: omp.target_data
  omp.target_data map_entries(%4 : !fir.ref<i32>) {
    %c1_i32 = arith.constant 1 : i32
    // CHECK: fir.store %c1_i32 to %[[ARG_1]]
    fir.store %c1_i32 to %4 : !fir.ref<i32>
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @sections(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_2:.*]]: !fir.ref<i32>
func.func @sections(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %funcArg2: !fir.ref<i32>) {
  // CHECK-NOT: omp.sections
  omp.sections {
    // CHECK-NOT: omp.section
    omp.section {
      // CHECK: fir.store
      fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
      // CHECK-NOT: omp.terminator
      omp.terminator
    }
    // CHECK-NOT: omp.section
    omp.section {
      // CHECK: fir.store
      fir.store %funcArg0 to %funcArg2 : !fir.ref<i32>
      // CHECK-NOT: omp.terminator
      omp.terminator
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %c0_i32 = arith.constant 0 : i32
  omp.yield(%c0_i32 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = arith.addi %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
// CHECK-LABEL: func.func @reduction_scan(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @reduction_scan(%funcArg0: i32, %funcArg1: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  // CHECK-NOT: omp.wsloop
  omp.wsloop reduction(mod: inscan, @add_reduction_i32 %funcArg1 -> %arg3 : !fir.ref<i32>) {
    // CHECK-NOT: omp.loop_nest
    // CHECK: fir.do_loop
    omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%c8_i32) inclusive step (%c1_i32) {
      // CHECK: fir.declare %[[ARG_1]]
      %1 = fir.declare %arg3 {uniq_name = "a"} : (!fir.ref<i32>) -> !fir.ref<i32>
      // CHECK-NOT: omp.scan
      omp.scan inclusive(%1 : !fir.ref<i32>)
      // CHECK: fir.store
      fir.store %funcArg0 to %1 : !fir.ref<i32>
      // CHECK-NOT: omp.yield
      omp.yield
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @ordered(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @ordered(%funcArg0: i32, %funcArg1: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK-NOT: omp.wsloop
    omp.wsloop ordered(0) {
      // CHECK-NOT: omp.loop_nest
      // CHECK: fir.do_loop
      omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
        // CHECK-NOT: omp.ordered.region
        omp.ordered.region {
          // CHECK: fir.store
          fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
          // CHECK-NOT: omp.terminator
          omp.terminator
        }
        // CHECK-NOT: omp.yield
        omp.yield
      }
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @master(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_2:.*]]: !fir.ref<i32>
func.func @master(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %funcArg2: !fir.ref<i32>) {
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK: fir.store
    fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
    // CHECK-NOT: omp.master
    omp.master {
      // CHECK: fir.store
      fir.store %funcArg0 to %funcArg2 : !fir.ref<i32>
      // CHECK-NOT: omp.terminator
      omp.terminator
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @masked(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_2:.*]]: !fir.ref<i32>
func.func @masked(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %funcArg2: !fir.ref<i32>) {
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK: fir.store
    fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
    // CHECK-NOT: omp.masked
    omp.masked {
      // CHECK: fir.store
      fir.store %funcArg0 to %funcArg2 : !fir.ref<i32>
      // CHECK-NOT: omp.terminator
      omp.terminator
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @critical(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_2:.*]]: !fir.ref<i32>
omp.critical.declare @mylock
func.func @critical(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %funcArg2: !fir.ref<i32>) {
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK: fir.store
    fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
    // CHECK-NOT: omp.critical
    omp.critical(@mylock) {
      // CHECK: fir.store
      fir.store %funcArg0 to %funcArg2 : !fir.ref<i32>
      // CHECK-NOT: omp.terminator
      omp.terminator
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @cancel(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_2:.*]]: !fir.ref<i32>, %[[ARG_3:.*]]: i1
func.func @cancel(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %funcArg2: !fir.ref<i32>, %funcArg3: i1) {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK-NOT: omp.wsloop
    omp.wsloop {
      // CHECK-NOT: omp.loop_nest
      // CHECK: fir.do_loop
      omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
        // CHECK: fir.store
        fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
        // CHECK-NOT: fir.if
        fir.if %funcArg3 {
          // CHECK-NOT: omp.cancel
          omp.cancel cancellation_construct_type(loop)
        }
        // CHECK-NOT: omp.cancellation_point
        omp.cancellation_point cancellation_construct_type(loop)
        // CHECK: fir.store
        fir.store %funcArg0 to %funcArg2 : !fir.ref<i32>
        // CHECK-NOT: omp.yield
        omp.yield
      }
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @atomic(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_2:.*]]: !fir.ref<i32>, %[[ARG_3:.*]]: i32
func.func @atomic(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %funcArg2: !fir.ref<i32>, %funcArg3: i32) {
  %c1_i32 = arith.constant 1 : i32
  %5 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
  // CHECK: %[[VAL_0:.*]] = fir.declare
  %6 = fir.declare %5 {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK-NOT: omp.atomic.write
    // CHECK: fir.store %[[ARG_0]] to %[[ARG_1]]
    omp.atomic.write %funcArg1 = %funcArg0 : !fir.ref<i32>, i32
    // CHECK-NOT: omp.atomic.read
    // CHECK: %[[VAL_1:.*]] = fir.load %[[ARG_1]]
    // CHECK-NEXT: fir.store %[[VAL_1]] to %[[ARG_2]]
    omp.atomic.read %funcArg2 = %funcArg1 : !fir.ref<i32>, !fir.ref<i32>, i32
    // CHECK-NOT: omp.atomic.update
    // CHECK: fir.load %[[VAL_0]]
    // CHECK-NEXT: %[[ADD_VAL:.*]] = arith.addi
    // CHECK-NOT: omp.yield
    // CHECK-NEXT: fir.store %[[ADD_VAL]] to %[[VAL_0]]
    omp.atomic.update %6 : !fir.ref<i32> {
    ^bb0(%arg3: i32):
      %88 = arith.addi %arg3, %c1_i32 : i32
      omp.yield(%88 : i32)
    }
    // CHECK-NOT: omp.atomic.read
    // CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]]
    // CHECK-NEXT: fir.store %[[VAL_2]] to %[[ARG_1]]
    omp.atomic.read %funcArg1 = %6 : !fir.ref<i32>, !fir.ref<i32>, i32
    // CHECK-NOT: omp.atomic.capture
    omp.atomic.capture {
      // CHECK-NOT: omp.atomic.read
      // CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_0]]
      // CHECK-NEXT: fir.store %[[VAL_3]] to %[[ARG_2]]
      omp.atomic.read %funcArg2 = %6 : !fir.ref<i32>, !fir.ref<i32>, i32
      // CHECK-NOT: omp.atomic.update
      // CHECK: fir.load %[[VAL_0]]
      // CHECK-NEXT: %[[ADD_VAL_2:.*]] = arith.addi
      // CHECK-NOT: omp.yield
      // CHECK-NEXT: fir.store %[[ADD_VAL_2]] to %[[VAL_0]]
      omp.atomic.update %6 : !fir.ref<i32> {
      ^bb0(%arg3: i32):
        %88 = arith.addi %arg3, %c1_i32 : i32
        omp.yield(%88 : i32)
      }
    }
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @multi_block(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_3:.*]]: i1
func.func @multi_block(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %6: i1) {
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: omp.parallel
  omp.parallel {
    // CHECK: cf.cond_br %[[ARG_3]], ^[[BB1:.*]], ^[[BB2:.*]]
    cf.cond_br %6, ^bb1, ^bb2
  // CHECK: ^[[BB1]]
  ^bb1:  // pred: ^bb0
    // CHECK: fir.call
    fir.call @_FortranAStopStatement(%c0_i32, %false, %false) fastmath<contract> : (i32, i1, i1) -> ()
    // CHECK-NOT: omp.terminator
    omp.terminator
  // CHECK: ^[[BB2]]
  ^bb2:  // pred: ^bb0
    // CHECK: fir.store
    fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
    // CHECK-NOT: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: func.func @do_multi_block(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>, %[[ARG_3:.*]]: i1
func.func @do_multi_block(%funcArg0: i32, %funcArg1: !fir.ref<i32>, %6: i1) {
  %false = arith.constant false
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  // CHECK-NOT: omp.wsloop
  omp.wsloop {
    // CHECK-NOT: omp.loop_nest
    // CHECK: cf.br ^[[CBB:.*]](
    // CHECK: ^[[CBB]]
    // CHECK: %[[CMP_VAL:.*]] = arith.cmpi
    // CHECK: cf.cond_br %[[CMP_VAL]], ^[[FBB:.*]], ^[[LBB:.*]]
    omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%c100_i32) inclusive step (%c1_i32) {
    // CHECK: ^[[FBB]]
      // CHECK: fir.store
      fir.store %funcArg0 to %funcArg1 : !fir.ref<i32>
      // CHECK: cf.br ^[[BBB:.*]]
      cf.br ^bb1
    // CHECK: ^[[BBB]]
    ^bb1:  // pred: ^bb0
      // CHECK: fir.store
      fir.store %c1_i32 to %funcArg1 : !fir.ref<i32>
      // CHECK: cf.cond_br
      cf.cond_br %6, ^bb2, ^bb3
    // CHECK: ^[[SBB:.*]]
    ^bb2:  // pred: ^bb1
      // CHECK: fir.call
      fir.call @_FortranAStopStatement(%c1_i32, %false, %false) fastmath<contract> : (i32, i1, i1) -> ()
      // CHECK-NOT: omp.yield
      omp.yield
      // CHECK: cf.br ^[[LBB:.*]]
    // CHECK: ^[[OBB:.*]]
      // CHECK: cf.br ^[[LBB]]
    // CHECK: ^[[LBB]]
      // CHECK: arith.subi
      // CHECK: cf.br ^[[CBB]]
    // CHECK: ^[[EBB:.*]]
    ^bb3:  // pred: ^bb1
      // CHECK-NOT: omp.yield
      omp.yield
    }
  }
  return
}
