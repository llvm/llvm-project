// RUN: fir-opt --split-input-file --verify-diagnostics --omp-simd-only %s | FileCheck %s

// Check that simd operations are not removed and rewritten, but all the other OpenMP ops are.
// Tests the logic in flang/lib/Optimizer/OpenMP/SimdOnly.cpp

// CHECK: omp.private
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

// CHECK-NOT: omp.private
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
      fir.result %arg4, %arg2 : index, i32
    }
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

// CHECK-LABEL: func.func @distribute_simd
func.func @distribute_simd(%arg0: i32, %arg1: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %c100000_i32 = arith.constant 100000 : i32
  // CHECK-NOT: omp.distribute
  omp.distribute {
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
  // CHECK-NOT: {omp.composite}
  } {omp.composite}
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

// CHECK-LABEL: func.func @map_info(
// CHECK-SAME: %[[ARG_0:.*]]: i32, %[[ARG_1:.*]]: !fir.ref<i32>
func.func @map_info(%funcArg0: i32, %funcArg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  // CHECK-NOT: omp.map.bounds
  %1 = omp.map.bounds lower_bound(%c1 : index) upper_bound(%c1 : index) extent(%c1 : index) stride(%c1 : index) start_idx(%c1 : index)
  // CHECK-NOT: omp.map.info
  %13 = omp.map.info var_ptr(%funcArg1 : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) bounds(%1) -> !fir.ref<i32>
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
  return
}
