// RUN: fir-opt --omp-loop-index-mem2reg %s | FileCheck %s

func.func private @foo(%arg0 : !fir.ref<i32>) -> i32

// CHECK-LABEL: @wsloop_remove_alloca
func.func @wsloop_remove_alloca() {
  // CHECK: %[[RESULT:.*]] = fir.alloca i32
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.wsloop for (%[[INDEX:.*]]) : i32
    %1 = fir.alloca i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.wsloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK-NOT: fir.alloca
      // CHECK: fir.store %[[INDEX]] to %[[RESULT]]
      // CHECK: omp.yield
      fir.store %arg0 to %1 : !fir.ref<i32>
      %2 = fir.load %1 : !fir.ref<i32>
      fir.store %2 to %0 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @simdloop_remove_alloca
func.func @simdloop_remove_alloca() {
  // CHECK: %[[RESULT:.*]] = fir.alloca i32
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.simdloop for (%[[INDEX:.*]]) : i32
    %1 = fir.alloca i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.simdloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK-NOT: fir.alloca
      // CHECK: fir.store %[[INDEX]] to %[[RESULT]]
      // CHECK: omp.yield
      fir.store %arg0 to %1 : !fir.ref<i32>
      %2 = fir.load %1 : !fir.ref<i32>
      fir.store %2 to %0 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @wsloop_push_alloca
func.func @wsloop_push_alloca() {
  // CHECK: %[[RESULT:.*]] = fir.alloca i32
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.wsloop for (%[[INDEX:.*]]) : i32
    %1 = fir.alloca i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.wsloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK: %[[ALLOCA:.*]] = fir.alloca i32
      // CHECK: fir.store %[[INDEX]] to %[[ALLOCA]]
      // CHECK: %[[RETURN:.*]] = func.call @foo(%[[ALLOCA]])
      // CHECK: fir.store %[[RETURN]] to %[[RESULT]]
      // CHECK: omp.yield
      fir.store %arg0 to %1 : !fir.ref<i32>
      %2 = func.call @foo(%1) : (!fir.ref<i32>) -> i32
      fir.store %2 to %0 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @simdloop_push_alloca
func.func @simdloop_push_alloca() {
  // CHECK: %[[RESULT:.*]] = fir.alloca i32
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.simdloop for (%[[INDEX:.*]]) : i32
    %1 = fir.alloca i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.simdloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK: %[[ALLOCA:.*]] = fir.alloca i32
      // CHECK: fir.store %[[INDEX]] to %[[ALLOCA]]
      // CHECK: %[[RETURN:.*]] = func.call @foo(%[[ALLOCA]])
      // CHECK: fir.store %[[RETURN]] to %[[RESULT]]
      // CHECK: omp.yield
      fir.store %arg0 to %1 : !fir.ref<i32>
      %2 = func.call @foo(%1) : (!fir.ref<i32>) -> i32
      fir.store %2 to %0 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @hlfir_wsloop_remove_alloca
func.func @hlfir_wsloop_remove_alloca() {
  // CHECK: %[[RESULT_ALLOCA:.*]] = fir.alloca i32
  // CHECK: %[[RESULT:.*]]:2 = hlfir.declare %[[RESULT_ALLOCA]]
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  %1:2 = hlfir.declare %0 {uniq_name = "result"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-NOT: hlfir.declare
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.wsloop for (%[[INDEX:.*]]) : i32
    %2 = fir.alloca i32
    %3:2 = hlfir.declare %2 {uniq_name = "index"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.wsloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK-NOT: fir.alloca
      // CHECK-NOT: hlfir.declare
      // CHECK: hlfir.assign %[[INDEX]] to %[[RESULT]]#0
      // CHECK: omp.yield
      fir.store %arg0 to %3#1 : !fir.ref<i32>
      %4 = fir.load %3#0 : !fir.ref<i32>
      hlfir.assign %4 to %1#0 : i32, !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @hlfir_simdloop_remove_alloca
func.func @hlfir_simdloop_remove_alloca() {
  // CHECK: %[[RESULT_ALLOCA:.*]] = fir.alloca i32
  // CHECK: %[[RESULT:.*]]:2 = hlfir.declare %[[RESULT_ALLOCA]]
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  %1:2 = hlfir.declare %0 {uniq_name = "result"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-NOT: hlfir.declare
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.simdloop for (%[[INDEX:.*]]) : i32
    %2 = fir.alloca i32
    %3:2 = hlfir.declare %2 {uniq_name = "index"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.simdloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK-NOT: fir.alloca
      // CHECK-NOT: hlfir.declare
      // CHECK: hlfir.assign %[[INDEX]] to %[[RESULT]]#0
      // CHECK: omp.yield
      fir.store %arg0 to %3#1 : !fir.ref<i32>
      %4 = fir.load %3#0 : !fir.ref<i32>
      hlfir.assign %4 to %1#0 : i32, !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @hlfir_wsloop_push_alloca
func.func @hlfir_wsloop_push_alloca() {
  // CHECK: %[[RESULT_ALLOCA:.*]] = fir.alloca i32
  // CHECK: %[[RESULT:.*]]:2 = hlfir.declare %[[RESULT_ALLOCA]]
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  %1:2 = hlfir.declare %0 {uniq_name = "result"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-NOT: hlfir.declare
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.wsloop for (%[[INDEX:.*]]) : i32
    %2 = fir.alloca i32
    %3:2 = hlfir.declare %2 {uniq_name = "index"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.wsloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK: %[[INDEX_ALLOCA:.*]] = fir.alloca i32
      // CHECK: %[[INDEX_DECL:.*]]:2 = hlfir.declare %[[INDEX_ALLOCA]]
      // CHECK: fir.store %[[INDEX]] to %[[INDEX_DECL]]#1
      // CHECK: %[[RETURN:.*]] = fir.call @foo(%[[INDEX_DECL]]#1)
      // CHECK: hlfir.assign %[[RETURN]] to %[[RESULT]]#0
      // CHECK: omp.yield
      fir.store %arg0 to %3#1 : !fir.ref<i32>
      %4 = fir.call @foo(%3#1) : (!fir.ref<i32>) -> i32
      hlfir.assign %4 to %1#0 : i32, !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @hlfir_simdloop_push_alloca
func.func @hlfir_simdloop_push_alloca() {
  // CHECK: %[[RESULT_ALLOCA:.*]] = fir.alloca i32
  // CHECK: %[[RESULT:.*]]:2 = hlfir.declare %[[RESULT_ALLOCA]]
  // CHECK: omp.parallel
  %0 = fir.alloca i32
  %1:2 = hlfir.declare %0 {uniq_name = "result"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  omp.parallel {
    // CHECK-NOT: fir.alloca
    // CHECK-NOT: hlfir.declare
    // CHECK-DAG: arith.constant 1
    // CHECK-DAG: arith.constant 10
    // CHECK-NEXT: omp.simdloop for (%[[INDEX:.*]]) : i32
    %2 = fir.alloca i32
    %3:2 = hlfir.declare %2 {uniq_name = "index"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    omp.simdloop for (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32) {
      // CHECK: %[[INDEX_ALLOCA:.*]] = fir.alloca i32
      // CHECK: %[[INDEX_DECL:.*]]:2 = hlfir.declare %[[INDEX_ALLOCA]]
      // CHECK: fir.store %[[INDEX]] to %[[INDEX_DECL]]#1
      // CHECK: %[[RETURN:.*]] = fir.call @foo(%[[INDEX_DECL]]#1)
      // CHECK: hlfir.assign %[[RETURN]] to %[[RESULT]]#0
      // CHECK: omp.yield
      fir.store %arg0 to %3#1 : !fir.ref<i32>
      %4 = fir.call @foo(%3#1) : (!fir.ref<i32>) -> i32
      hlfir.assign %4 to %1#0 : i32, !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}
