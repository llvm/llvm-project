// Tests mapping of a basic `do concurrent` loop to `!$omp parallel do`.

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

// CHECK-LABEL: func.func @do_concurrent_basic
func.func @do_concurrent_basic() attributes {fir.bindc_name = "do_concurrent_basic"} {
    // CHECK: %[[ARR:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)

    %0 = fir.alloca i32 {bindc_name = "i"}
    %1:2 = hlfir.declare %0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xi32>>
    %c10 = arith.constant 10 : index
    %3 = fir.shape %c10 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %2(%3) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
    %c1_i32 = arith.constant 1 : i32
    %7 = fir.convert %c1_i32 : (i32) -> index
    %c10_i32 = arith.constant 10 : i32
    %8 = fir.convert %c10_i32 : (i32) -> index
    %c1 = arith.constant 1 : index

    // CHECK-NOT: fir.do_loop

    // CHECK: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK: %[[LB:.*]] = fir.convert %[[C1]] : (i32) -> index
    // CHECK: %[[C10:.*]] = arith.constant 10 : i32
    // CHECK: %[[UB:.*]] = fir.convert %[[C10]] : (i32) -> index
    // CHECK: %[[STEP:.*]] = arith.constant 1 : index

    // CHECK: omp.parallel {

    // CHECK-NEXT: %[[ITER_VAR:.*]] = fir.alloca i32 {bindc_name = "i"}
    // CHECK-NEXT: %[[BINDING:.*]]:2 = hlfir.declare %[[ITER_VAR]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    // CHECK: omp.wsloop {
    // CHECK-NEXT: omp.loop_nest (%[[ARG0:.*]]) : index = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
    // CHECK-NEXT: %[[IV_IDX:.*]] = fir.convert %[[ARG0]] : (index) -> i32
    // CHECK-NEXT: fir.store %[[IV_IDX]] to %[[BINDING]]#1 : !fir.ref<i32>
    // CHECK-NEXT: %[[IV_VAL1:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    // CHECK-NEXT: %[[IV_VAL2:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    // CHECK-NEXT: %[[IV_VAL_I64:.*]] = fir.convert %[[IV_VAL2]] : (i32) -> i64
    // CHECK-NEXT: %[[ARR_ACCESS:.*]] = hlfir.designate %[[ARR]]#0 (%[[IV_VAL_I64]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
    // CHECK-NEXT: hlfir.assign %[[IV_VAL1]] to %[[ARR_ACCESS]] : i32, !fir.ref<i32>
    // CHECK-NEXT: omp.yield
    // CHECK-NEXT: }
    // CHECK-NEXT: }

    // CHECK-NEXT: omp.terminator
    // CHECK-NEXT: }
    fir.do_loop %arg0 = %7 to %8 step %c1 unordered {
      %13 = fir.convert %arg0 : (index) -> i32
      fir.store %13 to %1#1 : !fir.ref<i32>
      %14 = fir.load %1#0 : !fir.ref<i32>
      %15 = fir.load %1#0 : !fir.ref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = hlfir.designate %4#0 (%16)  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
      hlfir.assign %14 to %17 : i32, !fir.ref<i32>
    }

    // CHECK-NOT: fir.do_loop

    return
  }
