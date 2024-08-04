// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// checks:
// nowait on final omp.single
func.func @wsfunc(%arg0: !fir.ref<!fir.array<42xi32>>) {
  omp.parallel {
    omp.workshare {
      %c42 = arith.constant 42 : index
      %c1_i32 = arith.constant 1 : i32
      %0 = fir.shape %c42 : (index) -> !fir.shape<1>
      %1:2 = hlfir.declare %arg0(%0) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
      %2 = fir.allocmem !fir.array<42xi32> {bindc_name = ".tmp.array", uniq_name = ""}
      %3:2 = hlfir.declare %2(%0) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
      %true = arith.constant true
      %c1 = arith.constant 1 : index
      omp.workshare.loop_wrapper {
        omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
          %7 = hlfir.designate %1#0 (%arg1)  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
          %8 = fir.load %7 : !fir.ref<i32>
          %9 = arith.subi %8, %c1_i32 : i32
          %10 = hlfir.designate %3#0 (%arg1)  : (!fir.heap<!fir.array<42xi32>>, index) -> !fir.ref<i32>
          hlfir.assign %9 to %10 temporary_lhs : i32, !fir.ref<i32>
          omp.yield
        }
        omp.terminator
      }
      %4 = fir.undefined tuple<!fir.heap<!fir.array<42xi32>>, i1>
      %5 = fir.insert_value %4, %true, [1 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
      %6 = fir.insert_value %5, %3#0, [0 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, !fir.heap<!fir.array<42xi32>>) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
      hlfir.assign %3#0 to %1#0 : !fir.heap<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
      fir.freemem %3#0 : !fir.heap<!fir.array<42xi32>>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

// checks:
// fir.alloca hoisted out and copyprivate'd
func.func @wsfunc(%arg0: !fir.ref<!fir.array<42xi32>>) {
  omp.workshare {
    %c1_i32 = arith.constant 1 : i32
    %alloc = fir.alloca i32
    fir.store %c1_i32 to %alloc : !fir.ref<i32>
    %c42 = arith.constant 42 : index
    %0 = fir.shape %c42 : (index) -> !fir.shape<1>
    %1:2 = hlfir.declare %arg0(%0) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
    %2 = fir.allocmem !fir.array<42xi32> {bindc_name = ".tmp.array", uniq_name = ""}
    %3:2 = hlfir.declare %2(%0) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    omp.workshare.loop_wrapper {
      omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
        %7 = hlfir.designate %1#0 (%arg1)  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
        %8 = fir.load %7 : !fir.ref<i32>
        %ld = fir.load %alloc : !fir.ref<i32>
        %n8 = arith.subi %8, %ld : i32
        %9 = arith.subi %n8, %c1_i32 : i32
        %10 = hlfir.designate %3#0 (%arg1)  : (!fir.heap<!fir.array<42xi32>>, index) -> !fir.ref<i32>
        hlfir.assign %9 to %10 temporary_lhs : i32, !fir.ref<i32>
        omp.yield
      }
      omp.terminator
    }
    %4 = fir.undefined tuple<!fir.heap<!fir.array<42xi32>>, i1>
    %5 = fir.insert_value %4, %true, [1 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
    %6 = fir.insert_value %5, %3#0, [0 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, !fir.heap<!fir.array<42xi32>>) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
    "test.test1"(%alloc) : (!fir.ref<i32>) -> ()
    hlfir.assign %3#0 to %1#0 : !fir.heap<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
    fir.freemem %3#0 : !fir.heap<!fir.array<42xi32>>
    omp.terminator
  }
  return
}

// CHECK-LABEL:   func.func private @_workshare_copy_heap_42xi32(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !fir.ref<!fir.heap<!fir.array<42xi32>>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !fir.ref<!fir.heap<!fir.array<42xi32>>>) {
// CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @wsfunc(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           omp.parallel {
// CHECK:             %[[VAL_1:.*]] = fir.alloca !fir.heap<!fir.array<42xi32>>
// CHECK:             omp.single copyprivate(%[[VAL_1]] -> @_workshare_copy_heap_42xi32 : !fir.ref<!fir.heap<!fir.array<42xi32>>>) {
// CHECK:               %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:               %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:               %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:               %[[VAL_5:.*]] = fir.allocmem !fir.array<42xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:               fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:               %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_3]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             %[[VAL_7:.*]] = arith.constant 42 : index
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_9:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_9]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:             %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_9]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest (%[[VAL_14:.*]]) : index = (%[[VAL_13]]) to (%[[VAL_7]]) inclusive step (%[[VAL_13]]) {
// CHECK:                 %[[VAL_15:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_14]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:                 %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
// CHECK:                 %[[VAL_17:.*]] = arith.subi %[[VAL_16]], %[[VAL_8]] : i32
// CHECK:                 %[[VAL_18:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_14]])  : (!fir.heap<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:                 hlfir.assign %[[VAL_17]] to %[[VAL_18]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.single nowait {
// CHECK:               hlfir.assign %[[VAL_12]]#0 to %[[VAL_10]]#0 : !fir.heap<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
// CHECK:               fir.freemem %[[VAL_12]]#0 : !fir.heap<!fir.array<42xi32>>
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.barrier
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_workshare_copy_heap_42xi32(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !fir.ref<!fir.heap<!fir.array<42xi32>>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !fir.ref<!fir.heap<!fir.array<42xi32>>>) {
// CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @_workshare_copy_i32(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @wsfunc(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca i32
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.heap<!fir.array<42xi32>>
// CHECK:           omp.single copyprivate(%[[VAL_1]] -> @_workshare_copy_i32 : !fir.ref<i32>, %[[VAL_2]] -> @_workshare_copy_heap_42xi32 : !fir.ref<!fir.heap<!fir.array<42xi32>>>) {
// CHECK:             %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:             fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<i32>
// CHECK:             %[[VAL_4:.*]] = arith.constant 42 : index
// CHECK:             %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:             %[[VAL_7:.*]] = fir.allocmem !fir.array<42xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:             fir.store %[[VAL_7]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:             %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_5]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_11]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_11]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:           omp.wsloop {
// CHECK:             omp.loop_nest (%[[VAL_16:.*]]) : index = (%[[VAL_15]]) to (%[[VAL_10]]) inclusive step (%[[VAL_15]]) {
// CHECK:               %[[VAL_17:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_16]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<i32>
// CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:               %[[VAL_20:.*]] = arith.subi %[[VAL_18]], %[[VAL_19]] : i32
// CHECK:               %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_9]] : i32
// CHECK:               %[[VAL_22:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_16]])  : (!fir.heap<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_21]] to %[[VAL_22]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:               omp.yield
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           omp.single nowait {
// CHECK:             "test.test1"(%[[VAL_1]]) : (!fir.ref<i32>) -> ()
// CHECK:             hlfir.assign %[[VAL_14]]#0 to %[[VAL_12]]#0 : !fir.heap<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
// CHECK:             fir.freemem %[[VAL_14]]#0 : !fir.heap<!fir.array<42xi32>>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           omp.barrier
// CHECK:           return
// CHECK:         }

