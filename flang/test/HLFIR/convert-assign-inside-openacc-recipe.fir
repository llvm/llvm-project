// Check that hlfir.assign codegen is able to insert fir.alloca's inside
// the regions of the OpenACC recipe.
// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

acc.reduction.recipe @reduction_add_box_heap_Uxi32 : !fir.box<!fir.heap<!fir.array<?xi32>>> reduction_operator <add> init {
^bb0(%arg0: !fir.box<!fir.heap<!fir.array<?xi32>>>):
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %1 = fir.shape %0#1 : (index) -> !fir.shape<1>
  %2 = fir.allocmem !fir.array<?xi32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
  %3:2 = hlfir.declare %2(%1) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
  hlfir.assign %c0_i32 to %3#0 : i32, !fir.box<!fir.array<?xi32>>
  acc.yield %3#0 : !fir.box<!fir.array<?xi32>>
} combiner {
^bb0(%arg0: !fir.box<!fir.heap<!fir.array<?xi32>>>, %arg1: !fir.box<!fir.heap<!fir.array<?xi32>>>, %arg2: index, %arg3: index, %arg4: index):
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = arith.subi %arg3, %arg2 : index
  %1 = arith.addi %0, %c1 : index
  %2 = arith.divsi %1, %arg4 : index
  %3 = arith.cmpi sgt, %2, %c0 : index
  %4 = arith.select %3, %2, %c0 : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = hlfir.designate %arg0 (%arg2:%arg3:%arg4)  shape %5 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  %7 = hlfir.designate %arg1 (%arg2:%arg3:%arg4)  shape %5 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  %8 = fir.allocmem !fir.array<?xi32>, %4 {bindc_name = ".tmp.array", uniq_name = ""}
  %9:2 = hlfir.declare %8(%5) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
  %true = arith.constant true
  %c1_0 = arith.constant 1 : index
  fir.do_loop %arg5 = %c1_0 to %4 step %c1_0 unordered {
    %13 = hlfir.designate %6 (%arg5)  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
    %14 = hlfir.designate %7 (%arg5)  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
    %15 = fir.load %13 : !fir.ref<i32>
    %16 = fir.load %14 : !fir.ref<i32>
    %17 = arith.addi %15, %16 : i32
    %18 = hlfir.designate %9#0 (%arg5)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    hlfir.assign %17 to %18 temporary_lhs : i32, !fir.ref<i32>
  }
  %10 = fir.undefined tuple<!fir.box<!fir.array<?xi32>>, i1>
  %11 = fir.insert_value %10, %true, [1 : index] : (tuple<!fir.box<!fir.array<?xi32>>, i1>, i1) -> tuple<!fir.box<!fir.array<?xi32>>, i1>
  %12 = fir.insert_value %11, %9#0, [0 : index] : (tuple<!fir.box<!fir.array<?xi32>>, i1>, !fir.box<!fir.array<?xi32>>) -> tuple<!fir.box<!fir.array<?xi32>>, i1>
  hlfir.assign %9#0 to %arg0 : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  acc.yield %arg0 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}
// CHECK-LABEL:   acc.reduction.recipe @reduction_add_box_heap_Uxi32 : !fir.box<!fir.heap<!fir.array<?xi32>>> reduction_operator <add> init {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.heap<!fir.array<?xi32>>>):
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK-LABEL:   } combiner {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.heap<!fir.array<?xi32>>>, %[[VAL_1:.*]]: !fir.box<!fir.heap<!fir.array<?xi32>>>, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index):
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
