// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Verify that the shape match is not required for optimizing
// elemental assignment, when lhs not an allocatable.
// The shapes of lhs and rhs must conform in a legal program.
//
// Example:
// subroutine test(a,b)
//   integer :: a(:), b(:)
//   a = b + 1
// end subroutine test

func.func @_QPtest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtestEa"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtestEb"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3:3 = fir.box_dims %2#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
  %5 = hlfir.elemental %4 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg2: index):
    %6 = hlfir.designate %2#0 (%arg2)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %7 = fir.load %6 : !fir.ref<i32>
    %8 = arith.addi %7, %c1_i32 : i32
    hlfir.yield_element %8 : i32
  }
  hlfir.assign %5 to %1#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %5 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL:   func.func @_QPtest(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                       %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_5]] {uniq_name = "_QFtestEa"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_5]] {uniq_name = "_QFtestEb"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_7]]#0, %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_2]] to %[[VAL_8]]#1 step %[[VAL_2]] unordered {
// CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_4]] : i32
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_12]] to %[[VAL_13]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
