// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' 2>&1 | FileCheck %s

// Exercise that a box reference does not alias with the address loaded from the box
// module mm
//   real, allocatable :: arr(:)
// contains
// subroutine sub
//   arr(1) = ubound(arr, 1)
// end subroutine
// end module

// CHECK: box#0 <-> boxaddr#0: NoAlias

fir.global @_QMmmEarr : !fir.box<!fir.heap<!fir.array<?xf32>>> {
  %c0 = arith.constant 0 : index
  %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
}
func.func @_QMmmPsub() {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMmmEarr) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmmEarr", test.ptr = "box"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  %4 = fir.convert %3#1 : (index) -> i64
  %5 = fir.convert %3#0 : (index) -> i64
  %6 = arith.addi %4, %5 : i64
  %7 = arith.subi %6, %c1_i64 : i64
  %8 = fir.convert %7 : (i64) -> i32
  %9 = fir.convert %8 : (i32) -> f32
  %10 = fir.box_addr %2 {test.ptr = "boxaddr"} : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %11 = fir.shape_shift %3#0, %3#1 : (index, index) -> !fir.shapeshift<1>
  %12 = fir.array_coor %10(%11) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  fir.store %9 to %12 : !fir.ref<f32>
  return
}
