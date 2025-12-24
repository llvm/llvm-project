// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// Tests checking that Flang's LICM works correctly:
//   * Descriptor loads may be hoisted unless they can be modified inside
//     the loop (e.g. global descriptors may be modifed by calls).
//   * Reads of scalar non-optional variables may be hoisted unless
//     they are ALLOCATABLE or POINTER (which means they can be unallocated
//     or disassociated in loops that have zero iterations).
//   * TODO: any invariant loads may be hoisted in loops having non-zero
//     iterations.

// subroutine test_dummy_scalar(r,x,n)
//   integer :: r(*), x, n
//   do i=1,n
//      r(i) = x
//      call external_sub
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_scalar(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalarEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_scalarEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_scalarEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_scalarEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_dummy_scalarEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<i32>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_1]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             fir.call @_QPexternal_sub() : () -> ()
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_scalar(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "x"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalarEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_scalarEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_scalarEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_scalarEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_dummy_scalarEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<i32>
    %14 = fir.convert %13 : (i32) -> i64
    %15 = fir.array_coor %6(%5) %14 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %12 to %15 : !fir.ref<i32>
    fir.call @_QPexternal_sub() : () -> ()
    %16 = fir.load %2 : !fir.ref<i32>
    %17 = arith.addi %16, %10 overflow<nsw> : i32
    fir.result %17 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_scalar_allocatable(r,x,n)
//   integer :: r(*), n
//   ! x may be unallocated:
//   integer, allocatable :: x
//   do i=1,n
//      r(i) = x
//      call external_sub
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_scalar_allocatable(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_allocatableEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_scalar_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_scalar_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_scalar_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_dummy_scalar_allocatableEx"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.heap<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             fir.call @_QPexternal_sub() : () -> ()
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_scalar_allocatable(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "x"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_allocatableEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_scalar_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_scalar_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_scalar_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_dummy_scalar_allocatableEx"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %13 = fir.box_addr %12 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
    %14 = fir.load %13 : !fir.heap<i32>
    %15 = fir.load %2 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.array_coor %6(%5) %16 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %14 to %17 : !fir.ref<i32>
    fir.call @_QPexternal_sub() : () -> ()
    %18 = fir.load %2 : !fir.ref<i32>
    %19 = arith.addi %18, %10 overflow<nsw> : i32
    fir.result %19 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_scalar_pointer(r,x,n)
//   integer :: r(*), n
//   ! x may be disassociated:
//   integer, pointer :: x
//   do i=1,n
//      r(i) = x
//      call external_sub
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_scalar_pointer(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_pointerEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_scalar_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_scalar_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_scalar_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_dummy_scalar_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.ptr<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             fir.call @_QPexternal_sub() : () -> ()
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_scalar_pointer(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "x"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_pointerEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_scalar_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_scalar_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_scalar_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_dummy_scalar_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<!fir.box<!fir.ptr<i32>>>
    %13 = fir.box_addr %12 : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
    %14 = fir.load %13 : !fir.ptr<i32>
    %15 = fir.load %2 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.array_coor %6(%5) %16 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %14 to %17 : !fir.ref<i32>
    fir.call @_QPexternal_sub() : () -> ()
    %18 = fir.load %2 : !fir.ref<i32>
    %19 = arith.addi %18, %10 overflow<nsw> : i32
    fir.result %19 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_scalar(r,n)
//   use data, only : glob_var
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_var
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_scalar(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_var) : !fir.ref<f32>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {uniq_name = "_QMdataEglob_var"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalarEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_scalarEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_scalarEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_scalarEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<f32>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_1]] : (f32) -> i32
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_2]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_scalar(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_var) : !fir.ref<f32>
  %2 = fir.declare %1 {uniq_name = "_QMdataEglob_var"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalarEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_scalarEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_scalarEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_scalarEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<f32>
    %14 = fir.convert %13 : (f32) -> i32
    %15 = fir.load %4 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.array_coor %8(%7) %16 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %14 to %17 : !fir.ref<i32>
    %18 = fir.load %4 : !fir.ref<i32>
    %19 = arith.addi %18, %11 overflow<nsw> : i32
    fir.result %19 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_scalar_allocatable(r,n)
//   use data, only : glob_alloc
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_alloc
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_scalar_allocatable(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_alloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdataEglob_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_allocatableEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_scalar_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_scalar_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_scalar_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.heap<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (f32) -> i32
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_scalar_allocatable(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_alloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdataEglob_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_allocatableEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_scalar_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_scalar_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_scalar_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<f32>>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
    %15 = fir.load %14 : !fir.heap<f32>
    %16 = fir.convert %15 : (f32) -> i32
    %17 = fir.load %4 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> i64
    %19 = fir.array_coor %8(%7) %18 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %16 to %19 : !fir.ref<i32>
    %20 = fir.load %4 : !fir.ref<i32>
    %21 = arith.addi %20, %11 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_scalar_pointer(r,n)
//   use data, only : glob_ptr
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_ptr
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_scalar_pointer(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_ptr) : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdataEglob_ptr"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_pointerEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_scalar_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_scalar_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_scalar_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.ptr<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (f32) -> i32
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_scalar_pointer(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_ptr) : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdataEglob_ptr"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_pointerEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_scalar_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_scalar_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_scalar_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %15 = fir.load %14 : !fir.ptr<f32>
    %16 = fir.convert %15 : (f32) -> i32
    %17 = fir.load %4 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> i64
    %19 = fir.array_coor %8(%7) %18 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %16 to %19 : !fir.ref<i32>
    %20 = fir.load %4 : !fir.ref<i32>
    %21 = arith.addi %20, %11 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_scalar_negative(r,n)
//   use data, only : glob_var
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_var
//      call external_sub
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_scalar_negative(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_var) : !fir.ref<f32>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {uniq_name = "_QMdataEglob_var"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_negativeEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_scalar_negativeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_scalar_negativeEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_scalar_negativeEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_1]] : (f32) -> i32
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_2]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             fir.call @_QPexternal_sub() : () -> ()
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_scalar_negative(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_var) : !fir.ref<f32>
  %2 = fir.declare %1 {uniq_name = "_QMdataEglob_var"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_negativeEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_scalar_negativeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_scalar_negativeEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_scalar_negativeEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<f32>
    %14 = fir.convert %13 : (f32) -> i32
    %15 = fir.load %4 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.array_coor %8(%7) %16 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %14 to %17 : !fir.ref<i32>
    fir.call @_QPexternal_sub() : () -> ()
    %18 = fir.load %4 : !fir.ref<i32>
    %19 = arith.addi %18, %11 overflow<nsw> : i32
    fir.result %19 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_scalar_allocatable_negative(r,n)
//   use data, only : glob_alloc
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_alloc
//      call external_sub
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_scalar_allocatable_negative(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_alloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdataEglob_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_allocatable_negativeEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_scalar_allocatable_negativeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_scalar_allocatable_negativeEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_scalar_allocatable_negativeEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.heap<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (f32) -> i32
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             fir.call @_QPexternal_sub() : () -> ()
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_scalar_allocatable_negative(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_alloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdataEglob_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_allocatable_negativeEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_scalar_allocatable_negativeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_scalar_allocatable_negativeEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_scalar_allocatable_negativeEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<f32>>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
    %15 = fir.load %14 : !fir.heap<f32>
    %16 = fir.convert %15 : (f32) -> i32
    %17 = fir.load %4 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> i64
    %19 = fir.array_coor %8(%7) %18 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %16 to %19 : !fir.ref<i32>
    fir.call @_QPexternal_sub() : () -> ()
    %20 = fir.load %4 : !fir.ref<i32>
    %21 = arith.addi %20, %11 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_scalar_pointer_negative(r,n)
//   use data, only : glob_ptr
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_ptr
//      call external_sub
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_scalar_pointer_negative(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_ptr) : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdataEglob_ptr"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_pointer_negativeEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_scalar_pointer_negativeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_scalar_pointer_negativeEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_scalar_pointer_negativeEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.ptr<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (f32) -> i32
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             fir.call @_QPexternal_sub() : () -> ()
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_scalar_pointer_negative(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_ptr) : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdataEglob_ptr"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_scalar_pointer_negativeEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_scalar_pointer_negativeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_scalar_pointer_negativeEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_scalar_pointer_negativeEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<f32>>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    %15 = fir.load %14 : !fir.ptr<f32>
    %16 = fir.convert %15 : (f32) -> i32
    %17 = fir.load %4 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> i64
    %19 = fir.array_coor %8(%7) %18 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %16 to %19 : !fir.ref<i32>
    fir.call @_QPexternal_sub() : () -> ()
    %20 = fir.load %4 : !fir.ref<i32>
    %21 = arith.addi %20, %11 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_array(r,x,n)
//   integer :: r(*), x(*), n
//   do i=1,n
//      r(i) = x(1)
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_array(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_arrayEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_arrayEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_arrayEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_arrayEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_dummy_arrayEx"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONSTANT_0]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_1]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_array(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "x"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_arrayEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_arrayEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_arrayEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_arrayEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1(%5) dummy_scope %0 arg 2 {uniq_name = "_QFtest_dummy_arrayEx"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.array_coor %7(%5) %c1 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %13 = fir.load %12 : !fir.ref<i32>
    %14 = fir.load %2 : !fir.ref<i32>
    %15 = fir.convert %14 : (i32) -> i64
    %16 = fir.array_coor %6(%5) %15 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %13 to %16 : !fir.ref<i32>
    %17 = fir.load %2 : !fir.ref<i32>
    %18 = arith.addi %17, %10 overflow<nsw> : i32
    fir.result %18 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_array_allocatable(r,x,n)
//   integer :: r(*), n
//   integer, allocatable :: x(:)
//   do i=1,n
//      r(i) = x(1)
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_array_allocatable(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_array_allocatableEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_array_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_array_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_array_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_dummy_array_allocatableEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_1]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONVERT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK:             %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:             %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_0]]#0, %[[BOX_DIMS_0]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[BOX_ADDR_0]](%[[SHAPE_SHIFT_0]]) %[[CONSTANT_1]] : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_2]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_array_allocatable(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_array_allocatableEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_array_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_array_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_array_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_dummy_array_allocatableEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %13 = fir.box_addr %12 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %14:3 = fir.box_dims %12, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %15 = fir.shape_shift %14#0, %14#1 : (index, index) -> !fir.shapeshift<1>
    %16 = fir.array_coor %13(%15) %c1 : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %18 = fir.load %2 : !fir.ref<i32>
    %19 = fir.convert %18 : (i32) -> i64
    %20 = fir.array_coor %6(%5) %19 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %17 to %20 : !fir.ref<i32>
    %21 = fir.load %2 : !fir.ref<i32>
    %22 = arith.addi %21, %10 overflow<nsw> : i32
    fir.result %22 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_array_pointer(r,x,n)
//   integer :: r(*), n
//   integer, pointer :: x(:)
//   do i=1,n
//      r(i) = x(1)
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_array_pointer(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_array_pointerEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_array_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_array_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_array_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_dummy_array_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_1]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONVERT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_0]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:             %[[SHIFT_0:.*]] = fir.shift %[[BOX_DIMS_0]]#0 : (index) -> !fir.shift<1>
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[LOAD_1]](%[[SHIFT_0]]) %[[CONSTANT_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, index) -> !fir.ref<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_2]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_array_pointer(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "x"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_array_pointerEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_array_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_array_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_array_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_dummy_array_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    %13:3 = fir.box_dims %12, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %14 = fir.shift %13#0 : (index) -> !fir.shift<1>
    %15 = fir.array_coor %12(%14) %c1 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, index) -> !fir.ref<i32>
    %16 = fir.load %15 : !fir.ref<i32>
    %17 = fir.load %2 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> i64
    %19 = fir.array_coor %6(%5) %18 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %16 to %19 : !fir.ref<i32>
    %20 = fir.load %2 : !fir.ref<i32>
    %21 = arith.addi %20, %10 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_array(r,n)
//   use data, only : glob_arr_var
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_arr_var(1)
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_array(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_arr_var) : !fir.ref<!fir.array<10xf32>>
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]](%[[SHAPE_0]]) {uniq_name = "_QMdataEglob_arr_var"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_arrayEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_arrayEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_arrayEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_1]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_arrayEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_0]](%[[SHAPE_0]]) %[[CONSTANT_0]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_1]] : (f32) -> i32
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_2]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_1]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_array(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_arr_var) : !fir.ref<!fir.array<10xf32>>
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3 = fir.declare %1(%2) {uniq_name = "_QMdataEglob_arr_var"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %4 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_arrayEi"}
  %5 = fir.declare %4 {uniq_name = "_QFtest_global_arrayEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %6 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_arrayEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %7 = fir.assumed_size_extent : index
  %8 = fir.shape %7 : (index) -> !fir.shape<1>
  %9 = fir.declare %arg0(%8) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_arrayEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %10 = fir.load %6 : !fir.ref<i32>
  %11 = fir.convert %10 : (i32) -> index
  %12 = fir.convert %c1 : (index) -> i32
  %13 = fir.do_loop %arg2 = %c1 to %11 step %c1 iter_args(%arg3 = %12) -> (i32) {
    fir.store %arg3 to %5 : !fir.ref<i32>
    %14 = fir.array_coor %3(%2) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %15 = fir.load %14 : !fir.ref<f32>
    %16 = fir.convert %15 : (f32) -> i32
    %17 = fir.load %5 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> i64
    %19 = fir.array_coor %9(%8) %18 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %16 to %19 : !fir.ref<i32>
    %20 = fir.load %5 : !fir.ref<i32>
    %21 = arith.addi %20, %12 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %13 to %5 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_array_allocatable(r,n)
//   use data, only : glob_arr_alloc
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_arr_alloc(1)
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_array_allocatable(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_arr_alloc) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdataEglob_arr_alloc"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_array_allocatableEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_array_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_array_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_array_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_1]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONVERT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:             %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:             %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_0]]#0, %[[BOX_DIMS_0]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[BOX_ADDR_0]](%[[SHAPE_SHIFT_0]]) %[[CONSTANT_1]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (f32) -> i32
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_array_allocatable(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_arr_alloc) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMdataEglob_arr_alloc"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_array_allocatableEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_array_allocatableEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_array_allocatableEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_array_allocatableEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %14 = fir.box_addr %13 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %15:3 = fir.box_dims %13, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %16 = fir.shape_shift %15#0, %15#1 : (index, index) -> !fir.shapeshift<1>
    %17 = fir.array_coor %14(%16) %c1 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
    %18 = fir.load %17 : !fir.ref<f32>
    %19 = fir.convert %18 : (f32) -> i32
    %20 = fir.load %4 : !fir.ref<i32>
    %21 = fir.convert %20 : (i32) -> i64
    %22 = fir.array_coor %8(%7) %21 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %19 to %22 : !fir.ref<i32>
    %23 = fir.load %4 : !fir.ref<i32>
    %24 = arith.addi %23, %11 overflow<nsw> : i32
    fir.result %24 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_global_array_pointer(r,n)
//   use data, only : glob_arr_ptr
//   integer :: r(*), n
//   do i=1,n
//      r(i) = glob_arr_ptr(1)
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_global_array_pointer(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMdataEglob_arr_ptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdataEglob_arr_ptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_array_pointerEi"}
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_global_array_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFtest_global_array_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_global_array_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_1]] : (index) -> i32
// CHECK:           %[[LOAD_1:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONVERT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:             %[[SHIFT_0:.*]] = fir.shift %[[BOX_DIMS_0]]#0 : (index) -> !fir.shift<1>
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[LOAD_1]](%[[SHIFT_0]]) %[[CONSTANT_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, index) -> !fir.ref<f32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_0]] : !fir.ref<f32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (f32) -> i32
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_3:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[DECLARE_3]](%[[SHAPE_0]]) %[[CONVERT_3]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[CONVERT_2]] to %[[ARRAY_COOR_1]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_global_array_pointer(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMdataEglob_arr_ptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMdataEglob_arr_ptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_global_array_pointerEi"}
  %4 = fir.declare %3 {uniq_name = "_QFtest_global_array_pointerEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_global_array_pointerEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %6 = fir.assumed_size_extent : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8 = fir.declare %arg0(%7) dummy_scope %0 arg 1 {uniq_name = "_QFtest_global_array_pointerEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %9 = fir.load %5 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = fir.convert %c1 : (index) -> i32
  %12 = fir.do_loop %arg2 = %c1 to %10 step %c1 iter_args(%arg3 = %11) -> (i32) {
    fir.store %arg3 to %4 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %14:3 = fir.box_dims %13, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
    %15 = fir.shift %14#0 : (index) -> !fir.shift<1>
    %16 = fir.array_coor %13(%15) %c1 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, index) -> !fir.ref<f32>
    %17 = fir.load %16 : !fir.ref<f32>
    %18 = fir.convert %17 : (f32) -> i32
    %19 = fir.load %4 : !fir.ref<i32>
    %20 = fir.convert %19 : (i32) -> i64
    %21 = fir.array_coor %8(%7) %20 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %18 to %21 : !fir.ref<i32>
    %22 = fir.load %4 : !fir.ref<i32>
    %23 = arith.addi %22, %11 overflow<nsw> : i32
    fir.result %23 : i32
  }
  fir.store %12 to %4 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_scalar_optional(r,x,n)
//   integer :: r(*), n
//   integer, optional :: x
//   do i=1,n
//      r(i) = x
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_scalar_optional(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "x", fir.optional},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_optionalEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_scalar_optionalEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_scalar_optionalEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_scalar_optionalEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_dummy_scalar_optionalEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_2]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_1]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_scalar_optional(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<i32> {fir.bindc_name = "x", fir.optional}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_optionalEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_scalar_optionalEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_scalar_optionalEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_scalar_optionalEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_dummy_scalar_optionalEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<i32>
    %13 = fir.load %2 : !fir.ref<i32>
    %14 = fir.convert %13 : (i32) -> i64
    %15 = fir.array_coor %6(%5) %14 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %12 to %15 : !fir.ref<i32>
    %16 = fir.load %2 : !fir.ref<i32>
    %17 = arith.addi %16, %10 overflow<nsw> : i32
    fir.result %17 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_scalar_allocatable_optional(r,x,n)
//   integer :: r(*), n
//   integer, allocatable, optional :: x
//   do i=1,n
//      r(i) = x
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_scalar_allocatable_optional(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "x", fir.optional},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {fortran_attrs = #fir.var_attrs<allocatable, optional>, uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEx"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.heap<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_scalar_allocatable_optional(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "x", fir.optional}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<allocatable, optional>, uniq_name = "_QFtest_dummy_scalar_allocatable_optionalEx"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %13 = fir.box_addr %12 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
    %14 = fir.load %13 : !fir.heap<i32>
    %15 = fir.load %2 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.array_coor %6(%5) %16 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %14 to %17 : !fir.ref<i32>
    %18 = fir.load %2 : !fir.ref<i32>
    %19 = arith.addi %18, %10 overflow<nsw> : i32
    fir.result %19 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}

// -----
// subroutine test_dummy_scalar_pointer_optional(r,x,n)
//   integer :: r(*), n
//   integer, pointer, optional :: x
//   do i=1,n
//      r(i) = x
//   end do
// end subroutine
// CHECK-LABEL:   func.func @_QPtest_dummy_scalar_pointer_optional(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "x", fir.optional},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_pointer_optionalEi"}
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = "_QFtest_dummy_scalar_pointer_optionalEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG2]] dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFtest_dummy_scalar_pointer_optionalEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[ASSUMED_SIZE_EXTENT_0:.*]] = fir.assumed_size_extent : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[ASSUMED_SIZE_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ARG0]](%[[SHAPE_0]]) dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_dummy_scalar_pointer_optionalEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {fortran_attrs = #fir.var_attrs<optional, pointer>, uniq_name = "_QFtest_dummy_scalar_pointer_optionalEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONSTANT_0]] : (index) -> i32
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONVERT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONVERT_1]]) -> (i32) {
// CHECK:             fir.store %[[VAL_1]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
// CHECK:             %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.ptr<i32>
// CHECK:             %[[LOAD_3:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[CONVERT_2:.*]] = fir.convert %[[LOAD_3]] : (i32) -> i64
// CHECK:             %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[DECLARE_2]](%[[SHAPE_0]]) %[[CONVERT_2]] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[LOAD_2]] to %[[ARRAY_COOR_0]] : !fir.ref<i32>
// CHECK:             %[[LOAD_4:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONVERT_1]] overflow<nsw> : i32
// CHECK:             fir.result %[[ADDI_0]] : i32
// CHECK:           }
// CHECK:           fir.store %[[DO_LOOP_0]] to %[[DECLARE_0]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest_dummy_scalar_pointer_optional(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "r"}, %arg1: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "x", fir.optional}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_dummy_scalar_pointer_optionalEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_dummy_scalar_pointer_optionalEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dummy_scalar_pointer_optionalEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.assumed_size_extent : index
  %5 = fir.shape %4 : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_scalar_pointer_optionalEr"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %7 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<optional, pointer>, uniq_name = "_QFtest_dummy_scalar_pointer_optionalEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.dscope) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  %8 = fir.load %3 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = fir.convert %c1 : (index) -> i32
  %11 = fir.do_loop %arg3 = %c1 to %9 step %c1 iter_args(%arg4 = %10) -> (i32) {
    fir.store %arg4 to %2 : !fir.ref<i32>
    %12 = fir.load %7 : !fir.ref<!fir.box<!fir.ptr<i32>>>
    %13 = fir.box_addr %12 : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
    %14 = fir.load %13 : !fir.ptr<i32>
    %15 = fir.load %2 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.array_coor %6(%5) %16 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
    fir.store %14 to %17 : !fir.ref<i32>
    %18 = fir.load %2 : !fir.ref<i32>
    %19 = arith.addi %18, %10 overflow<nsw> : i32
    fir.result %19 : i32
  }
  fir.store %11 to %2 : !fir.ref<i32>
  return
}
