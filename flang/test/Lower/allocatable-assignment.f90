! Test allocatable assignments
! RUN: bbc -emit-hlfir  %s -o - | FileCheck %s

module alloc_assign
  type t
    integer :: i
  end type
contains

! -----------------------------------------------------------------------------
!            Test simple scalar RHS
! -----------------------------------------------------------------------------

subroutine test_simple_scalar(x)
  real, allocatable  :: x
  x = 42.
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_simple_scalar(
! CHECK-SAME:                                                  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_simple_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:           hlfir.assign %[[VAL_3]] to %[[VAL_2]]#0 realloc : f32, !fir.ref<!fir.box<!fir.heap<f32>>>

subroutine test_simple_local_scalar()
  real, allocatable  :: x
  x = 42.
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_simple_local_scalar() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "x", uniq_name = "_QMalloc_assignFtest_simple_local_scalarEx"}
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<f32>
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_simple_local_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_3]]#0 realloc : f32, !fir.ref<!fir.box<!fir.heap<f32>>>

! -----------------------------------------------------------------------------
!            Test character scalar RHS
! -----------------------------------------------------------------------------

subroutine test_deferred_char_scalar(x)
  character(:), allocatable  :: x

  x = "Hello world!"
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_deferred_char_scalar(
! CHECK-SAME:                                                         %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_deferred_char_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QQclX48656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[VAL_4]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX48656C6C6F20776F726C6421"} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
! CHECK:           hlfir.assign %[[VAL_5]]#0 to %[[VAL_2]]#0 realloc : !fir.ref<!fir.char<1,12>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>

subroutine test_cst_char_scalar(x)
  character(10), allocatable  :: x
  x = "Hello world!"
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_cst_char_scalar(
! CHECK-SAME:                                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_2]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_cst_char_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>, index, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>)
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQclX48656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %[[VAL_5]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX48656C6C6F20776F726C6421"} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
! CHECK:           hlfir.assign %[[VAL_6]]#0 to %[[VAL_3]]#0 realloc keep_lhs_len : !fir.ref<!fir.char<1,12>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>

subroutine test_dyn_char_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x
  x = "Hello world!"
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_dyn_char_scalar(
! CHECK-SAME:                                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                    %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_dyn_char_scalarEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_7]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_dyn_char_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, i32, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QQclX48656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX48656C6C6F20776F726C6421"} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
! CHECK:           hlfir.assign %[[VAL_11]]#0 to %[[VAL_8]]#0 realloc keep_lhs_len : !fir.ref<!fir.char<1,12>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>

subroutine test_derived_scalar(x, s)
  type(t), allocatable  :: x
  type(t) :: s
  x = s
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_derived_scalar(
! CHECK-SAME:                                                   %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                   %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>> {fir.bindc_name = "s"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_derived_scalarEs"} : (!fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>, !fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_derived_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>)
! CHECK:           hlfir.assign %[[VAL_3]]#0 to %[[VAL_4]]#0 realloc : !fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>

! -----------------------------------------------------------------------------
!            Test numeric/logical array RHS
! -----------------------------------------------------------------------------

subroutine test_from_cst_shape_array(x, y)
  real, allocatable  :: x(:, :)
  real :: y(2, 3)
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_from_cst_shape_array(
! CHECK-SAME:                                                         %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                         %[[VAL_1:.*]]: !fir.ref<!fir.array<2x3xf32>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_from_cst_shape_arrayEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_6]]) dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_from_cst_shape_arrayEy"} : (!fir.ref<!fir.array<2x3xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<2x3xf32>>, !fir.ref<!fir.array<2x3xf32>>)
! CHECK:           hlfir.assign %[[VAL_7]]#0 to %[[VAL_3]]#0 realloc : !fir.ref<!fir.array<2x3xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>

subroutine test_from_dyn_shape_array(x, y)
  real, allocatable  :: x(:, :)
  real :: y(:, :)
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_from_dyn_shape_array(
! CHECK-SAME:                                                         %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                         %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_from_dyn_shape_arrayEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_from_dyn_shape_arrayEy"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           hlfir.assign %[[VAL_4]]#0 to %[[VAL_3]]#0 realloc : !fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>

subroutine test_with_lbounds(x, y)
  real, allocatable  :: x(:, :)
  real :: y(10:, 20:)
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_with_lbounds(
! CHECK-SAME:                                                 %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                 %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_with_lboundsEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : i64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:           %[[VAL_6:.*]] = arith.constant 20 : i64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:           %[[VAL_8:.*]] = fir.shift %[[VAL_5]], %[[VAL_7]] : (index, index) -> !fir.shift<2>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_8]]) dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_with_lboundsEy"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           hlfir.assign %[[VAL_9]]#0 to %[[VAL_3]]#0 realloc : !fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>

subroutine test_runtime_shape(x)
  real, allocatable  :: x(:, :)
  interface
   function return_pointer()
     real, pointer :: return_pointer(:, :)
   end function
  end interface
  x = return_pointer()
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_runtime_shape(
! CHECK-SAME:                                                  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>> {bindc_name = ".result"}
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_runtime_shapeEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_4:.*]] = fir.call @_QPreturn_pointer() fastmath<contract> : () -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:           fir.save_result %[[VAL_4]] to %[[VAL_1]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 realloc : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>

subroutine test_scalar_rhs(x, y)
  real, allocatable  :: x(:)
  real :: y
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_scalar_rhs(
! CHECK-SAME:                                               %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                               %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_scalar_rhsEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_scalar_rhsEy"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 realloc : f32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>

! -----------------------------------------------------------------------------
!            Test character array RHS
! -----------------------------------------------------------------------------


! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char_rhs_scalar(x)
!  character(:), allocatable  :: x(:)
!  x = "Hello world!"
!end subroutine

subroutine test_cst_char_rhs_scalar(x)
  character(10), allocatable  :: x(:)
  x = "Hello world!"
  ! TODO: runtime error if unallocated
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_cst_char_rhs_scalar(
! CHECK-SAME:                                                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_2]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_cst_char_rhs_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, index, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>)
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQclX48656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %[[VAL_5]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX48656C6C6F20776F726C6421"} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
! CHECK:           hlfir.assign %[[VAL_6]]#0 to %[[VAL_3]]#0 realloc keep_lhs_len : !fir.ref<!fir.char<1,12>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>

subroutine test_dyn_char_rhs_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x(:)
  x = "Hello world!"
  ! TODO: runtime error if unallocated
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_dyn_char_rhs_scalar(
! CHECK-SAME:                                                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_dyn_char_rhs_scalarEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_7]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_dyn_char_rhs_scalarEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, i32, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QQclX48656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
! CHECK:           %[[VAL_10:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX48656C6C6F20776F726C6421"} : (!fir.ref<!fir.char<1,12>>, index) -> (!fir.ref<!fir.char<1,12>>, !fir.ref<!fir.char<1,12>>)
! CHECK:           hlfir.assign %[[VAL_11]]#0 to %[[VAL_8]]#0 realloc keep_lhs_len : !fir.ref<!fir.char<1,12>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>

! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char(x, c)
!  character(:), allocatable  :: x(:)
!  character(12) :: c(20)
!  x = "Hello world!"
!end subroutine

subroutine test_cst_char(x, c)
  character(10), allocatable  :: x(:)
  character(12) :: c(20)
  x = c
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_cst_char(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                             %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<20x!fir.char<1,12>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 12 : index
! CHECK:           %[[VAL_6:.*]] = arith.constant 20 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_7]]) typeparams %[[VAL_5]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_cst_charEc"} : (!fir.ref<!fir.array<20x!fir.char<1,12>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.ref<!fir.array<20x!fir.char<1,12>>>, !fir.ref<!fir.array<20x!fir.char<1,12>>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_9]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_cst_charEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, index, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>)
! CHECK:           hlfir.assign %[[VAL_8]]#0 to %[[VAL_10]]#0 realloc keep_lhs_len : !fir.ref<!fir.array<20x!fir.char<1,12>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>

subroutine test_dyn_char(x, n, c)
  integer :: n
  character(n), allocatable  :: x(:)
  character(*) :: c(20)
  x = c
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_dyn_char(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                             %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                             %[[VAL_2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<20x!fir.char<1,?>>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 20 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_7]]) typeparams %[[VAL_4]]#1 dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_dyn_charEc"} : (!fir.ref<!fir.array<20x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<20x!fir.char<1,?>>>, !fir.ref<!fir.array<20x!fir.char<1,?>>>)
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_dyn_charEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_13]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_dyn_charEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, i32, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
! CHECK:           hlfir.assign %[[VAL_8]]#0 to %[[VAL_14]]#0 realloc keep_lhs_len : !fir.box<!fir.array<20x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>

subroutine test_derived_with_init(x, y)
  type t 
    integer, allocatable :: a(:)
  end type                                                                                     
  type(t), allocatable :: x                                                                    
  type(t) :: y                                                                                 
  ! The allocatable component of `x` need to be initialized
  ! during the automatic allocation (setting its rank and allocation
  ! status) before it is assigned with the component of `y` 
  x = y
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_derived_with_init(
! CHECK-SAME:                                                      %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                      %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_derived_with_initEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>)
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_derived_with_initEy"} : (!fir.ref<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
! CHECK:           hlfir.assign %[[VAL_10]]#0 to %[[VAL_9]]#0 realloc : !fir.ref<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>

subroutine test_vector_subscript(x, y, v)
  ! Test that the new shape is computed correctly in presence of
  ! vector subscripts on the RHS and that it is used to allocate
  ! the new storage and to drive the implicit loop.
  integer, allocatable :: x(:)
  integer :: y(:), v(:)
  x = y(v)
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_vector_subscript(
! CHECK-SAME:                                                     %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                                     %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"},
! CHECK-SAME:                                                     %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "v"}) {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_vector_subscriptEv"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_vector_subscriptEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QMalloc_assignFtest_vector_subscriptEy"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:           hlfir.assign %[[VAL_16]] to %[[VAL_5]]#0 realloc : !hlfir.expr<?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

subroutine test_both_sides_with_elemental_call(x)
  interface
     elemental real function elt(x)
       real, intent(in) :: x
     end function elt
  end interface
  real, allocatable  :: x(:)
  x = elt(x)
end subroutine
! CHECK-LABEL:   func.func @_QMalloc_assignPtest_both_sides_with_elemental_call(
! CHECK-SAME:                                                                   %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMalloc_assignFtest_both_sides_with_elemental_callEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:           ^bb0(%[[VAL_8:.*]]: index):
! CHECK:             %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_10]]#0, %[[VAL_11]] : index
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_8]], %[[VAL_12]] : index
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_3]] (%[[VAL_13]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:             %[[VAL_15:.*]] = fir.call @_QPelt(%[[VAL_14]]) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<f32>) -> f32
! CHECK:             hlfir.yield_element %[[VAL_15]] : f32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_2]]#0 realloc : !hlfir.expr<?xf32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>


end module

!  use alloc_assign
!  real :: y(2, 3) = reshape([1,2,3,4,5,6], [2,3])
!  real, allocatable :: x (:, :)
!  allocate(x(2,2))
!  call test_with_lbounds(x, y) 
!  print *, x(10, 20)
!  print *, x
!end
