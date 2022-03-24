! Test calls with POINTER dummy arguments on the caller side.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module call_defs
interface
  subroutine scalar_ptr(p)
    integer, pointer, intent(in) :: p
  end subroutine
  subroutine array_ptr(p)
    integer, pointer, intent(in) :: p(:)
  end subroutine
  subroutine char_array_ptr(p)
    character(:), pointer, intent(in) :: p(:)
  end subroutine
  subroutine non_deferred_char_array_ptr(p)
    character(10), pointer, intent(in) :: p(:)
  end subroutine
end interface
contains

! -----------------------------------------------------------------------------
!     Test passing POINTER actual arguments
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMcall_defsPtest_ptr_to_scalar_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "p"}) {
subroutine test_ptr_to_scalar_ptr(p)
  integer, pointer :: p
! CHECK:  fir.call @_QPscalar_ptr(%[[VAL_0]]) : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> ()
  call scalar_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_ptr_to_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "p"}) {
subroutine test_ptr_to_array_ptr(p)
  integer, pointer :: p(:)
  call array_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_ptr_to_char_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "p"}) {
subroutine test_ptr_to_char_array_ptr(p)
  character(:), pointer :: p(:)
! CHECK:  fir.call @_QPchar_array_ptr(%[[VAL_0]]) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
  call char_array_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_ptr_to_non_deferred_char_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "p"}) {
subroutine test_ptr_to_non_deferred_char_array_ptr(p)
  character(:), pointer :: p(:)
! CHECK:  %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:  fir.call @_QPnon_deferred_char_array_ptr(%[[VAL_1]]) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>) -> ()
  call non_deferred_char_array_ptr(p)
end subroutine

! -----------------------------------------------------------------------------
!     Test passing non-POINTER actual arguments (implicit pointer assignment)
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMcall_defsPtest_non_ptr_to_scalar_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "p", fir.target}) {
subroutine test_non_ptr_to_scalar_ptr(p)
  integer, target :: p
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK:  %[[VAL_2:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  fir.call @_QPscalar_ptr(%[[VAL_1]]) : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> ()
  call scalar_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_non_ptr_to_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "p", fir.target}) {
subroutine test_non_ptr_to_array_ptr(p)
  integer, target :: p(:)
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:  %[[VAL_2:.*]] = fir.rebox %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:  fir.call @_QParray_ptr(%[[VAL_1]]) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> ()
  call array_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_non_ptr_to_array_ptr_lower_bounds(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "p", fir.target}) {
subroutine test_non_ptr_to_array_ptr_lower_bounds(p)
  ! Test that local lower bounds of the actual argument are applied.
  integer, target :: p(42:)
  ! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_2:.*]] = arith.constant 42 : i64
  ! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
  ! CHECK:  %[[VAL_4:.*]] = fir.shift %[[VAL_3]] : (index) -> !fir.shift<1>
  ! CHECK:  %[[VAL_5:.*]] = fir.rebox %[[VAL_0]](%[[VAL_4]]) : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  fir.call @_QParray_ptr(%[[VAL_1]]) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> ()
  call array_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_non_ptr_to_char_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "p", fir.target}) {
subroutine test_non_ptr_to_char_array_ptr(p)
  character(10), target :: p(10)
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:  %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:  %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.array<10x!fir.char<1,10>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:  %[[VAL_8:.*]] = fir.embox %[[VAL_7]](%[[VAL_6]]) typeparams %[[VAL_3]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:  fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  fir.call @_QPchar_array_ptr(%[[VAL_1]]) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
  call char_array_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_non_ptr_to_non_deferred_char_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "p", fir.target}) {
subroutine test_non_ptr_to_non_deferred_char_array_ptr(p)
  character(*), target :: p(:)
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
! CHECK:  %[[VAL_2:.*]] = fir.rebox %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
! CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:  fir.call @_QPnon_deferred_char_array_ptr(%[[VAL_1]]) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>>) -> ()
  call non_deferred_char_array_ptr(p)
end subroutine

! CHECK-LABEL: func @_QMcall_defsPtest_allocatable_to_array_ptr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "p", fir.target}) {
subroutine test_allocatable_to_array_ptr(p)
  integer, allocatable, target :: p(:)
  call array_ptr(p)
  ! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
  ! CHECK:  %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  ! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK:  %[[VAL_6:.*]] = fir.shape_shift %[[VAL_4]]#0, %[[VAL_4]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_5]](%[[VAL_6]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  fir.call @_QParray_ptr(%[[VAL_1]]) : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> ()
end subroutine

end module
