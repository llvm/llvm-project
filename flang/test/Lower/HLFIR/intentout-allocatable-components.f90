! Test that allocatable components of non pointer/non allocatable INTENT(OUT)
! dummy arguments are deallocated.
! RUN: bbc -emit-hlfir -polymorphic-type %s -o - -I nowhere | FileCheck %s

subroutine test_intentout_component_deallocate(a)
  type :: t
    integer, allocatable :: x
  end type
  type(t), intent(out) :: a
end subroutine
! CHECK-LABEL:   func.func @_QPtest_intentout_component_deallocate(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QFtest_intentout_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFtest_intentout_component_deallocateEa"}
! CHECK:  %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#1 : (!fir.ref<!fir.type<_QFtest_intentout_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFtest_intentout_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.type<_QFtest_intentout_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
! CHECK:  %[[VAL_4:.*]] = fir.call @_FortranADestroy(%[[VAL_3]]) fastmath<contract> : (!fir.box<none>) -> none

subroutine test_intentout_optional_component_deallocate(a)
  type :: t
    integer, allocatable :: x
  end type
  type(t), optional, intent(out) :: a
end subroutine
! CHECK-LABEL:   func.func @_QPtest_intentout_optional_component_deallocate(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QFtest_intentout_optional_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_out, optional>, uniq_name = "_QFtest_intentout_optional_component_deallocateEa"}
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#1 : (!fir.ref<!fir.type<_QFtest_intentout_optional_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>) -> i1
! CHECK:  fir.if %[[VAL_2]] {
! CHECK:    %[[VAL_3:.*]] = fir.embox %[[VAL_1]]#1 : (!fir.ref<!fir.type<_QFtest_intentout_optional_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFtest_intentout_optional_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:    %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.type<_QFtest_intentout_optional_component_deallocateTt{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
! CHECK:    %[[VAL_5:.*]] = fir.call @_FortranADestroy(%[[VAL_4]]) fastmath<contract> : (!fir.box<none>) -> none
! CHECK:  }
