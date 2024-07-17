! Test passing polymorphic variable for non-polymorphic dummy arguments:
! RUN: bbc -emit-hlfir -o - -I nowhere %s | FileCheck %s

subroutine test_sequence_association(x)
  type t
    integer :: i
  end type
  interface
    subroutine sequence_assoc(x, n)
      import :: t
      type(t) :: x(n)
    end subroutine
  end interface
  class(t) :: x(:, :)
  call sequence_assoc(x, 100)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_sequence_association(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.copy_in %[[VAL_3]]#0 to %[[VAL_1]] : (!fir.class<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>, i1)
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]]#0 : (!fir.class<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>) -> !fir.ref<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>) -> !fir.ref<!fir.array<?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>
! CHECK:           fir.call @_QPsequence_assoc(%[[VAL_7]], %{{.*}})
! CHECK:           hlfir.copy_out %[[VAL_1]], %[[VAL_5]]#1 to %[[VAL_3]]#0 : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>>>, i1, !fir.class<!fir.array<?x?x!fir.type<_QFtest_sequence_associationTt{i:i32}>>>) -> ()
