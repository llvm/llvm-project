! RUN: bbc -emit-hlfir -o - %s -I nowhere | FileCheck %s

module polymorphic_expressions_types
  type t
     integer c
  end type t
end module polymorphic_expressions_types

! Test that proper polymorphic type used for hlfir.as_expr,
! and that hlfir.association has polymorphic result type.
subroutine test1(a)
  use polymorphic_expressions_types
  interface
     subroutine callee(x)
       use polymorphic_expressions_types
       class(t) :: x(:)
     end subroutine callee
  end interface
  class(t), allocatable :: a
  call callee(spread(a, 1, 2))
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>, !fir.shift<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>)
! CHECK:           %[[VAL_22:.*]] = arith.constant true
! CHECK:           %[[VAL_23:.*]] = hlfir.as_expr %[[VAL_21]]#0 move %[[VAL_22]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>, i1) -> !hlfir.expr<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>?>
! CHECK:           %[[VAL_24:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_21]]#0, %[[VAL_24]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_25]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_27:.*]]:3 = hlfir.associate %[[VAL_23]](%[[VAL_26]]) {adapt.valuebyref} : (!hlfir.expr<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>?>, !fir.shape<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>, i1)
! CHECK:           %[[VAL_28:.*]] = fir.rebox %[[VAL_27]]#0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>) -> !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>
! CHECK:           fir.call @_QPcallee(%[[VAL_28]]) fastmath<contract> : (!fir.class<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_27]]#0, %[[VAL_27]]#2 : !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>>>>, i1
! CHECK:           hlfir.destroy %[[VAL_23]] : !hlfir.expr<?x!fir.type<_QMpolymorphic_expressions_typesTt{c:i32}>?>
