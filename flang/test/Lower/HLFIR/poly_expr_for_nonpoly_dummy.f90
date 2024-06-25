! Test passing polymorphic expression for non-polymorphic contiguous
! dummy argument:
! RUN: bbc -emit-hlfir -o - -I nowhere %s | FileCheck %s

module types
  type t
  end type t
contains
  subroutine callee(x)
    type(t), intent(in) :: x(:)
  end subroutine callee
end module types

subroutine test1(x)
  use types
  class(t), intent(in) :: x(:)
  call callee(cshift(x, 1))
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.shift<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>)
! CHECK:           %[[VAL_22:.*]] = arith.constant true
! CHECK:           %[[VAL_23:.*]] = hlfir.as_expr %[[VAL_21]]#0 move %[[VAL_22]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>
! CHECK:           %[[VAL_24:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_21]]#0, %[[VAL_24]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_25]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_27:.*]]:3 = hlfir.associate %[[VAL_23]](%[[VAL_26]]) {adapt.valuebyref} : (!hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.shape<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1)
! CHECK:           %[[VAL_28:.*]] = fir.rebox %[[VAL_27]]#0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>
! CHECK:           %[[VAL_29:.*]]:2 = hlfir.copy_in %[[VAL_28]] to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, i1)
! CHECK:           fir.call @_QMtypesPcallee(%[[VAL_29]]#0) fastmath<contract> : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_29]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, i1) -> ()
! CHECK:           hlfir.end_associate %[[VAL_27]]#0, %[[VAL_27]]#2 : !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1
! CHECK:           hlfir.destroy %[[VAL_23]] : !hlfir.expr<?x!fir.type<_QMtypesTt>?>

subroutine test2(x)
  use types
  class(t), intent(in) :: x(:)
  call callee((x))
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK:           %[[VAL_5:.*]] = hlfir.elemental %{{.*}} mold %{{.*}} unordered : (!fir.shape<1>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?> {
! CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_5]](%{{.*}}) {adapt.valuebyref} : (!hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.shape<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1)
! CHECK:           %[[VAL_10:.*]] = fir.rebox %[[VAL_9]]#0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.copy_in %[[VAL_10]] to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, i1)
! CHECK:           fir.call @_QMtypesPcallee(%[[VAL_11]]#0) fastmath<contract> : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_11]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, i1) -> ()
! CHECK:           hlfir.end_associate %[[VAL_9]]#0, %[[VAL_9]]#2 : !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1
! CHECK:           hlfir.destroy %[[VAL_5]] : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
