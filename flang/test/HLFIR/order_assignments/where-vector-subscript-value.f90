! Test that a vector subscripted designator whose element is a variable can
! be used where a value is required inside a masked WHERE.
! RUN: bbc -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" %s | FileCheck %s

module where_vector_subscript_value
  type :: t
    integer :: v
  end type
contains
  elemental integer function g(x)
    type(t), intent(in) :: x
    g = x%v
  end function

  subroutine elemental_call_parameter(r, idx, msk)
    type(t), parameter :: q(3) = [t(1), t(2), t(3)]
    integer :: r(3), idx(3)
    logical :: msk(3)
    where (msk) r = g(q(idx))
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePelemental_call_parameter(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : ({{.*}}, i64) -> !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>) -> !hlfir.expr<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             %[[ASSOC:.*]]:3 = hlfir.associate %[[VAL]] {adapt.valuebyref}
! CHECK:             fir.call @_QMwhere_vector_subscript_valuePg(%[[ASSOC]]#0)
! CHECK:             hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2

  subroutine merge_derived(dd, de, idx, msk)
    type(t) :: dd(3), de(3)
    integer :: idx(3)
    logical :: msk(3)
    where (msk) de = merge(dd(idx), de, msk)
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePmerge_derived(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : ({{.*}}, i64) -> !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>) -> !hlfir.expr<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             %[[ASSOC:.*]]:3 = hlfir.associate %[[VAL]] {adapt.valuebyref}
! CHECK:             arith.select %{{.*}}, %[[ASSOC]]#0, %{{.*}} : !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>

  subroutine parens_derived(dd, de, idx, msk)
    type(t) :: dd(3), de(3)
    integer :: idx(3)
    logical :: msk(3)
    where (msk) de = (dd(idx))
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePparens_derived(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : ({{.*}}, i64) -> !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>) -> !hlfir.expr<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             %[[PAREN:.*]] = hlfir.no_reassoc %[[VAL]] : !hlfir.expr<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             hlfir.assign %[[PAREN]] to %{{.*}} : !hlfir.expr<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>, !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>

  subroutine parens_character(cc, cd, idx, msk)
    character(4) :: cc(3), cd(3)
    integer :: idx(3)
    logical :: msk(3)
    where (msk) cd = (cc(idx))
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePparens_character(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  typeparams %{{.*}} : ({{.*}}, i64, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:             %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.char<1,4>>) -> !hlfir.expr<!fir.char<1,4>>
! CHECK:             hlfir.no_reassoc %[[VAL]] : !hlfir.expr<!fir.char<1,4>>

  ! Direct assignment: the element is assigned without an hlfir.as_expr copy.
  subroutine assign_derived(dd, de, idx, msk)
    type(t) :: dd(3), de(3)
    integer :: idx(3)
    logical :: msk(3)
    where (msk) de = dd(idx)
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePassign_derived(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : ({{.*}}, i64) -> !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK-NOT:         hlfir.as_expr
! CHECK:             hlfir.assign %[[ELT]] to %{{.*}} : !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>, !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>

  ! Assumed-shape dummies are used because explicit-shape character dummies
  ! (passed as fir.boxchar) are conservatively assumed to alias with the
  ! other dummies, which would cause the RHS to be saved before the loop.
  subroutine assign_character(cc, cd, idx, msk)
    character(4) :: cc(:), cd(:)
    integer :: idx(:)
    logical :: msk(:)
    where (msk) cd = cc(idx)
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePassign_character(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  typeparams %{{.*}} : ({{.*}}, i64, index) -> !fir.ref<!fir.char<1,4>>
! CHECK-NOT:         hlfir.as_expr
! CHECK:             hlfir.assign %[[ELT]] to %{{.*}} : !fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>

  ! The RHS reads the LHS through the vector subscript: the read must be seen
  ! by the scheduling so that the RHS is saved before the assignment.
  subroutine overlap_character(cc, idx, msk)
    character(4) :: cc(:)
    integer :: idx(:)
    logical :: msk(:)
    where (msk) cc = cc(idx)
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePoverlap_character(
! CHECK:         %[[STACK:.*]] = fir.call @_FortranACreateValueStack(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  typeparams %{{.*}} : ({{.*}}, i64, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:             hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.char<1,4>>) -> !hlfir.expr<!fir.char<1,4>>
! CHECK:             fir.call @_FortranAPushValue(%[[STACK]],
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             fir.call @_FortranAValueAt(%[[STACK]],
! CHECK:             hlfir.assign %{{.*}} to %{{.*}} : !fir.heap<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
! CHECK:         fir.call @_FortranADestroyValueStack(%[[STACK]])

  subroutine overlap_derived_parens(de, idx, msk)
    type(t) :: de(:)
    integer :: idx(:)
    logical :: msk(:)
    where (msk) de = (de(idx))
  end subroutine
! CHECK-LABEL: func.func @_QMwhere_vector_subscript_valuePoverlap_derived_parens(
! CHECK:         %[[STACK:.*]] = fir.call @_FortranACreateValueStack(
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : ({{.*}}, i64) -> !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>) -> !hlfir.expr<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:             hlfir.no_reassoc %[[VAL]]
! CHECK:             fir.call @_FortranAPushValue(%[[STACK]],
! CHECK:         fir.do_loop
! CHECK:           fir.if
! CHECK:             fir.call @_FortranAValueAt(%[[STACK]],
! CHECK:             hlfir.assign %{{.*}} to %{{.*}} : !fir.heap<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>, !fir.ref<!fir.type<_QMwhere_vector_subscript_valueTt{v:i32}>>
! CHECK:         fir.call @_FortranADestroyValueStack(%[[STACK]])
end module
