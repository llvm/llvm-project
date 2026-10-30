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
end module
