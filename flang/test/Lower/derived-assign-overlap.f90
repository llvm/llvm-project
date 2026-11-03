! Test how lowering selects the no_overlap attribute on the fir.copy emitted for
! a simple derived-type assignment: the attribute (memcpy) must be present only
! when the operands are proven disjoint; otherwise the copy lowers to memmove.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! A Cray pointee may alias the TARGET variable it points into, so the two
! operands are potentially overlapping and the copy must not carry no_overlap.
subroutine may_alias(x)
  use iso_c_binding, only: c_int
  type, bind(c) :: t
    integer(c_int) :: a(6)
  end type
  type(t), target :: x
  type(t) :: pte
  pointer (p, pte)
  p = loc(x)
  x = pte
end subroutine
! CHECK-LABEL: func.func @_QPmay_alias(
! CHECK:         fir.copy %[[SRC:[0-9]+]] to %[[DST:[0-9]+]] : !fir.ref<!fir.type<_QFmay_aliasTt{a:!fir.array<6xi32>}>>, !fir.ref<!fir.type<_QFmay_aliasTt{a:!fir.array<6xi32>}>>

! Two distinct local variables cannot overlap: no_overlap is set.
subroutine disjoint_control()
  use iso_c_binding, only: c_int
  type, bind(c) :: t
    integer(c_int) :: a(6)
  end type
  type(t) :: x, y
  y%a = 1
  x = y
end subroutine
! CHECK-LABEL: func.func @_QPdisjoint_control(
! CHECK:         fir.copy %[[SRC:[0-9]+]] to %[[DST:[0-9]+]] no_overlap : !fir.ref<!fir.type<_QFdisjoint_controlTt{a:!fir.array<6xi32>}>>, !fir.ref<!fir.type<_QFdisjoint_controlTt{a:!fir.array<6xi32>}>>
