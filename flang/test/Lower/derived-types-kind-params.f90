! Test lowering of derived type with kind parameters
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module m
  type t(k1, k2)
    integer(4), kind :: k1 = 7
    integer(8), kind :: k2
    character(k1) :: c(k2)
  end type

  type t2(k1, k2)
    integer(4), kind :: k1
    integer(8), kind :: k2
    type(t(k1+3, k2+4)) :: at
  end type

  type t3(k)
    integer, kind :: k
    type(t3(k)), pointer :: at3
  end type

  type t4(k)
    integer, kind :: k
    real(-k) :: i
  end type

contains

! -----------------------------------------------------------------------------
!            Test mangling of derived type with kind parameters
! -----------------------------------------------------------------------------

  ! CHECK-LABEL: func @_QMmPfoo
  ! CHECK-SAME: !fir.ref<!fir.type<_QMmTtK7K12{c:!fir.array<12x!fir.char<1,?>>
  subroutine foo(at)
    type(t(k2=12)) :: at
  end subroutine

  ! CHECK-LABEL: func @_QMmPfoo2
  ! CHECK-SAME: !fir.ref<!fir.type<_QMmTt2K12K13{at:!fir.type<_QMmTtK15K17{c:!fir.array<17x!fir.char<1,?>>}>}>>
  subroutine foo2(at2)
    type(t2(12, 13)) :: at2
  end subroutine

  ! CHECK-LABEL: func @_QMmPfoo3
  ! CHECK-SAME: !fir.ref<!fir.type<_QMmTt3K7{at3:!fir.box<!fir.ptr<!fir.type<_QMmTt3K7>>>}>>
  subroutine foo3(at3)
    type(t3(7)) :: at3
  end subroutine

  ! CHECK-LABEL: func @_QMmPfoo4
  ! CHECK-SAME: !fir.ref<!fir.type<_QMmTt4KN4{i:f32}>>
  subroutine foo4(at4)
    type(t4(-4)) :: at4
  end subroutine
end module
