! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! The recovered loop must have the same inferred mode as a plain loop.
! In a kernels region, this is AUTO, not INDEPENDENT or SEQ.
subroutine recovered_modes(a)
  real :: a(10)
  integer :: i
  !$acc kernels
  !$acc loop seq independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc loop independent seq
  do i = 1, 10
    a(i) = i
  end do
  !$acc loop seq seq independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc loop independent seq independent seq independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc loop
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPrecovered_modes
! CHECK: acc.kernels
! CHECK: acc.loop
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_{{ *$}}
! CHECK: acc.loop
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_{{ *$}}
! CHECK: acc.loop
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_{{ *$}}
! CHECK: acc.loop
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_{{ *$}}
! CHECK: acc.loop
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_{{ *$}}

! Unrelated scheduling clauses must survive recovery on combined directives.
subroutine recovered_gang(a)
  real :: a(10)
  integer :: i
  !$acc parallel loop gang seq independent seq independent
  do i = 1, 10
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPrecovered_gang
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel) gang
! CHECK: } inclusiveUpperbound(array<i1: true>) independent{{ *$}}
