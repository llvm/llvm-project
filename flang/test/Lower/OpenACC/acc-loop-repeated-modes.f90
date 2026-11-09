! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Repeated clauses must retain their mode, rather than fall back to AUTO.
subroutine repeated_seq_loop(a)
  real :: a(10)
  integer :: i
  !$acc kernels
  !$acc loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_seq_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq{{ *$}}

subroutine repeated_seq_parallel_loop(a)
  real :: a(10)
  integer :: i
  !$acc parallel loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_seq_parallel_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq{{ *$}}

subroutine repeated_seq_kernels_loop(a)
  real :: a(10)
  integer :: i
  !$acc kernels loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_seq_kernels_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq{{ *$}}

subroutine repeated_seq_serial_loop(a)
  real :: a(10)
  integer :: i
  !$acc serial loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_seq_serial_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq{{ *$}}

subroutine repeated_independent_loop(a)
  real :: a(10)
  integer :: i
  !$acc kernels
  !$acc loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_independent_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) independent{{ *$}}

subroutine repeated_independent_parallel_loop(a)
  real :: a(10)
  integer :: i
  !$acc parallel loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_independent_parallel_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) independent{{ *$}}

subroutine repeated_independent_kernels_loop(a)
  real :: a(10)
  integer :: i
  !$acc kernels loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_independent_kernels_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) independent{{ *$}}

subroutine repeated_independent_serial_loop(a)
  real :: a(10)
  integer :: i
  !$acc serial loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_independent_serial_loop
! CHECK: acc.loop
! CHECK-SAME: private(
! CHECK: } inclusiveUpperbound(array<i1: true>) independent{{ *$}}
