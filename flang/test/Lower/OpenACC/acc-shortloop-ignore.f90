! Test that non-standard shortloop clause is accepted and ignored with a
! warning.

! RUN: %flang_fc1 -fopenacc -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK:  warning: Non-standard shortloop clause ignored
! CHECK:  warning: Non-standard shortloop clause ignored
! CHECK:  warning: Non-standard shortloop clause ignored
! CHECK:  warning: Non-standard shortloop clause ignored

subroutine test_loop(a, b, c)
  implicit none
  real, dimension(100) :: a,b,c
  integer :: i
  !$acc loop vector shortloop
  do i=1,100
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_loop
! CHECK: acc.loop vector

subroutine test_kernels_loop(a, b, c)
  implicit none
  real, dimension(100) :: a,b,c
  integer :: i
  !$acc kernels loop vector shortloop
  do i=1,100
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_kernels_loop
! CHECK: acc.loop combined(kernels) vector

subroutine test_parallel_loop(a, b, c)
  implicit none
  real, dimension(100) :: a,b,c
  integer :: i
  !$acc parallel loop vector shortloop
  do i=1,100
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_parallel_loop
! CHECK: acc.loop combined(parallel) vector

subroutine test_serial_loop(a, b, c)
  implicit none
  real, dimension(100) :: a,b,c
  integer :: i
  !$acc serial loop vector shortloop
  do i=1,100
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_serial_loop
! CHECK: acc.loop combined(serial) vector
