! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc -Werror
! RUN: %flang_fc1 -fopenacc -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenacc -fsyntax-only -Werror -w %s
! RUN: %flang_fc1 -fopenacc -fsyntax-only -Werror -Wno-openacc-usage %s

! Repetitions retain one clause and emit one warning per directive.
! CHECK-COUNT-8: warning: Repeated
! CHECK-NOT: warning:
! CHECK-NOT: error:
subroutine repeated_seq_loop(a)
  real :: a(10)
  integer :: i
  !$acc kernels
  !WARNING: Repeated SEQ clauses on the LOOP directive are ignored [-Wopenacc-usage]
  !$acc loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels
end subroutine

subroutine repeated_seq_parallel_loop(a)
  real :: a(10)
  integer :: i
  !WARNING: Repeated SEQ clauses on the PARALLEL LOOP directive are ignored [-Wopenacc-usage]
  !$acc parallel loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
end subroutine

subroutine repeated_seq_kernels_loop(a)
  real :: a(10)
  integer :: i
  !WARNING: Repeated SEQ clauses on the KERNELS LOOP directive are ignored [-Wopenacc-usage]
  !$acc kernels loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
end subroutine

subroutine repeated_seq_serial_loop(a)
  real :: a(10)
  integer :: i
  !WARNING: Repeated SEQ clauses on the SERIAL LOOP directive are ignored [-Wopenacc-usage]
  !$acc serial loop seq private(i) seq seq
  do i = 1, 10
    a(i) = i
  end do
end subroutine

subroutine repeated_independent_loop(a)
  real :: a(10)
  integer :: i
  !$acc kernels
  !WARNING: Repeated INDEPENDENT clauses on the LOOP directive are ignored [-Wopenacc-usage]
  !$acc loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels
end subroutine

subroutine repeated_independent_parallel_loop(a)
  real :: a(10)
  integer :: i
  !WARNING: Repeated INDEPENDENT clauses on the PARALLEL LOOP directive are ignored [-Wopenacc-usage]
  !$acc parallel loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
end subroutine

subroutine repeated_independent_kernels_loop(a)
  real :: a(10)
  integer :: i
  !WARNING: Repeated INDEPENDENT clauses on the KERNELS LOOP directive are ignored [-Wopenacc-usage]
  !$acc kernels loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
end subroutine

subroutine repeated_independent_serial_loop(a)
  real :: a(10)
  integer :: i
  !WARNING: Repeated INDEPENDENT clauses on the SERIAL LOOP directive are ignored [-Wopenacc-usage]
  !$acc serial loop independent private(i) independent independent
  do i = 1, 10
    a(i) = i
  end do
end subroutine
