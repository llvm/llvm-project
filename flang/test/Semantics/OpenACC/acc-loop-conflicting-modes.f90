! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc -Werror
! RUN: %flang_fc1 -fopenacc -fsyntax-only %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fopenacc -fsyntax-only -Werror -w %s
! RUN: %flang_fc1 -fopenacc -fsyntax-only -Werror -Wno-openacc-usage %s

! Conflicting SEQ and INDEPENDENT modes are both ignored, regardless of order.
! CHECK-COUNT-11: warning: SEQ and INDEPENDENT clauses on the
! CHECK-NOT: warning:
! CHECK-NOT: error:
subroutine conflicting_modes(a)
  real :: a(10)
  integer :: i

  !$acc kernels
  !WARNING: SEQ and INDEPENDENT clauses on the LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc loop seq independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels

  !$acc kernels
  !WARNING: SEQ and INDEPENDENT clauses on the LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc loop seq seq independent
  do i = 1, 10
    a(i) = i
  end do
  !WARNING: SEQ and INDEPENDENT clauses on the LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc loop independent independent seq
  do i = 1, 10
    a(i) = i
  end do
  !WARNING: SEQ and INDEPENDENT clauses on the LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc loop seq independent seq independent seq independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels

  ! Removing SEQ must also avoid a spurious conflict with GANG.
  !WARNING: SEQ and INDEPENDENT clauses on the PARALLEL LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc parallel loop independent seq gang private(i) seq independent
  do i = 1, 10
    a(i) = i
  end do

  !WARNING: SEQ and INDEPENDENT clauses on the KERNELS LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc kernels loop seq independent
  do i = 1, 10
    a(i) = i
  end do

  !WARNING: SEQ and INDEPENDENT clauses on the SERIAL LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc serial loop independent seq
  do i = 1, 10
    a(i) = i
  end do

  !$acc kernels
  !$acc loop seq
  do i = 1, 10
    a(i) = i
  end do
  !$acc loop independent
  do i = 1, 10
    a(i) = i
  end do
  !$acc loop auto
  do i = 1, 10
    a(i) = i
  end do
  !$acc end kernels
end subroutine

! Recovery deliberately does not depend on the associated DO's loop control.
! DO WHILE is already accepted by semantics with a plain LOOP directive, even
! though OpenACC lowering currently crashes on that form. That separate lowering
! bug must not prevent the same clause recovery here. Keep this a semantics-only
! test: it checks the recovery decision without requiring the lowering bug to be
! fixed or preserving the crash as expected behavior.
subroutine conflicting_modes_while(a)
  real :: a(10)
  integer :: i

  i = 0
  !$acc kernels
  !$acc loop
  do while (i < 10)
    i = i + 1
    a(i) = i
  end do
  !WARNING: SEQ and INDEPENDENT clauses on the LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc loop seq independent seq independent
  do while (i > 0)
    a(i) = i
    i = i - 1
  end do
  !$acc end kernels

  !WARNING: SEQ and INDEPENDENT clauses on the PARALLEL LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc parallel loop independent seq
  do while (i < 10)
    i = i + 1
    a(i) = i
  end do

  !WARNING: SEQ and INDEPENDENT clauses on the KERNELS LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc kernels loop seq independent
  do while (i > 0)
    a(i) = i
    i = i - 1
  end do

  !WARNING: SEQ and INDEPENDENT clauses on the SERIAL LOOP directive are ignored because they are mutually exclusive [-Wopenacc-usage]
  !$acc serial loop independent seq independent seq
  do while (i < 10)
    i = i + 1
    a(i) = i
  end do
end subroutine
