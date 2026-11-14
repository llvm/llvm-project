! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! An explicit OpenACC loop requires integer controls. A REAL lower bound,
! upper bound, or step is an error. A DO inside a compute construct with no
! !$acc loop is implicit and is not diagnosed here.

program acc_loop_real_bounds
  implicit none
  integer :: i, j, n
  real :: r, s, t
  double precision :: d
  complex :: a(2), b

  n = 1
  r = 1.0
  s = 2.0
  t = 1.0
  d = 2.0
  a = (1, 3)
  b = (5, 6)

  !$acc loop
  do i = 1, n
  end do

  !$acc loop
  !ERROR: OpenACC loop bound expression must have INTEGER type
  !ERROR: OpenACC loop bound expression must have INTEGER type
  do i = r, s
  end do

  !$acc loop
  !ERROR: OpenACC loop bound expression must have INTEGER type
  do i = r, n
  end do

  !$acc loop
  !ERROR: OpenACC loop bound expression must have INTEGER type
  do i = 1, n, t
  end do

  !$acc loop
  !ERROR: OpenACC loop bound expression must have INTEGER type
  do i = 1, d
  end do

  !$acc loop
  !ERROR: OpenACC loop bound expression must have INTEGER type
  !ERROR: OpenACC loop bound expression must have INTEGER type
  do i = a(2)%re, b%im
  end do

  !$acc serial loop
  !ERROR: OpenACC loop bound expression must have INTEGER type
  !ERROR: OpenACC loop bound expression must have INTEGER type
  do i = r, s
  end do

  !$acc loop collapse(2)
  do i = 1, n
    !ERROR: OpenACC loop bound expression must have INTEGER type
    !ERROR: OpenACC loop bound expression must have INTEGER type
    do j = r, s
    end do
  end do

  !$acc serial
  do i = r, s
  end do
  !$acc end serial
end program
