! RUN: %python %S/../test_errors.py %s %flang -fopenacc -facc-allow-default-none-scalars -Wacc-implicit-scalar

! With -facc-allow-default-none-scalars, scalar variables without explicit
! data clauses under DEFAULT(NONE) are allowed and get a warning
! (-Wacc-implicit-scalar) instead of an error.  Arrays continue to error.

subroutine default_none_scalars()
  integer :: a
  integer :: b(10)
  !$acc parallel default(none)
  !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'a' [-Wacc-implicit-scalar]
  a = 1
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-mapping clause
  b(1) = 2
  !$acc end parallel
end subroutine

! Scalar computed before an ACC parallel loop and read-only inside it.
subroutine default_none_scalar_precomputed(x, n, p, q)
  implicit none
  real :: x(n), factor, p, q
  integer :: n, i
  factor = p / q
  !$acc parallel loop default(none) present(x)
  !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'n' [-Wacc-implicit-scalar]
  do i = 1, n
    !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'factor' [-Wacc-implicit-scalar]
    x(i) = x(i) * factor
  end do
  !$acc end parallel loop
end subroutine

! Non-scalar types always require an explicit data clause even with the flag:
! character, derived type, allocatable, and pointer variables are excluded
! from the pre-3.2 scalar extension.
subroutine default_none_non_scalars()
  character(10) :: c
  type :: t
    integer :: i
  end type
  type(t) :: x
  integer, allocatable :: a
  integer, pointer :: p
  !$acc parallel default(none)
  !ERROR: The DEFAULT(NONE) clause requires that 'c' must be listed in a data-mapping clause
  c = "hi"
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  x%i = 1
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
  a = 1
  !ERROR: The DEFAULT(NONE) clause requires that 'p' must be listed in a data-mapping clause
  p = 1
  !$acc end parallel
end subroutine

! Outer serial loop variable and scalars assigned in the outer loop body
! are read inside a nested ACC parallel loop.
subroutine default_none_outer_loop_scalars(x, n, m)
  implicit none
  real :: x(n), thresh, coef, val
  integer :: n, m, k, i
  do k = 1, m
    coef   = real(m) / real(k)
    thresh = real(k) - real(k) / real(m)
    val    = real(k) * 4.0
    !$acc parallel loop default(none) present(x)
    !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'n' [-Wacc-implicit-scalar]
    do i = 1, n
      !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'thresh' [-Wacc-implicit-scalar]
      !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'coef' [-Wacc-implicit-scalar]
      !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'val' [-Wacc-implicit-scalar]
      !WARNING: Implicit attribute inferred for DEFAULT(NONE) scalar 'k' [-Wacc-implicit-scalar]
      if (x(i) < thresh) x(i) = x(i) + coef * val * real(k)
    end do
    !$acc end parallel loop
  end do
end subroutine
