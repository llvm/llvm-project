! RUN: %python %S/../test_errors.py %s %flang -fopenacc -facc-allow-default-none-scalars -Wacc-implicit-scalar

! With -facc-allow-default-none-scalars, scalar variables without explicit
! data clauses under DEFAULT(NONE) are allowed and get a warning
! (-Wacc-implicit-scalar) instead of an error.  Arrays continue to error.

subroutine default_none_scalars_extension()
  integer :: a
  integer :: b(10)
  !$acc parallel default(none)
  !WARNING: 'a' has no data clause under DEFAULT(NONE); implicit attribute inferred (-facc-allow-default-none-scalars enables pre-OpenACC-3.2 behavior) [-Wacc-implicit-scalar]
  a = 1
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-mapping clause
  b(1) = 2
  !$acc end parallel
end subroutine
