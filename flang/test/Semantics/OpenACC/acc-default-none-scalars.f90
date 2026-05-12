! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Without -facc-allow-default-none-scalars, DEFAULT(NONE) errors on both
! scalars and arrays per OpenACC v3.2 section 2.5.14.

subroutine default_none_scalars_error()
  integer :: a
  integer :: b(10)
  !$acc parallel default(none)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
  a = 1
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-mapping clause
  b(1) = 2
  !$acc end parallel
end subroutine
