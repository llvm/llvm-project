! RUN: %S/../test_errors.sh %s %t %flang -fopenacc

! Data-Mapping Attribute Clauses
! 2.15.14 default Clause

subroutine default_none()
  integer a(3)

  A = 1
  B = 2
  !$acc parallel default(none) private(c)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
  A(1:2) = 3
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-mapping clause
  B = 4
  C = 5
  !$acc end parallel
end subroutine default_none

program mm
  call default_none()
end
