! RUN: %S/test_errors.sh %s %t %flang -fopenmp

! 2.15.3 Data-Sharing Attribute Clauses
! 2.15.3.1 default Clause

subroutine default_none()
  integer a(3)

  A = 1
  B = 2
  !$omp parallel default(none) private(c)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing attribute clause
  A(1:2) = 3
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing attribute clause
  B = 4
  C = 5
  !$omp end parallel
end subroutine default_none

program mm
  call default_none()
  !TODO: private, firstprivate, shared
end
