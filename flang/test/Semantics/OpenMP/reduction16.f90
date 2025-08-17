!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50

!Ref: [5.0:298:19]
! A type parameter inquiry cannot appear in a reduction clause.

subroutine f00
  integer :: x
!ERROR: Type parameter inquiry is not permitted in REDUCTION clause
!$omp do reduction (+ : x%kind)
  do i = 1, 10
  end do
!$omp end do
end subroutine

