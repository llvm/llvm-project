!RUN: %python %S/../test_errors.py %s %flang -fopenmp

program main
  common /cmn/ k1
!$omp threadprivate(/cmn/)
contains
  subroutine ss1
  k1 = 1
!$omp parallel copyin (k1)
!$omp end parallel
  end subroutine ss1
end program main
