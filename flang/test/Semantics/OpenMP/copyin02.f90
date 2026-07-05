! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.15.4.1 copyin Clause
! A common block name that appears in a copyin clause must be declared to be
! a common block in the same scoping unit in which the copyin clause appears.

subroutine copyin()
  integer :: a = 10
  common /cmn/ a

  !$omp threadprivate(/cmn/)
  call copyin_clause()

  contains

    subroutine copyin_clause()
      !ERROR: COMMON block must be declared in the same scoping unit in which the OpenMP directive or clause appears
      !$omp parallel copyin(/cmn/)
      print *, a
      !$omp end parallel
    end subroutine copyin_clause

end subroutine copyin
