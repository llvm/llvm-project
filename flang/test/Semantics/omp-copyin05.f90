! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! OpenMP Version 4.5
! 2.15.4.1 copyin Clause
! A common block name that appears in a copyin clause must be declared to be
! a common block in the same scoping unit in which the copyin clause appears.

subroutine copyin()
  call copyin_clause()

  contains

    subroutine copyin_clause()
      integer :: a = 20
      common /cmn/ a

      !$omp threadprivate(/cmn/)

      !$omp parallel copyin(/cmn/)
      print *, a
      !$omp end parallel
    end subroutine copyin_clause

end subroutine copyin
