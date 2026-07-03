! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
subroutine intrinsic_reduction
  type local
     integer alpha
  end type local
  type(local) a
!ERROR: A variable that is part of another variable cannot appear on the REDUCTION clause
!$omp parallel reduction(+:a%alpha)
!$omp end parallel
end subroutine intrinsic_reduction
