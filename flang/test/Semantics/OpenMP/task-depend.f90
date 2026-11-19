! RUN: %python %S/../test_errors.py %s %flang -fopenmp

program test
! ERROR: One of dependence-type or task-dependence-type modifiers is required on DEPEND clause
! ERROR: A DEPEND clause on a TASK construct must have a valid task dependence type
!$omp task depend(ii)
!$omp end task
end

