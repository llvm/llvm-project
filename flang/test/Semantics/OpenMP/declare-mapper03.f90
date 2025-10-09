! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! Test the declare mapper construct with two default mappers.

type :: t1
   integer :: y
end type t1

!error: 'default' is already declared in this scoping unit

!$omp declare mapper(t1::x) map(x, x%y)
!$omp declare mapper(t1::x) map(x)
end
