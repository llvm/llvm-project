! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! Test the source code starting with omp syntax

type, abstract :: t1
   integer :: y
end type t1

!ERROR: ABSTRACT derived type may not be used here
!$omp declare mapper(mm : t1::x) map(x, x%y)
end
