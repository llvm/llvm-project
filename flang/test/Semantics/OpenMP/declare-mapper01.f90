! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! Test the declare mapper with non-derived type.

integer :: y

!ERROR: Type is not a derived type
!$omp declare mapper(mm : integer::x) map(x, y)
end
