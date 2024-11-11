! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
  ! Test the source code starting with omp syntax

integer :: y

!ERROR: Type is not a derived type
!$omp declare mapper(mm : integer::x) map(x, y)
end
