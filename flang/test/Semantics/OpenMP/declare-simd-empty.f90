! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Test the source code starting with omp syntax

!$omp declare simd
integer :: x
end
