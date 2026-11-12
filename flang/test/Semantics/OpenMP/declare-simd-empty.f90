! RUN: %python %S/../test_errors.py %s %flang -fopenmp -Werror -Wno-experimental-option
! Test the source code starting with omp syntax

!WARNING: DECLARE SIMD directive is not implemented yet and will be ignored
!$omp declare simd
integer :: x
end
