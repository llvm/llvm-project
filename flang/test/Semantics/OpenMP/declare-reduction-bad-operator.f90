!RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50

function func(n)
!ERROR: Invalid reduction intrinsic operator in DECLARE REDUCTION directive
    !$omp declare reduction(/:integer:omp_out=omp_out+omp_in)
end function func
