! RUN: not %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s 2>&1 | FileCheck %s

function func(n)
    !$omp declare reduction(/:integer:omp_out=omp_out+omp_in)
!CHECK: error: Unsupported operator in DECLARE REDUCTION
end function func
