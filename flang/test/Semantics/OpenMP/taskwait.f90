! RUN: %python %S/../test_errors.py %s %flang -fopenmp

!$omp taskwait
end
