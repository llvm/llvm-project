! RUN: %python %S/../test_errors.py %s %flang -fopenmp

!$omp requires reverse_offload
end
