! RUN: %python %S/../test_errors.py %s %flang -fopenmp

!$omp requires reverse_offload unified_shared_memory

!ERROR: NOWAIT clause is not allowed on the REQUIRES directive
!$omp requires nowait
end
