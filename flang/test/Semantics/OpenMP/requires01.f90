! RUN: %python %S/../test_errors.py %s %flang -fopenmp

!WARNING: REVERSE_OFFLOAD clause is not supported and will be ignored
!$omp requires reverse_offload unified_shared_memory

!ERROR: NOWAIT clause is not allowed on REQUIRES directive
!$omp requires nowait
end
