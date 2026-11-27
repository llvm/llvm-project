! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

!WARNING: REVERSE_OFFLOAD clause is not supported and will be ignored
!$omp requires reverse_offload unified_shared_memory

!ERROR: One of ATOMIC_DEFAULT_MEM_ORDER, DYNAMIC_ALLOCATORS, REVERSE_OFFLOAD, UNIFIED_ADDRESS or UNIFIED_SHARED_MEMORY clauses is required on REQUIRES directive
!ERROR: NOWAIT clause is not allowed on REQUIRES directive
!$omp requires nowait
end
