module requires_module
  !$omp requires atomic_default_mem_order(seq_cst), unified_shared_memory
end module
