!RUN: %flang_fc1 -fopenmp -fopenmp-version=60 -fdebug-dump-symbols %s | FileCheck %s

module fred
!$omp requires atomic_default_mem_order(relaxed)
contains
subroutine f00
  !$omp requires unified_address
end
subroutine f01
  !$omp requires unified_shared_memory
end
end module

!CHECK: fred: Module OmpRequirements:(atomic_default_mem_order(relaxed),unified_address,unified_shared_memory)
