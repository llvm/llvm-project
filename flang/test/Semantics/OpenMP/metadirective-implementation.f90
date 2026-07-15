!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! The IMPLEMENTATION trait set

subroutine f00
  !$omp metadirective &
!ERROR: Trait property should be a clause
  !$omp & when(implementation={atomic_default_mem_order(0)}: nothing)
end

subroutine f01
  !$omp metadirective &
!ERROR: ATOMIC_DEFAULT_MEM_ORDER trait requires a clause from the memory-order clause set
  !$omp & when(implementation={atomic_default_mem_order(nowait)}: nothing)
end

subroutine f02
  !$omp metadirective &
!ERROR: REQUIRES trait requires a clause from the requirement clause set
!ERROR: Invalid clause specification for SHARED
  !$omp & when(implementation={requires(shared)}: nothing)
end

subroutine f03
  !$omp metadirective &
!This is ok
  !$omp & when(implementation={ &
  !$omp &         atomic_default_mem_order(relaxed), &
  !$omp &         extension("foo"), &
  !$omp &         requires(unified_address),
  !$omp &         vendor(some_vendor) &
  !$omp &      }: nothing)
end
