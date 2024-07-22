! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.0
! 2.4 Requires directive
! All atomic_default_mem_order clauses in 'requires' directives must come
! strictly before any atomic directives on which the memory_order clause is not
! specified.

subroutine f
  integer :: a = 0
  !$omp atomic
  a = a + 1
end subroutine f

subroutine g
  !ERROR: REQUIRES directive with 'ATOMIC_DEFAULT_MEM_ORDER' clause found lexically after atomic operation without a memory order clause
  !$omp requires atomic_default_mem_order(relaxed)
end subroutine g
