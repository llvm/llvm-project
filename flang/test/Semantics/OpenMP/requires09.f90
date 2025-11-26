! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.0
! 2.4 Requires directive
! All atomic_default_mem_order clauses in 'requires' directives found within a
! compilation unit must specify the same ordering.
!ERROR: Conflicting 'ATOMIC_DEFAULT_MEM_ORDER' REQUIRES clauses found in compilation unit
module m
contains

subroutine f
  !$omp requires atomic_default_mem_order(seq_cst)
end subroutine f

subroutine g
  !$omp requires atomic_default_mem_order(relaxed)
end subroutine g

end module
