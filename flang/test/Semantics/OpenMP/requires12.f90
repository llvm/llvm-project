! RUN: rm -rf %t && mkdir %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -module-dir %t '%S/../Inputs/requires_module.f90'
! RUN: %python %S/../test_errors.py %s %flang -fopenmp -module-dir %t
! OpenMP Version 5.0
! 2.4 Requires directive
! atomic_default_mem_order clauses in REQUIRES directives must come strictly
! before any atomic constructs with no explicit memory order set. Test that this
! is propagated from imported modules.

subroutine f
  integer :: a = 0
  !$omp atomic
  a = a + 1
end subroutine f

program requires
  !ERROR: 'requires_module' module containing 'ATOMIC_DEFAULT_MEM_ORDER' REQUIRES clause imported lexically after atomic operation without a memory order clause
  use requires_module
end program requires
