! RUN: rm -rf %t && mkdir %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -module-dir %t '%S/../Inputs/requires_module.f90'
! RUN: %python %S/../test_errors.py %s %flang -fopenmp -module-dir %t
! OpenMP Version 5.0
! 2.4 Requires directive
! All atomic_default_mem_order clauses in REQUIRES directives found within a
! compilation unit must specify the same ordering. Test that this is propagated
! from imported modules

!ERROR: Conflicting 'ATOMIC_DEFAULT_MEM_ORDER' REQUIRES clauses found in compilation unit
use requires_module
!$omp requires atomic_default_mem_order(relaxed)

end program
