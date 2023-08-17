! RUN: rm -rf %t && mkdir %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -module-dir %t '%S/../Inputs/requires_module.f90'
! RUN: %python %S/../test_errors.py %s %flang -fopenmp -module-dir %t
! OpenMP Version 5.0
! 2.4 Requires directive
! Target-related clauses in REQUIRES directives must come strictly before any
! device constructs, such as declare target with extended list. Test that this
! is propagated from imported modules.

subroutine f
  !$omp declare target (f)
end subroutine f

program requires
  !ERROR: 'requires_module' module containing device-related REQUIRES directive imported lexically after device construct
  use requires_module
end program requires
