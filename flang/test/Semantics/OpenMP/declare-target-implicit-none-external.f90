! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s 2>&1 | FileCheck %s

! Test that IMPLICIT NONE(EXTERNAL) blocks implicit procedure creation
! per OpenMP 5.2 3.2.1, 7.8 &7.8.1

program test_implicit_none_external
  implicit none(external)
  integer :: n = 10
  ! With IMPLICIT NONE(EXTERNAL), ext_sub should NOT be implicitly
  ! created as an external procedure - should get an error
  !$omp declare target(ext_sub)

  !$omp target
    call ext_sub(n)
  !$omp end target
end program

! CHECK: error:
