! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s 2>&1 | FileCheck %s

! Test OpenMP 5.2 3.2.1, 7.8 & 7.8.1: implicit external procedure in DECLARE TARGET
! A name in DECLARE TARGET with no explicit type is treated as external subroutine

program test_implicit_declare_target
  integer :: n = 10
  ! ext_sub should be implicitly treated as an external subroutine
  !$omp declare target(ext_sub)

  !$omp target
    call ext_sub(n)
  !$omp end target
end program

subroutine ext_sub(x)
  !$omp declare target
  integer, intent(in) :: x
  print *, "Called with:", x
end subroutine

! CHECK-NOT: error:
