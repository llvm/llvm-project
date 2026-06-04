! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s 2>&1 | FileCheck %s

! Test OpenMP 5.2 , 3.2.1, 7.8 & 7.8.1: implicit external procedure in DECLARE TARGET
! When the name is later defined as a function, it should work correctly

program test_implicit_declare_target_func
  integer :: n = 10
  integer :: result
  ! ext_func should be implicitly treated as an external procedure
  ! Note: OMP 5.2 says "subroutine" but we test function here
  !$omp declare target(ext_func)

  !$omp target map(from: result)
    result = ext_func(n)
  !$omp end target
  print *, result
end program

function ext_func(x) result(y)
  !$omp declare target
  integer, intent(in) :: x
  integer :: y
  y = x * 2
end function

! CHECK-NOT: error:
