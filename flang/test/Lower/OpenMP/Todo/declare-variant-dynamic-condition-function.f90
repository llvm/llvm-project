! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! A run-time DECLARE VARIANT user condition on a function is not yet supported;
! only subroutines are handled (the if/else cascade cannot yet carry a result).

! CHECK: not yet implemented: dynamic user condition on a function in DECLARE VARIANT

module m
contains
  integer function fbase(x)
    integer :: x
    !$omp declare variant (fvar) match (user={condition(x > 0)})
    fbase = 1
  end function
  integer function fvar(x)
    integer :: x
    fvar = 2
  end function
end module

subroutine caller(x)
  use m
  integer :: x, r
  r = fbase(x)
end subroutine
