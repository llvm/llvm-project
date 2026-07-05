! Tests reduction processor behavior when a reduction symbol is not supported.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine foo
  implicit none
  integer :: k, i

  interface max
    function max(m1,m2)
      integer :: m1, m2
    end function
  end interface

  !CHECK: not yet implemented: Lowering unrecognised reduction type
  !$omp do reduction (max: k)
  do i=1,10
  end do
  !$omp end do
end
