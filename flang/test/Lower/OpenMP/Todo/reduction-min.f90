! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction of intrinsic procedures is not supported
subroutine reduction_min(y)
  integer :: x, y(:)
  x = 0
  !$omp parallel
  !$omp do reduction(min:x)
  do i=1, 100
    x = min(x, y(i))
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine
