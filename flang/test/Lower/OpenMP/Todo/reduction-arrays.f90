! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction of some types is not supported
subroutine reduction_array(y)
  integer :: x(100), y(100,100)
  !$omp parallel
  !$omp do reduction(+:x)
  do i=1, 100
    x = x + y(:,i)
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine
