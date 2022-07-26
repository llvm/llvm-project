! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction of some types is not supported
subroutine reduction_real
  real :: x
  x = 0.0
  !$omp parallel
  !$omp do reduction(+:x)
  do i=1, 100
    x = x + 1.0
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine
