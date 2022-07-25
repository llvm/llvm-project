! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction of some intrinsic operators is not supported
subroutine reduction_multiply
  integer :: x
  !$omp parallel
  !$omp do reduction(*:x)
  do i=1, 100
    x = x * i
  end do
  !$omp end do
  !$omp end parallel
  print *, x
end subroutine
