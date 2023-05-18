! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction of some types is not supported
subroutine reduction_allocatable
  integer, allocatable :: x
  integer :: i = 1

  allocate(x)
  x = 0

  !$omp parallel num_threads(4)
  !$omp do reduction(+:x)
  do i = 1, 10
    x = x + i
  enddo
  !$omp end do
  !$omp end parallel

  print *, x
end subroutine
