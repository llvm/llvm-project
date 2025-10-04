!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: OMPD_ordered
subroutine f00(x)
  integer :: a(10)

  do i = 1, 10
    !$omp do ordered(3)
    do j = 1, 10
      do k = 1, 10
        do m = 1, 10
          !$omp ordered doacross(sink: m+1, k+0, j-2)
          a(i) = i
        enddo
      enddo
    enddo
    !$omp end do
  enddo
end
