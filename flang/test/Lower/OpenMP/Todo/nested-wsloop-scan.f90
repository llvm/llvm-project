! Tests scan reduction behavior when used in nested workshare loops

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program nested_scan_example
  implicit none
  integer, parameter :: n = 4, m = 5
  integer :: a(n, m), b(n, m)
  integer :: i, j
  integer :: row_sum, col_sum

  do i = 1, n
     do j = 1, m
        a(i, j) = i + j
     end do
  end do

  !$omp parallel do reduction(inscan, +: row_sum) private(col_sum, j)
  do i = 1, n
     row_sum = row_sum + i
     !$omp scan inclusive(row_sum)

     col_sum = 0
     !$omp parallel do reduction(inscan, +: col_sum)
     do j = 1, m
        col_sum = col_sum + a(i, j)
        !CHECK: not yet implemented: Scan directive inside nested workshare loops
        !$omp scan inclusive(col_sum)
        b(i, j) = col_sum + row_sum 
     end do
     !$omp end parallel do
  end do
  !$omp end parallel do
end program nested_scan_example
