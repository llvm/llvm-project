! Tests scan reduction when a BLOCK construct is nested between the DO loop and
! the scan directive. A BLOCK brackets its body with stacksave/stackrestore,
! which the scan loop-splitting would tear across the input- and scan-phase
! loops, so this is rejected for now.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program block_scan_example
  implicit none
  integer, parameter :: n = 8
  integer :: i, x
  integer :: a(n)

  x = 0
  !$omp parallel do reduction(inscan, +: x)
  do i = 1, n
     block
        x = x + i
        !CHECK: not yet implemented: Scan directive nested inside a BLOCK construct
        !$omp scan inclusive(x)
        a(i) = x
     end block
  end do
end program block_scan_example
