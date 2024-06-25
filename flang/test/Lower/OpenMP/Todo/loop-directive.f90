! This test checks lowering of OpenMP loop Directive.

! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled directive loop
subroutine test_loop()
  integer :: i, j = 1
  !$omp loop
  do i=1,10
   j = j + 1
  end do
  !$omp end loop
end subroutine

