! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: lastprivate clause with CONDITIONAL modifier
subroutine foo()
  integer :: x, i
  x = 1
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, 100
    x = x + 1
  enddo
end
