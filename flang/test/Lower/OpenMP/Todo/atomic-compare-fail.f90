! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP atomic compare
program p
  integer :: x
  logical :: r
  !$omp atomic compare fail(relaxed)
  if (x .eq. 0) then
     x = 2
  end if
end program p
