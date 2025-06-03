! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMPDispatchConstruct
program p
  integer r
  r = 1
!$omp dispatch nowait
  call foo()
contains
  subroutine foo
  end subroutine
end program p
