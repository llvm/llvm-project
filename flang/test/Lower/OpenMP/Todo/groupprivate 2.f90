!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: GROUPPRIVATE

module m
implicit none
integer :: x
!$omp groupprivate(x)
end module
