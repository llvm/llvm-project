! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP ALLOCATE directive in unsupported declaration scope
module omp_allocate_module
  implicit none
  integer :: x
  !$omp allocate(x)
end module omp_allocate_module
