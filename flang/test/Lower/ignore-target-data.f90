!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

!Make sure that there are no calls to the mapper.

!CHECK-NOT: call{{.*}}__tgt_target_data_begin_mapper
!CHECK-NOT: call{{.*}}__tgt_target_data_end_mapper

program test

call f(1, 2)

contains

subroutine f(x, y)
  integer :: x, y
  !$omp target data map(tofrom: x, y)
  x = x + y
  !$omp end target data
end subroutine
end
