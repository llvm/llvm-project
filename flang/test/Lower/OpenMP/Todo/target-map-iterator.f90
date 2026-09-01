! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

subroutine target_map_iterator()
  integer :: a(8)
  integer :: i

  ! CHECK: not yet implemented: TARGET construct with MAP iterator modifier
  !$omp target map(iterator(i = 1:8), to: a(i))
  a(1) = 42
  !$omp end target
end subroutine
