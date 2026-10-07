! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/iterator-2d.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-iterator.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: vector subscript in AFFINITY clause

!--- iterator.f90
subroutine s(a, m)
  integer :: a(8), m
  !$omp task affinity(iterator(i = 1:m): a([i]))
  !$omp end task
end

!--- iterator-2d.f90
subroutine s(a, m)
  integer :: a(2, 8), m
  !$omp task affinity(iterator(i = 1:m): a(1:2, [i]))
  !$omp end task
end

!--- no-iterator.f90
subroutine s(a)
  integer :: a(8)
  !$omp task affinity(a([1]))
  !$omp end task
end
