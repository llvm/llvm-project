! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-iterator.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: complex part in AFFINITY clause

!--- iterator.f90
subroutine s(z)
  complex :: z(8)
  !$omp task affinity(iterator(i = 1:2): z(i)%re)
  !$omp end task
end

!--- no-iterator.f90
subroutine s(z)
  complex :: z(8)
  !$omp task affinity(z(1)%im)
  !$omp end task
end
