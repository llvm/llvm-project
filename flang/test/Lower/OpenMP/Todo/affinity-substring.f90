! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/parent-iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/parent-no-iterator.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: substring in AFFINITY clause

!--- iterator.f90
subroutine s(a)
  type t
    character(8) :: field
  end type
  type(t) :: a(8)
  !$omp task affinity(iterator(i = 1:2): a(i)%field(1:4))
  !$omp end task
end

!--- no-iterator.f90
subroutine s(a)
  type t
    character(8) :: field
  end type
  type(t) :: a(8)
  !$omp task affinity(a(1)%field(1:4))
  !$omp end task
end

!--- parent-iterator.f90
subroutine s(a)
  type t
    character(8) :: field
  end type
  type(t) :: a(8)
  !$omp task affinity(iterator(i = 1:2): a(i:i)%field(1:4))
  !$omp end task
end

!--- parent-no-iterator.f90
subroutine s(a)
  type t
    character(8) :: field
  end type
  type(t) :: a(8)
  !$omp task affinity(a(1:2)%field(1:4))
  !$omp end task
end
