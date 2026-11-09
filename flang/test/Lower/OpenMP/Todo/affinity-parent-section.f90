! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/folded.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/parent-iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unused.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/ordinary.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/nested.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/intermediate.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/whole-iterator.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/whole-ordinary.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: array-valued parent in AFFINITY clause

!--- iterator.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  !$omp task affinity(iterator(i=1:2): a(1:2)%field(i))
  !$omp end task
end

!--- folded.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  !$omp task affinity(iterator(i=1:2): a(1:2)%field(1+0*i))
  !$omp end task
end

!--- parent-iterator.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  !$omp task affinity(iterator(i=1:2): a(1:i)%field(1))
  !$omp end task
end

!--- unused.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  !$omp task affinity(iterator(i=1:0): a(1:2)%field(1))
  !$omp end task
end

!--- ordinary.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  !$omp task affinity(a(1:2)%field(1))
  !$omp end task
end

!--- nested.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type outer
    type(t) :: nested
  end type
  type(outer) :: a(8)
  !$omp task affinity(iterator(i=1:2): a(1:2)%nested%field(i))
  !$omp end task
end

!--- intermediate.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type outer
    type(t) :: nested(8)
  end type
  type(outer) :: a
  !$omp task affinity(iterator(i=1:2): a%nested(1:2)%field(i))
  !$omp end task
end

!--- whole-iterator.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  !$omp task affinity(iterator(i=1:2): a%field(i))
  !$omp end task
end

!--- whole-ordinary.f90
subroutine s(a)
  type t
    integer :: field(4)
  end type
  type(t) :: a(8)
  !$omp task affinity(a%field(1))
  !$omp end task
end
