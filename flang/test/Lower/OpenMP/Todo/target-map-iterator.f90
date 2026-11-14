! RUN: rm -rf %t && split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/map-clause.f90 2>&1 | \
! RUN:   FileCheck %t/map-clause.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   -o - %t/component-mapper.f90 2>&1 | FileCheck %t/component-mapper.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   -o - %t/component-mapper-implicit.f90 2>&1 | \
! RUN:   FileCheck %t/component-mapper-implicit.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   -o - %t/component-mapper-enter.f90 2>&1 | \
! RUN:   FileCheck %t/component-mapper-enter.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   -o - %t/recursive-iterated.f90 2>&1 | \
! RUN:   FileCheck %t/recursive-iterated.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   -o - %t/recursive-later-clause.f90 2>&1 | \
! RUN:   FileCheck %t/recursive-later-clause.f90

!--- map-clause.f90
subroutine target_map_iterator()
  integer :: a(8)
  integer :: i

  ! CHECK: not yet implemented: TARGET construct with MAP iterator modifier
  !$omp target map(iterator(i = 1:8), to: a(i))
  a(1) = 42
  !$omp end target
end subroutine

!--- component-mapper.f90
! The implicit default mapper for outer would not use the mapper of inner.
subroutine target_component_iterator_mapper()
  type :: inner
    integer :: a(2)
    integer, allocatable :: extra(:)
  end type
  type :: outer
    type(inner) :: in
  end type
  type(outer) :: x
  !$omp declare mapper(inner :: v) map(iterator(i = 1:2), to: v%a(i))

  ! CHECK: not yet implemented: implicit default mapper for a type with a component whose default mapper has an iterator modifier
  !$omp target map(tofrom: x)
  x%in%a(1) = 1
  !$omp end target
end subroutine

!--- component-mapper-implicit.f90
subroutine target_component_iterator_mapper_implicit()
  type :: inner
    integer :: a(2)
    integer, allocatable :: extra(:)
  end type
  type :: outer
    type(inner) :: in
  end type
  type(outer) :: x
  !$omp declare mapper(inner :: v) map(iterator(i = 1:2), to: v%a(i))

  ! CHECK: not yet implemented: implicit default mapper for a type with a component whose default mapper has an iterator modifier
  !$omp target
  x%in%a(1) = 1
  !$omp end target
end subroutine

!--- component-mapper-enter.f90
! TARGET ENTER DATA maps outer without a mapper, skipping the mapper of inner.
subroutine target_enter_data_component_iterator_mapper()
  type :: inner
    integer :: a(2)
    integer, allocatable :: extra(:)
  end type
  type :: outer
    type(inner) :: in
  end type
  type(outer) :: x
  !$omp declare mapper(inner :: v) map(iterator(i = 1:2), to: v%a(i))

  ! CHECK: not yet implemented: implicit default mapper for a type with a component whose default mapper has an iterator modifier
  !$omp target enter data map(to: x)
end subroutine

!--- recursive-iterated.f90
! The iterated component is allocatable, which the mapper emitted in host
! offload compilation does not support yet; this is diagnosed before the
! implicit mapper for holder is considered.
subroutine recursive_iterated_component()
  type :: inner
    integer :: a(2)
    type(holder), allocatable :: h(:)
  end type
  type :: holder
    type(inner), allocatable :: in
  end type

  ! CHECK: not yet implemented: iterator modifier with ALLOCATABLE or POINTER component in declare mapper
  !$omp declare mapper(inner :: v) map(iterator(i = 1:2), to: v%h(i))
end subroutine

!--- recursive-later-clause.f90
subroutine recursive_later_clause_component()
  type :: inner
    integer :: a(2)
    type(holder), allocatable :: h(:)
  end type
  type :: holder
    type(inner), allocatable :: in
  end type

  ! CHECK: not yet implemented: implicit default mapper for a type with a component whose default mapper has an iterator modifier
  !$omp declare mapper(inner :: v) map(iterator(i = 1:2), to: v%a(i)) map(to: v%h)
end subroutine
