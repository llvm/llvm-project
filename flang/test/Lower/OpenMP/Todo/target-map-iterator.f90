! RUN: rm -rf %t && split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/map-clause.f90 2>&1 | \
! RUN:   FileCheck %t/map-clause.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/explicit-mapper.f90 2>&1 | \
! RUN:   FileCheck %t/explicit-mapper.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/default-mapper.f90 2>&1 | \
! RUN:   FileCheck %t/default-mapper.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/nested-mapper.f90 2>&1 | \
! RUN:   FileCheck %t/nested-mapper.f90

!--- map-clause.f90
subroutine target_map_iterator()
  integer :: a(8)
  integer :: i

  ! CHECK: not yet implemented: TARGET construct with MAP iterator modifier
  !$omp target map(iterator(i = 1:8), to: a(i))
  a(1) = 42
  !$omp end target
end subroutine

!--- explicit-mapper.f90
subroutine target_explicit_iterator_mapper()
  type :: t
    integer :: a(2)
  end type
  type(t) :: x
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:2), to: v%a(i))

  ! CHECK: not yet implemented: TARGET construct using a declare mapper with iterator modifier
  !$omp target map(mapper(m), tofrom: x)
  x%a(1) = 1
  !$omp end target
end subroutine

!--- default-mapper.f90
subroutine target_default_iterator_mapper()
  type :: t
    integer :: a(2)
  end type
  type(t) :: x
  !$omp declare mapper(t :: v) map(iterator(i = 1:2), to: v%a(i))

  ! CHECK: not yet implemented: TARGET construct using a declare mapper with iterator modifier
  !$omp target
  x%a(1) = 1
  !$omp end target
end subroutine

!--- nested-mapper.f90
subroutine target_nested_iterator_mapper()
  type :: inner
    integer :: a(2)
  end type
  type :: outer
    type(inner) :: in
  end type
  type(outer) :: x
  !$omp declare mapper(mi: inner :: v) map(iterator(i = 1:2), to: v%a(i))
  !$omp declare mapper(mo: outer :: w) map(mapper(mi), tofrom: w%in)

  ! CHECK: not yet implemented: TARGET construct using a declare mapper with iterator modifier
  !$omp target map(mapper(mo), tofrom: x)
  x%in%a(1) = 1
  !$omp end target
end subroutine
