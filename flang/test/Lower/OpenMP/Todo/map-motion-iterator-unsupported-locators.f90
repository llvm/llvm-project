! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/array-member.f90 2>&1 | \
! RUN:   FileCheck %t/array-member.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/map-member.f90 2>&1 | \
! RUN:   FileCheck %t/map-member.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/motion-member.f90 2>&1 | \
! RUN:   FileCheck %t/motion-member.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/map-optional.f90 2>&1 | \
! RUN:   FileCheck %t/map-optional.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/motion-optional.f90 2>&1 | \
! RUN:   FileCheck %t/motion-optional.f90

! Unsupported iterator locators in map and motion clauses.

!--- array-member.f90
! CHECK: not yet implemented: iterator modifier with derived type member map

subroutine map_iterator_array_derived_member
  type :: t
    integer :: b
  end type
  type(t) :: x(10)
  integer :: i

  !$omp target data map(iterator(i = 1:10), tofrom: x(i)%b)
  !$omp end target data
end subroutine

!--- map-member.f90
subroutine target_data_derived_member_iterator()
  type :: s
    integer :: a(10)
  end type
  type(s) :: x
  integer :: i

  !CHECK: not yet implemented: iterator modifier with derived type member map
  !$omp target data map(iterator(i = 1:10), tofrom: x%a(i))
  !$omp end target data
end subroutine

!--- motion-member.f90
subroutine target_update_derived_member_iterator()
  type :: s
    integer :: a(10)
  end type
  type(s) :: x
  integer :: i

  !CHECK: not yet implemented: iterator modifier with derived type member map
  !$omp target update to(iterator(i = 1:10): x%a(i))
end subroutine

!--- map-optional.f90
subroutine target_enter_data_optional_iterator(a, n)
  integer, optional, intent(inout) :: a(:)
  integer, intent(in) :: n
  integer :: i

  ! CHECK: not yet implemented: iterator modifier with optional locator
  !$omp target enter data map(iterator(i = 1:n), to: a(i))
end subroutine

!--- motion-optional.f90
subroutine target_update_optional_iterator(a, n)
  integer, allocatable, optional, intent(inout) :: a(:)
  integer, intent(in) :: n
  integer :: i

  ! CHECK: not yet implemented: iterator modifier with optional locator
  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine
