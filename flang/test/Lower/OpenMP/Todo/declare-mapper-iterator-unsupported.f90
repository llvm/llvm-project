! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/coarray.f90 2>&1 | \
! RUN:   FileCheck %t/coarray.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/external-array.f90 2>&1 | \
! RUN:   FileCheck %t/external-array.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/external-member.f90 2>&1 | \
! RUN:   FileCheck %t/external-member.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/object-type.f90 2>&1 | \
! RUN:   FileCheck %t/object-type.f90

! Unsupported iterator locators inside declare mapper.

!--- coarray.f90
! CHECK: not yet implemented: iterator modifier with locator outside
! CHECK-SAME: declare mapper variable

module declare_mapper_iterator_coarray
  integer, save :: a(10)[*]
  type :: t
    integer :: x
  end type

  !$omp declare mapper(mm: t :: v) map(iterator(i = 1:10): a(i)[1])
end module

!--- external-array.f90
! CHECK: not yet implemented: iterator modifier with locator outside
! CHECK-SAME: declare mapper variable

module declare_mapper_external_array
  integer :: tbl(100)
  type :: t
    real :: a(100)
  end type

  !$omp declare mapper(mm: t :: v) &
  !$omp& map(iterator(i = 1:100): v%a(i), tbl(i))
end module

!--- external-member.f90
! CHECK: not yet implemented: iterator modifier with derived type member map

module declare_mapper_external_member
  type :: t
    integer :: a(10)
  end type
  type(t) :: w

  !$omp declare mapper(m: t :: v) map(iterator(i = 1:10): w%a(i))
end module

!--- object-type.f90
! CHECK: not yet implemented: object type not supported by iterator modifier

module declare_mapper_iterator_object_type
  type :: t
    complex :: c(10)
  end type

  !$omp declare mapper(mm: t :: v) map(iterator(i = 1:10): v%c(i)%re)
end module
