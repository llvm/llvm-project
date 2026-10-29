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
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-parent.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-parent.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-section-member.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-section-member.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-substring.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-substring.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-complex-part.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-complex-part.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fcoarray \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-coindexed.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-coindexed.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-vector.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-vector.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-stride.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-stride.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/unreferenced-empty.f90 2>&1 | \
! RUN:   FileCheck %t/unreferenced-empty.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/later-clause-complex-part.f90 2>&1 | \
! RUN:   FileCheck %t/later-clause-complex-part.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/earlier-clause-vector.f90 2>&1 | \
! RUN:   FileCheck %t/earlier-clause-vector.f90
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %t/earlier-map-clause-vector.f90 2>&1 | \
! RUN:   FileCheck %t/earlier-map-clause-vector.f90

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

!--- unreferenced-parent.f90
subroutine target_update_unreferenced_array_parent()
  type :: s
    integer :: a(2)
  end type
  type(s) :: x(2)

  ! CHECK: not yet implemented: iterator modifier with array-valued parent
  !$omp target update to(iterator(i = 1:1): x(1:2)%a(1))
end subroutine

!--- unreferenced-section-member.f90
subroutine target_exit_data_unreferenced_section_member()
  type :: s
    integer :: c
  end type
  type(s) :: x(4)

  ! CHECK: not yet implemented: iterator modifier with array-valued parent
  !$omp target exit data map(iterator(i = 1:1), from: x(1:2)%c)
end subroutine

!--- unreferenced-substring.f90
subroutine target_enter_data_unreferenced_substring()
  character(len=4) :: s

  ! CHECK: not yet implemented: object type not supported by iterator modifier
  !$omp target enter data map(iterator(i = 1:1), to: s(1:2))
end subroutine

!--- unreferenced-complex-part.f90
subroutine target_enter_data_unreferenced_complex_part()
  complex :: z

  ! CHECK: not yet implemented: object type not supported by iterator modifier
  !$omp target enter data map(iterator(i = 1:1), to: z%re)
end subroutine

!--- unreferenced-coindexed.f90
subroutine target_update_unreferenced_coindexed(a)
  integer :: a(2)
  integer, save :: ca(2)[*]

  ! CHECK: not yet implemented: object type not supported by iterator modifier
  !$omp target update to(iterator(i = 1:2): a(i), ca(1)[1])
end subroutine

!--- unreferenced-vector.f90
subroutine target_exit_data_unreferenced_vector(a)
  integer :: a(4)

  ! CHECK: not yet implemented: iterator modifier with unsupported array section
  !$omp target exit data map(iterator(i = 1:1), from: a([1, 3]))
end subroutine

!--- unreferenced-stride.f90
subroutine target_update_unreferenced_stride(a)
  integer :: a(4)

  ! CHECK: not yet implemented: iterator modifier with unsupported array section
  !$omp target update from(iterator(i = 1:1): a(1:4:2))
end subroutine

!--- unreferenced-empty.f90
subroutine target_exit_data_unreferenced_empty(a)
  integer :: a(4)

  ! CHECK: not yet implemented: iterator modifier with unsupported array section
  !$omp target exit data map(iterator(i = 1:1), from: a(1:0))
end subroutine

!--- later-clause-complex-part.f90
subroutine target_enter_data_later_clause_complex_part(a, z)
  integer :: a(4)
  complex :: z

  ! CHECK: not yet implemented: object type not supported by iterator modifier
  !$omp target enter data map(iterator(i = 1:2), to: a(i)) map(to: z%re)
end subroutine

!--- earlier-clause-vector.f90
! TO clauses are processed before FROM clauses.
subroutine target_update_earlier_clause_vector(a, b)
  integer :: a(4), b(4)

  ! CHECK: not yet implemented: iterator modifier with unsupported array section
  !$omp target update from(iterator(i = 1:2): a(i)) to(b([1, 3]))
end subroutine

!--- earlier-map-clause-vector.f90
subroutine target_enter_data_earlier_clause_vector(a, b)
  integer :: a(4), b(4)

  ! CHECK: not yet implemented: iterator modifier with unsupported array section
  !$omp target enter data map(to: b([1, 3])) map(iterator(i = 1:2), to: a(i))
end subroutine
