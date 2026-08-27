! RUN: %python %S/../test_errors.py %s %flang -fopenacc -fno-openacc-default-none-scalars-strict

! Derived-type component references in OpenACC clauses are accepted. Exact
! duplicate and conflicting component references in data-sharing clauses are
! diagnosed, but broader containment is deliberately limited for now:
! - component clauses satisfy DEFAULT(NONE) only for contained references;
! - bare whole-object/component conflicts are not diagnosed.

module component_ref_types
  implicit none
  type :: point_t
    real :: x
    real :: y
  end type
  type :: vec_t
    real :: arr(10)
    real :: scale
  end type
  type :: nested_t
    type(point_t) :: pt
    integer :: tag
  end type
end module

subroutine test_component_clauses_are_accepted()
  use component_ref_types, only: nested_t, point_t, vec_t
  type(point_t) :: p
  type(vec_t) :: v
  type(nested_t) :: n
  !$acc parallel copy(p%x, p%y, v%arr, v%arr(1:5), n%pt%x)
  p%x = 1.0
  p%y = 2.0
  v%arr(1) = p%x
  n%pt%x = v%arr(1)
  !$acc end parallel
end subroutine

subroutine test_default_none_component_covers_same_component()
  use component_ref_types, only: point_t
  type(point_t) :: p
  !$acc parallel default(none) copy(p%x)
  p%x = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_component_does_not_cover_sibling()
  use component_ref_types, only: point_t
  type(point_t) :: p
  !$acc parallel default(none) copy(p%x)
  !ERROR: The DEFAULT(NONE) clause requires that 'p' must be listed in a data-mapping clause
  p%y = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_component_covers_contained_component()
  use component_ref_types, only: nested_t
  type(nested_t) :: n
  !$acc parallel default(none) copy(n%pt)
  n%pt%x = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_component_does_not_cover_parent()
  use component_ref_types, only: nested_t, point_t
  type(nested_t) :: n
  type(point_t) :: p
  !$acc parallel default(none) copy(n%pt%x, p)
  !ERROR: The DEFAULT(NONE) clause requires that 'n' must be listed in a data-mapping clause
  n%pt = p
  !$acc end parallel
end subroutine

subroutine test_default_none_whole_object_covers_components()
  use component_ref_types, only: point_t
  type(point_t) :: p, q
  !$acc parallel default(none) copy(p, q)
  p%x = q%y
  p = q
  !$acc end parallel
end subroutine

subroutine test_default_none_unlisted_component_object()
  use component_ref_types, only: point_t
  type(point_t) :: p, q
  !$acc parallel copy(p%x)
  p%x = 1.0
  q%y = 2.0
  !$acc end parallel
end subroutine

subroutine test_same_object_same_dsa_components()
  use component_ref_types, only: point_t
  type(point_t) :: p
  integer :: i
  !WARNING: 'p%x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel loop private(p%x, p%y, p%x)
  do i = 1, 10
    p%x = real(i)
    p%y = p%x
  end do
  !$acc end parallel loop
end subroutine

subroutine test_same_object_incompatible_same_component()
  use component_ref_types, only: point_t
  type(point_t) :: p
  integer :: i
  !ERROR: 'p%x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(p%x) firstprivate(p%x)
  do i = 1, 10
    p%x = real(i)
  end do
  !$acc end parallel loop
end subroutine

subroutine test_same_object_incompatible_different_components()
  use component_ref_types, only: point_t
  type(point_t) :: p
  integer :: i
  !$acc parallel loop private(p%x) firstprivate(p%y)
  do i = 1, 10
    p%x = real(i)
    p%y = p%x
  end do
  !$acc end parallel loop
end subroutine

subroutine test_contained_component_same_dsa()
  use component_ref_types, only: nested_t
  type(nested_t) :: n
  integer :: i
  !ERROR: 'n%pt%x' overlaps another object in the same kind of data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(n%pt, n%pt%x)
  do i = 1, 10
    n%pt%x = real(i)
  end do
  !$acc end parallel loop
end subroutine

subroutine test_contained_component_same_dsa_child_first()
  use component_ref_types, only: nested_t
  type(nested_t) :: n
  integer :: i
  !ERROR: 'n%pt' overlaps another object in the same kind of data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(n%pt%x, n%pt)
  do i = 1, 10
    n%pt%x = real(i)
  end do
  !$acc end parallel loop
end subroutine

subroutine test_contained_component_incompatible_parent_first()
  use component_ref_types, only: nested_t
  type(nested_t) :: n
  integer :: i
  !ERROR: 'n%pt%x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(n%pt) firstprivate(n%pt%x)
  do i = 1, 10
    n%pt%x = real(i)
  end do
  !$acc end parallel loop
end subroutine

subroutine test_contained_component_incompatible_child_first()
  use component_ref_types, only: nested_t
  type(nested_t) :: n
  integer :: i
  !ERROR: 'n%pt' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(n%pt%x) firstprivate(n%pt)
  do i = 1, 10
    n%pt%x = real(i)
  end do
  !$acc end parallel loop
end subroutine

subroutine test_whole_object_incompatible_with_component()
  use component_ref_types, only: point_t
  type(point_t) :: p
  integer :: i
  !ERROR: 'p%x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(p) firstprivate(p%x)
  do i = 1, 10
    p%x = real(i)
  end do
  !$acc end parallel loop
end subroutine

subroutine test_distinct_objects_same_type_same_component()
  use component_ref_types, only: point_t
  type(point_t) :: p, q
  integer :: i
  !$acc parallel loop private(p%x) firstprivate(q%x)
  do i = 1, 10
    p%x = q%x + real(i)
  end do
  !$acc end parallel loop
end subroutine

subroutine test_indexed_component_refs_are_not_conflicts()
  use component_ref_types, only: vec_t
  type(vec_t) :: v
  integer :: i
  !$acc parallel loop private(v%arr(1:5)) firstprivate(v%arr(6:10))
  do i = 1, 10
    v%arr(i) = real(i)
  end do
  !$acc end parallel loop
end subroutine
