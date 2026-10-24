! RUN: %python %S/../test_errors.py %s %flang -fopenacc -fno-openacc-default-none-scalars-strict -Wno-openacc-default-none-scalars-strict

! DEFAULT(NONE) must check a reference to a single element of an array
! component, such as x%b(i).  When OpenACC analysis runs, name resolution has
! not yet rewritten such a reference into an array element: it is still a
! function reference whose procedure designator is a component.  It must
! nevertheless be checked like the equivalent section x%b(i:i).

module component_element_types
  implicit none
  abstract interface
    real function ifc(i)
      integer, intent(in) :: i
    end function
  end interface
  type :: t
    real :: b(10)
    real :: c(10)
    real :: s
    procedure(ifc), pointer, nopass :: proc
  end type
end module

subroutine test_unlisted_element_reads(x, i)
  use component_element_types
  type(t) :: x
  integer :: i
  real :: r(2)
  !$acc parallel default(none) copyout(r)
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  r(1) = x%b(1)
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  r(2) = x%b(i)
  !$acc end parallel
end subroutine

subroutine test_unlisted_element_write(x, i)
  use component_element_types
  type(t) :: x
  integer :: i
  !$acc parallel default(none)
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  x%b(1) = 1.0
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  x%b(i) = 1.0
  !$acc end parallel
end subroutine

subroutine test_unlisted_element_actual_argument(x)
  use component_element_types
  type(t) :: x
  !$acc parallel default(none)
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  call use_real(x%b(1))
  !$acc end parallel
end subroutine

subroutine test_unlisted_section_is_checked_the_same_way(x)
  use component_element_types
  type(t) :: x
  real :: r(2)
  !$acc parallel default(none) copyout(r)
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  r(1:1) = x%b(1:1)
  !$acc end parallel
end subroutine

subroutine test_listed_component_covers_its_elements(x, i)
  use component_element_types
  type(t) :: x
  integer :: i
  real :: r(2)
  !$acc parallel default(none) copy(x%b) copyout(r)
  r(1) = x%b(1)
  r(2) = x%b(i)
  x%b(2) = 1.0
  !$acc end parallel
end subroutine

subroutine test_listed_component_section_covers_contained_element(x)
  use component_element_types
  type(t) :: x
  real :: r
  !$acc parallel default(none) copy(x%b(1:5)) copyout(r)
  r = x%b(3)
  !$acc end parallel
end subroutine

subroutine test_listed_component_section_rejects_other_element(x)
  use component_element_types
  type(t) :: x
  real :: r
  !$acc parallel default(none) copy(x%b(1:5)) copyout(r)
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  r = x%b(7)
  !$acc end parallel
end subroutine

subroutine test_listed_component_does_not_cover_sibling_element(x)
  use component_element_types
  type(t) :: x
  real :: r
  !$acc parallel default(none) copy(x%b) copyout(r)
  !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-mapping clause
  r = x%c(1)
  !$acc end parallel
end subroutine

subroutine test_listed_object_covers_element_and_procedure_pointer(x)
  use component_element_types
  type(t) :: x
  real :: r(2)
  !$acc parallel default(none) copy(x) copyout(r)
  r(1) = x%b(1)
  r(2) = x%proc(1)
  !$acc end parallel
end subroutine

subroutine test_elements_of_listed_array_of_objects(y, i)
  use component_element_types
  type(t) :: y(3)
  integer :: i
  real :: r
  !$acc parallel default(none) copy(y) copyout(r)
  r = y(2)%b(i)
  !$acc end parallel
end subroutine

subroutine test_without_default_none_elements_are_not_flagged(x)
  use component_element_types
  type(t) :: x
  real :: r
  !$acc parallel copyout(r)
  r = x%b(1)
  !$acc end parallel
end subroutine
