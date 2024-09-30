! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51

subroutine foo(b)
use iso_c_binding
integer :: x,y
type(C_PTR) :: b
!ERROR: Variable 'x' may not appear on both MAP and PRIVATE clauses on a TARGET construct
!$omp target map(x) private(x)
  x = x + 1
!$omp end target

!ERROR: Variable 'y' in IS_DEVICE_PTR clause must be of type C_PTR
!$omp target map(x) is_device_ptr(y)
  x = x + 1
!$omp end target

!ERROR: Variable 'b' may not appear on both IS_DEVICE_PTR and HAS_DEVICE_ADDR clauses on a TARGET construct
!$omp target map(x) is_device_ptr(b) has_device_addr(b)
  x = x + 1
!$omp end target

!ERROR: Variable 'b' may not appear on both IS_DEVICE_PTR and PRIVATE clauses on a TARGET construct
!$omp target map(x) is_device_ptr(b) private(b)
  x = x + 1
!$omp end target

!ERROR: Variable 'y' may not appear on both HAS_DEVICE_ADDR and FIRSTPRIVATE clauses on a TARGET construct
!$omp target map(x) has_device_addr(y) firstprivate(y)
  y = y - 1
!$omp end target

end subroutine foo

subroutine bar(b1, b2, b3)
  use iso_c_binding
  integer :: y
  type(c_ptr) :: c
  type(c_ptr), allocatable :: b1
  type(c_ptr), pointer :: b2
  type(c_ptr), value :: b3

  !WARNING: Variable 'c' in IS_DEVICE_PTR clause must be a dummy argument. This semantic check is deprecated from OpenMP 5.2 and later.
  !$omp target is_device_ptr(c)
    y = y + 1
  !$omp end target
  !WARNING: Variable 'b1' in IS_DEVICE_PTR clause must be a dummy argument that does not have the ALLOCATABLE, POINTER or VALUE attribute. This semantic check is deprecated from OpenMP 5.2 and later.
  !$omp target is_device_ptr(b1)
    y = y + 1
  !$omp end target
  !WARNING: Variable 'b2' in IS_DEVICE_PTR clause must be a dummy argument that does not have the ALLOCATABLE, POINTER or VALUE attribute. This semantic check is deprecated from OpenMP 5.2 and later.
  !$omp target is_device_ptr(b2)
    y = y + 1
  !$omp end target
  !WARNING: Variable 'b3' in IS_DEVICE_PTR clause must be a dummy argument that does not have the ALLOCATABLE, POINTER or VALUE attribute. This semantic check is deprecated from OpenMP 5.2 and later.
  !$omp target is_device_ptr(b3)
    y = y + 1
  !$omp end target
end subroutine bar
