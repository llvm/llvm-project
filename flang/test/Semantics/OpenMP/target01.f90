! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
 
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

end
