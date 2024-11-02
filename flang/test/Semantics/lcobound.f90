! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in lcobound() function references

program lcobound_tests
  use iso_c_binding, only : c_int32_t
  implicit none

  integer n, i, array(1), non_coarray(1), scalar_coarray[*], array_coarray(1)[*], non_constant, scalar
  logical non_integer
  integer, allocatable :: lcobounds(:)

  !___ standard-conforming statement with no optional arguments present ___
  lcobounds = lcobound(scalar_coarray)
  lcobounds = lcobound(array_coarray)
  lcobounds = lcobound(coarray=scalar_coarray)

  !___ standard-conforming statements with optional dim argument present ___
  n = lcobound(scalar_coarray, 1)
  n = lcobound(scalar_coarray, dim=1)
  n = lcobound(coarray=scalar_coarray, dim=1)
  n = lcobound( dim=1, coarray=scalar_coarray)

  !___ standard-conforming statements with optional kind argument present ___
  n = lcobound(scalar_coarray, 1, c_int32_t)

  n = lcobound(scalar_coarray, 1, kind=c_int32_t)

  n = lcobound(scalar_coarray, dim=1, kind=c_int32_t)
  n = lcobound(scalar_coarray, kind=c_int32_t, dim=1)

  lcobounds = lcobound(scalar_coarray, kind=c_int32_t)

  lcobounds = lcobound(coarray=scalar_coarray, kind=c_int32_t)
  lcobounds = lcobound(kind=c_int32_t, coarray=scalar_coarray)

  n = lcobound(coarray=scalar_coarray, dim=1, kind=c_int32_t)
  n = lcobound(dim=1, coarray=scalar_coarray, kind=c_int32_t)
  n = lcobound(kind=c_int32_t, coarray=scalar_coarray, dim=1)
  n = lcobound(dim=1, kind=c_int32_t, coarray=scalar_coarray)
  n = lcobound(kind=c_int32_t, dim=1, coarray=scalar_coarray)

  !___ non-conforming statements ___
  n = lcobound(scalar_coarray, dim=1)
  n = lcobound(array_coarray, dim=2)
  scalar = lcobound(scalar_coarray)

  n = lcobound(dim=i)

  n = lcobound(scalar_coarray, non_integer)

  n = lcobound(scalar_coarray, dim=non_integer)

  lcobounds = lcobound(scalar_coarray, kind=non_integer)
  lcobounds = lcobound(scalar_coarray, kind=non_constant)

  n = lcobound(dim=i, kind=c_int32_t)

  n = lcobound(coarray=scalar_coarray, i)

  lcobounds = lcobound(3.4)

  n = lcobound(scalar_coarray, i, c_int32_t, 0)

  lcobounds = lcobound(coarray=non_coarray)

  n = lcobound(scalar_coarray, i, kind=non_integer)

  n = lcobound(scalar_coarray, array )

  lcobounds = lcobound(c=scalar_coarray)

  n = lcobound(scalar_coarray, dims=i)

  n = lcobound(scalar_coarray, i, kinds=c_int32_t)

end program lcobound_tests
