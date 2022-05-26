! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in ucobound() function references

program ucobound_tests
  use iso_c_binding, only : c_int32_t
  implicit none

  integer n, i, array(1), non_coarray(1), scalar_coarray[*], array_coarray(1)[*], non_constant, scalar
  logical non_integer
  integer, allocatable :: ucobounds(:)
  integer, parameter :: non_existent=2

  !___ standard-conforming statement with no optional arguments present ___
  ucobounds = ucobound(scalar_coarray)
  ucobounds = ucobound(array_coarray)
  ucobounds = ucobound(coarray=scalar_coarray)

  !___ standard-conforming statements with optional dim argument present ___
  n = ucobound(scalar_coarray, 1)
  n = ucobound(scalar_coarray, dim=1)
  n = ucobound(coarray=scalar_coarray, dim=1)
  n = ucobound( dim=1, coarray=scalar_coarray)

  !___ standard-conforming statements with optional kind argument present ___
  n = ucobound(scalar_coarray, 1, c_int32_t)

  n = ucobound(scalar_coarray, 1, kind=c_int32_t)

  n = ucobound(scalar_coarray, dim=1, kind=c_int32_t)
  n = ucobound(scalar_coarray, kind=c_int32_t, dim=1)

  ucobounds = ucobound(scalar_coarray, kind=c_int32_t)

  ucobounds = ucobound(coarray=scalar_coarray, kind=c_int32_t)
  ucobounds = ucobound(kind=c_int32_t, coarray=scalar_coarray)

  n = ucobound(coarray=scalar_coarray, dim=1, kind=c_int32_t)
  n = ucobound(dim=1, coarray=scalar_coarray, kind=c_int32_t)
  n = ucobound(kind=c_int32_t, coarray=scalar_coarray, dim=1)
  n = ucobound(dim=1, kind=c_int32_t, coarray=scalar_coarray)
  n = ucobound(kind=c_int32_t, dim=1, coarray=scalar_coarray)

  !___ non-conforming statements ___
  n = ucobound(scalar_coarray, dim=1)
  n = ucobound(array_coarray, dim=non_existent)
  scalar = ucobound(scalar_coarray)

  n = ucobound(dim=i)

  n = ucobound(scalar_coarray, non_integer)

  n = ucobound(scalar_coarray, dim=non_integer)

  ucobounds = ucobound(scalar_coarray, kind=non_integer)
  ucobounds = ucobound(scalar_coarray, kind=non_constant)

  n = ucobound(dim=i, kind=c_int32_t)

  n = ucobound(coarray=scalar_coarray, i)

  ucobounds = ucobound(3.4)

  n = ucobound(scalar_coarray, i, c_int32_t, 0)

  ucobounds = ucobound(coarray=non_coarray)

  n = ucobound(scalar_coarray, i, kind=non_integer)

  n = ucobound(scalar_coarray, array )

  ucobounds = ucobound(c=scalar_coarray)

  n = ucobound(scalar_coarray, dims=i)

  n = ucobound(scalar_coarray, i, kinds=c_int32_t)

end program ucobound_tests
