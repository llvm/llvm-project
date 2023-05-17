! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ucobound() function references

program ucobound_tests
  use iso_c_binding, only : c_int32_t, c_int64_t
  implicit none

  integer n, i, array(1), non_coarray(1), scalar_coarray[*], array_coarray(1)[*], non_constant, scalar
  integer, parameter :: const_out_of_range_dim = 5, const_in_range_dim = 1
  real, allocatable :: coarray_corank3[:,:,:]
  logical non_integer, logical_coarray[3,*]
  logical, parameter :: const_non_integer = .true.
  integer, allocatable :: ucobounds(:)

  !___ standard-conforming statement with no optional arguments present ___
  ucobounds = ucobound(scalar_coarray)
  ucobounds = ucobound(array_coarray)
  ucobounds = ucobound(coarray_corank3)
  ucobounds = ucobound(logical_coarray)
  ucobounds = ucobound(coarray=scalar_coarray)

  !___ standard-conforming statements with optional dim argument present ___
  n = ucobound(scalar_coarray, 1)
  n = ucobound(coarray_corank3, 1)
  n = ucobound(coarray_corank3, 3)
  n = ucobound(scalar_coarray, const_in_range_dim)
  n = ucobound(logical_coarray, const_in_range_dim)
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

  !ERROR: DIM=0 dimension is out of range for coarray with corank 1
  n = ucobound(scalar_coarray, dim=0)

  !ERROR: DIM=0 dimension is out of range for coarray with corank 3
  n = ucobound(coarray_corank3, dim=0)

  !ERROR: DIM=-1 dimension is out of range for coarray with corank 1
  n = ucobound(scalar_coarray, dim=-1)

  !ERROR: DIM=2 dimension is out of range for coarray with corank 1
  n = ucobound(array_coarray, dim=2)

  !ERROR: DIM=2 dimension is out of range for coarray with corank 1
  n = ucobound(array_coarray, 2)

  !ERROR: DIM=4 dimension is out of range for coarray with corank 3
  n = ucobound(coarray_corank3, dim=4)

  !ERROR: DIM=4 dimension is out of range for coarray with corank 3
  n = ucobound(dim=4, coarray=coarray_corank3)

  !ERROR: DIM=5 dimension is out of range for coarray with corank 3
  n = ucobound(coarray_corank3, const_out_of_range_dim)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
  scalar = ucobound(scalar_coarray)

  !ERROR: missing mandatory 'coarray=' argument
  n = ucobound(dim=i)

  !ERROR: Actual argument for 'dim=' has bad type 'LOGICAL(4)'
  n = ucobound(scalar_coarray, non_integer)

  !ERROR: Actual argument for 'dim=' has bad type 'LOGICAL(4)'
  n = ucobound(scalar_coarray, dim=non_integer)

  !ERROR: Actual argument for 'kind=' has bad type 'LOGICAL(4)'
  ucobounds = ucobound(scalar_coarray, kind=const_non_integer)

  !ERROR: Actual argument for 'kind=' has bad type 'LOGICAL(4)'
  n = ucobound(scalar_coarray, 1, const_non_integer)

  !ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
  ucobounds = ucobound(scalar_coarray, kind=non_constant)

  !ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
  n = ucobound(scalar_coarray, dim=1, kind=non_constant)

  !ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
  n = ucobound(scalar_coarray, 1, non_constant)

  !ERROR: missing mandatory 'coarray=' argument
  n = ucobound(dim=i, kind=c_int32_t)

  !ERROR: actual argument #2 without a keyword may not follow an actual argument with a keyword
  n = ucobound(coarray=scalar_coarray, i)

  n = ucobound(coarray=scalar_coarray, dim=i)

  !ERROR: missing mandatory 'coarray=' argument
  ucobounds = ucobound()

  !ERROR: 'coarray=' argument must have corank > 0 for intrinsic 'ucobound'
  ucobounds = ucobound(3.4)

  !ERROR: keyword argument to intrinsic 'ucobound' was supplied positionally by an earlier actual argument
  n = ucobound(scalar_coarray, 1, coarray=scalar_coarray)

  !ERROR: too many actual arguments for intrinsic 'ucobound'
  n = ucobound(scalar_coarray, i, c_int32_t, 0)

  !ERROR: 'coarray=' argument must have corank > 0 for intrinsic 'ucobound'
  ucobounds = ucobound(coarray=non_coarray)

  !ERROR: 'coarray=' argument must have corank > 0 for intrinsic 'ucobound'
  n = ucobound(coarray=non_coarray, dim=1)

  !ERROR: 'dim=' argument has unacceptable rank 1
  n = ucobound(scalar_coarray, array )

  !ERROR: unknown keyword argument to intrinsic 'ucobound'
  ucobounds = ucobound(c=scalar_coarray)

  !ERROR: unknown keyword argument to intrinsic 'ucobound'
  n = ucobound(scalar_coarray, dims=i)

  !ERROR: unknown keyword argument to intrinsic 'ucobound'
  n = ucobound(scalar_coarray, i, kinds=c_int32_t)

  !ERROR: repeated keyword argument to intrinsic 'ucobound'
  n = ucobound(scalar_coarray, dim=1, dim=2)

  !ERROR: repeated keyword argument to intrinsic 'ucobound'
  ucobounds = ucobound(coarray=scalar_coarray, coarray=array_coarray)

  !ERROR: repeated keyword argument to intrinsic 'ucobound'
  ucobounds = ucobound(scalar_coarray, kind=c_int32_t, kind=c_int64_t)

end program ucobound_tests
