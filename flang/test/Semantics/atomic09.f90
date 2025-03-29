! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in atomic_or subroutine calls based on
! the interface defined in section 16.9.28 of the Fortran 2018 standard.

program test_atomic_or
  use iso_fortran_env, only: atomic_int_kind, atomic_logical_kind
  implicit none(external, type)

  integer(kind=atomic_int_kind) :: scalar_coarray[*], non_scalar_coarray(10)[*], val, non_coarray
  integer(kind=atomic_int_kind) :: repeated_atom[*], repeated_val, array(10)
  integer :: status, default_kind_coarray[*], coindexed_status[*], extra_arg, repeated_status, status_array(10)
  integer(kind=1) :: kind1_coarray[*]
  real :: non_integer_coarray[*]
  logical :: non_integer
  logical(atomic_logical_kind) :: atomic_logical[*]

  !___ standard-conforming calls ___
  call atomic_or(scalar_coarray, val)
  call atomic_or(scalar_coarray[1], val)
  call atomic_or(scalar_coarray, val, status)
  call atomic_or(scalar_coarray[1], val, status)
  call atomic_or(scalar_coarray, val, stat=status)
  call atomic_or(scalar_coarray, value=val, stat=status)
  call atomic_or(atom=scalar_coarray, value=val)
  call atomic_or(atom=scalar_coarray, value=val, stat=status)
  call atomic_or(stat=status, value=val, atom=scalar_coarray)

  !___ non-standard-conforming calls ___

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_or'
  call atomic_or(non_scalar_coarray, val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_or'
  call atomic_or(non_scalar_coarray[1], val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_or'
  call atomic_or(non_coarray, val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_or'
  call atomic_or(array, val)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_or(default_kind_coarray, val)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(1)'
  call atomic_or(kind1_coarray, val)

  !ERROR: Actual argument for 'atom=' has bad type 'REAL(4)'
  call atomic_or(non_integer_coarray, val)

  !ERROR: Actual argument for 'atom=' has bad type 'LOGICAL(8)'
  call atomic_or(atomic_logical, val)

  !ERROR: 'value=' argument has unacceptable rank 1
  call atomic_or(scalar_coarray, array)

  !ERROR: Actual argument for 'value=' has bad type 'LOGICAL(4)'
  call atomic_or(scalar_coarray, non_integer)

  !ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call atomic_or(scalar_coarray, val, non_integer)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call atomic_or(scalar_coarray, val, status_array)

  !ERROR: 'stat' argument to 'atomic_or' may not be a coindexed object
  call atomic_or(scalar_coarray, val, coindexed_status[1])

  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' is not definable
  !BECAUSE: '1_4' is not a variable or pointer
  call atomic_or(scalar_coarray, val, 1)

  !ERROR: missing mandatory 'atom=' argument
  call atomic_or()

  !ERROR: missing mandatory 'atom=' argument
  call atomic_or(value=val, stat=status)

  !ERROR: missing mandatory 'value=' argument
  call atomic_or(scalar_coarray)

  !ERROR: missing mandatory 'value=' argument
  call atomic_or(atom=scalar_coarray, stat=status)

  !ERROR: too many actual arguments for intrinsic 'atomic_or'
  call atomic_or(scalar_coarray, val, status, extra_arg)

  !ERROR: repeated keyword argument to intrinsic 'atomic_or'
  call atomic_or(atom=scalar_coarray, atom=repeated_atom, value=val, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_or'
  call atomic_or(atom=scalar_coarray, value=val, value=repeated_val, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_or'
  call atomic_or(atom=scalar_coarray, value=val, stat=status, stat=repeated_status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_or'
  call atomic_or(atomic=scalar_coarray, value=val, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_or'
  call atomic_or(atom=scalar_coarray, values=val, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_or'
  call atomic_or(atom=scalar_coarray, value=val, status=status)

  !ERROR: keyword argument to intrinsic 'atomic_or' was supplied positionally by an earlier actual argument
  call atomic_or(scalar_coarray, val, atom=repeated_atom)

  !ERROR: keyword argument to intrinsic 'atomic_or' was supplied positionally by an earlier actual argument
  call atomic_or(scalar_coarray, val, value=repeated_val)

  !ERROR: keyword argument to intrinsic 'atomic_or' was supplied positionally by an earlier actual argument
  call atomic_or(scalar_coarray, val, status, stat=repeated_status)

end program test_atomic_or
