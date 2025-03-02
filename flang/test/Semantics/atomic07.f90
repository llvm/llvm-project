! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in atomic_fetch_or subroutine calls based on
! the interface defined in section 16.9.26 of the Fortran 2018 standard.

program test_atomic_fetch_or
  use iso_fortran_env, only: atomic_int_kind
  implicit none(external, type)

  integer(kind=atomic_int_kind) :: scalar_coarray[*], non_scalar_coarray(10)[*], val, old_val, non_coarray
  integer(kind=atomic_int_kind) :: repeated_atom[*], repeated_old, repeated_val, array(10), val_coarray[*], old_val_coarray[*]
  integer :: status, default_kind_coarray[*], not_same_kind_as_atom, coindexed_status[*]
  integer :: extra_arg, repeated_status, status_array(10)
  integer(kind=1) :: kind1_coarray[*]
  real :: non_integer_coarray[*], not_same_type_as_atom
  logical :: non_integer

  !___ standard-conforming calls ___
  call atomic_fetch_or(scalar_coarray, val, old_val)
  call atomic_fetch_or(scalar_coarray[1], val, old_val)
  call atomic_fetch_or(scalar_coarray, val, old_val, status)
  call atomic_fetch_or(scalar_coarray[1], val, old_val, status)
  call atomic_fetch_or(atom=scalar_coarray, value=val, old=old_val, stat=status)
  call atomic_fetch_or(stat=status, old=old_val, value=val, atom=scalar_coarray)

  !___ non-standard-conforming calls ___

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(non_scalar_coarray, val, old_val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(non_coarray, val, old_val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(array, val, old_val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(non_scalar_coarray[1], val, old_val)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_fetch_or(default_kind_coarray, val, old_val)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(1)'
  call atomic_fetch_or(kind1_coarray, val, old_val)

  !ERROR: Actual argument for 'atom=' has bad type 'REAL(4)'
  call atomic_fetch_or(non_integer_coarray, val, old_val)

  !ERROR: 'value=' argument has unacceptable rank 1
  call atomic_fetch_or(scalar_coarray, array, old_val)

  !ERROR: Actual argument for 'value=' has bad type 'LOGICAL(4)'
  call atomic_fetch_or(scalar_coarray, non_integer, old_val)

  !ERROR: Actual argument for 'old=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_fetch_or(atom=scalar_coarray, value=val, old=not_same_kind_as_atom)

  !ERROR: Actual argument for 'old=' has bad type 'REAL(4)'
  call atomic_fetch_or(atom=scalar_coarray, value=val, old=not_same_type_as_atom)

  !ERROR: Actual argument for 'old=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_fetch_or(scalar_coarray, val, 1)

  !ERROR: 'old=' argument has unacceptable rank 1
  call atomic_fetch_or(scalar_coarray, val, array)

  !ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call atomic_fetch_or(scalar_coarray, val, old_val, non_integer)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call atomic_fetch_or(scalar_coarray, val, old_val, status_array)

  !ERROR: 'stat' argument to 'atomic_fetch_or' may not be a coindexed object
  call atomic_fetch_or(scalar_coarray, val, old_val, coindexed_status[1])

  !ERROR: 'stat' argument to 'atomic_fetch_or' may not be a coindexed object
  call atomic_fetch_or(scalar_coarray[1], val_coarray[1], old_val_coarray[1], coindexed_status[1])

  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' is not definable
  !BECAUSE: '1_4' is not a variable or pointer
  call atomic_fetch_or(scalar_coarray, val, old_val, 1)

  !ERROR: missing mandatory 'atom=' argument
  call atomic_fetch_or()

  !ERROR: missing mandatory 'atom=' argument
  call atomic_fetch_or(value=val, old=old_val, stat=status)

  !ERROR: missing mandatory 'value=' argument
  call atomic_fetch_or(scalar_coarray)

  !ERROR: missing mandatory 'value=' argument
  call atomic_fetch_or(atom=scalar_coarray, old=old_val, stat=status)

  !ERROR: missing mandatory 'old=' argument
  call atomic_fetch_or(scalar_coarray, val)

  !ERROR: missing mandatory 'old=' argument
  call atomic_fetch_or(atom=scalar_coarray, value=val)

  !ERROR: too many actual arguments for intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(scalar_coarray, val, old_val, status, extra_arg)

  !ERROR: repeated keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atom=scalar_coarray, atom=repeated_atom, value=val, old=old_val, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atom=scalar_coarray, value=val, value=repeated_val, old=old_val, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atom=scalar_coarray, value=val, old=old_val, old=repeated_old, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atom=scalar_coarray, value=val, old=old_val, stat=status, stat=repeated_status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atomic=scalar_coarray, value=val, old=old_val, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atom=scalar_coarray, values=val, old=old_val, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atom=scalar_coarray, value=val, oldvalue=old_val, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_fetch_or'
  call atomic_fetch_or(atom=scalar_coarray, value=val, old=old_val, status=status)

  !ERROR: keyword argument to intrinsic 'atomic_fetch_or' was supplied positionally by an earlier actual argument
  call atomic_fetch_or(scalar_coarray, val, old_val, atom=repeated_atom)

  !ERROR: keyword argument to intrinsic 'atomic_fetch_or' was supplied positionally by an earlier actual argument
  call atomic_fetch_or(scalar_coarray, val, old_val, value=repeated_val)

  !ERROR: keyword argument to intrinsic 'atomic_fetch_or' was supplied positionally by an earlier actual argument
  call atomic_fetch_or(scalar_coarray, val, old_val, status, stat=repeated_status)

end program test_atomic_fetch_or
