! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in atomic_define subroutine calls based on
! the interface defined in section 16.9.23 of the Fortran 2018 standard.

program test_atomic_define
  use iso_fortran_env, only: atomic_int_kind, atomic_logical_kind
  implicit none(external, type)

  integer(kind=atomic_int_kind) :: scalar_coarray[*], non_scalar_coarray(10)[*], val, non_coarray
  integer(kind=atomic_int_kind) :: repeated_atom[*], repeated_val, array(10)
  integer :: status, default_kind_coarray[*], coindexed_status[*], extra_arg, repeated_status, status_array(10), default_int_val
  real :: non_integer_coarray[*], non_int_or_logical
  logical(kind=atomic_logical_kind) :: atom_logical[*], val_logical, non_scalar_logical_coarray(10)[*], non_coarray_logical
  logical :: non_integer, default_kind_logical_coarray[*], default_logical_val

  ! These variables are used in this test based on the assumption that atomic_int_kind is not equal to kind=1
  ! This is true at the time of writing of the test, but of course is not guaranteed to stay the same
  integer(kind=1) :: kind1_coarray[*]
  logical(kind=1) :: kind1_logical_coarray[*]

  !___ standard-conforming calls ___
  call atomic_define(scalar_coarray, val)
  call atomic_define(scalar_coarray, default_int_val)
  call atomic_define(scalar_coarray[1], val)
  call atomic_define(scalar_coarray, default_int_val, status)
  call atomic_define(scalar_coarray[1], val, status)
  call atomic_define(scalar_coarray, default_int_val, stat=status)
  call atomic_define(scalar_coarray, value=val, stat=status)
  call atomic_define(atom=scalar_coarray, value=default_int_val)
  call atomic_define(atom=scalar_coarray, value=val, stat=status)
  call atomic_define(stat=status, value=default_int_val, atom=scalar_coarray)

  call atomic_define(atom_logical, val_logical)
  call atomic_define(atom_logical, default_logical_val)
  call atomic_define(atom_logical[1], val_logical)
  call atomic_define(atom_logical, val_logical, status)
  call atomic_define(atom_logical[1], val_logical, status)
  call atomic_define(atom_logical, val_logical, stat=status)
  call atomic_define(atom_logical, value=val_logical, stat=status)
  call atomic_define(atom=atom_logical, value=val_logical)
  call atomic_define(atom=atom_logical, value=val_logical, stat=status)
  call atomic_define(stat=status, value=val_logical, atom=atom_logical)

  !___ non-standard-conforming calls ___

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_define'
  call atomic_define(non_scalar_coarray, val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_define'
  call atomic_define(non_scalar_coarray[1], val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_define'
  call atomic_define(non_scalar_logical_coarray, val_logical)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_define'
  call atomic_define(non_scalar_logical_coarray[1], val_logical)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_define'
  call atomic_define(non_coarray, val)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_define'
  call atomic_define(non_coarray_logical, val_logical)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_define'
  call atomic_define(array, val)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_define(default_kind_coarray, val)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(1)'
  call atomic_define(kind1_coarray, val)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_logical_kind, but is 'LOGICAL(4)'
  call atomic_define(default_kind_logical_coarray, val_logical)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_logical_kind, but is 'LOGICAL(1)'
  call atomic_define(kind1_logical_coarray, val_logical)

  !ERROR: 'value=' argument to 'atomic_define' must have same type as 'atom=', but is 'LOGICAL(8)'
  call atomic_define(scalar_coarray, val_logical)

  !ERROR: 'value=' argument to 'atomic_define' must have same type as 'atom=', but is 'INTEGER(8)'
  call atomic_define(atom_logical, val)

  !ERROR: Actual argument for 'atom=' has bad type 'REAL(4)'
  call atomic_define(non_integer_coarray, val)

  !ERROR: 'value=' argument has unacceptable rank 1
  call atomic_define(scalar_coarray, array)

  !ERROR: Actual argument for 'value=' has bad type 'REAL(4)'
  call atomic_define(scalar_coarray, non_int_or_logical)

  !ERROR: Actual argument for 'value=' has bad type 'REAL(4)'
  call atomic_define(atom_logical, non_int_or_logical)

  !ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call atomic_define(scalar_coarray, val, non_integer)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call atomic_define(scalar_coarray, val, status_array)

  !ERROR: 'stat' argument to 'atomic_define' may not be a coindexed object
  call atomic_define(scalar_coarray, val, coindexed_status[1])

  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' is not definable
  !BECAUSE: '1_4' is not a variable or pointer
  call atomic_define(scalar_coarray, val, 1)

  !ERROR: missing mandatory 'atom=' argument
  call atomic_define()

  !ERROR: missing mandatory 'atom=' argument
  call atomic_define(value=val, stat=status)

  !ERROR: missing mandatory 'value=' argument
  call atomic_define(scalar_coarray)

  !ERROR: missing mandatory 'value=' argument
  call atomic_define(atom=scalar_coarray, stat=status)

  !ERROR: too many actual arguments for intrinsic 'atomic_define'
  call atomic_define(scalar_coarray, val, status, extra_arg)

  !ERROR: repeated keyword argument to intrinsic 'atomic_define'
  call atomic_define(atom=scalar_coarray, atom=repeated_atom, value=val, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_define'
  call atomic_define(atom=scalar_coarray, value=val, value=repeated_val, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_define'
  call atomic_define(atom=scalar_coarray, value=val, stat=status, stat=repeated_status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_define'
  call atomic_define(atomic=scalar_coarray, value=val, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_define'
  call atomic_define(atom=scalar_coarray, values=val, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_define'
  call atomic_define(atom=scalar_coarray, value=val, status=status)

  !ERROR: keyword argument to intrinsic 'atomic_define' was supplied positionally by an earlier actual argument
  call atomic_define(scalar_coarray, val, atom=repeated_atom)

  !ERROR: keyword argument to intrinsic 'atomic_define' was supplied positionally by an earlier actual argument
  call atomic_define(scalar_coarray, val, value=repeated_val)

  !ERROR: keyword argument to intrinsic 'atomic_define' was supplied positionally by an earlier actual argument
  call atomic_define(scalar_coarray, val, status, stat=repeated_status)

end program test_atomic_define
