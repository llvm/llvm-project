! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in atomic_ref subroutine calls based on
! the interface defined in section 16.9.29 of the Fortran 2018 standard.

program test_atomic_ref
  use iso_fortran_env, only: atomic_int_kind, atomic_logical_kind
  implicit none

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
  call atomic_ref(val, scalar_coarray)
  call atomic_ref(default_int_val, scalar_coarray)
  call atomic_ref(val, scalar_coarray[1])
  call atomic_ref(default_int_val, scalar_coarray, status)
  call atomic_ref(val, scalar_coarray[1], status)
  call atomic_ref(default_int_val, scalar_coarray, stat=status)
  call atomic_ref(val, atom=scalar_coarray, stat=status)
  call atomic_ref(value=default_int_val, atom=scalar_coarray)
  call atomic_ref(value=val, atom=scalar_coarray, stat=status)
  call atomic_ref(stat=status, atom=scalar_coarray, value=default_int_val)

  call atomic_ref(val_logical, atom_logical)
  call atomic_ref(default_logical_val, atom_logical)
  call atomic_ref(val_logical, atom_logical[1])
  call atomic_ref(val_logical, atom_logical, status)
  call atomic_ref(val_logical, atom_logical[1], status)
  call atomic_ref(val_logical, atom_logical, stat=status)
  call atomic_ref(val_logical, atom=atom_logical, stat=status)
  call atomic_ref(value=val_logical, atom=atom_logical)
  call atomic_ref(value=val_logical, atom=atom_logical, stat=status)
  call atomic_ref(stat=status, atom=atom_logical, value=val_logical)

  !___ non-standard-conforming calls ___

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_ref'
  call atomic_ref(val, non_scalar_coarray)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_ref'
  call atomic_ref(val_logical, non_scalar_logical_coarray)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_ref'
  call atomic_ref(val, non_coarray)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_ref'
  call atomic_ref(val_logical, non_coarray_logical)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_ref'
  call atomic_ref(val, array)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind or atomic_logical_kind, but is 'INTEGER(4)'
  call atomic_ref(val, default_kind_coarray)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind or atomic_logical_kind, but is 'INTEGER(1)'
  call atomic_ref(val, kind1_coarray)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind or atomic_logical_kind, but is 'LOGICAL(4)'
  call atomic_ref(val_logical, default_kind_logical_coarray)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind or atomic_logical_kind, but is 'LOGICAL(1)'
  call atomic_ref(val_logical, kind1_logical_coarray)

  !ERROR: Actual argument for 'value=' must have same type as 'atom=', but is 'LOGICAL(8)'
  call atomic_ref(val_logical, scalar_coarray)

  !ERROR: Actual argument for 'value=' must have same type as 'atom=', but is 'INTEGER(8)'
  call atomic_ref(val, atom_logical)

  !ERROR: Actual argument for 'atom=' has bad type 'REAL(4)'
  call atomic_ref(val, non_integer_coarray)

  !ERROR: 'value=' argument has unacceptable rank 1
  call atomic_ref(array, scalar_coarray)

  !ERROR: Actual argument for 'value=' has bad type 'REAL(4)'
  call atomic_ref(non_int_or_logical, scalar_coarray)

  !ERROR: Actual argument for 'value=' has bad type 'REAL(4)'
  call atomic_ref(non_int_or_logical, atom_logical)

  !ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call atomic_ref(val, scalar_coarray, non_integer)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call atomic_ref(val, scalar_coarray, status_array)

  !ERROR: 'stat' argument to 'atomic_ref' may not be a coindexed object
  call atomic_ref(val, scalar_coarray, coindexed_status[1])

  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' must be definable
  call atomic_ref(val, scalar_coarray, 1)

  !ERROR: missing mandatory 'value=' argument
  call atomic_ref()

  !ERROR: missing mandatory 'value=' argument
  call atomic_ref(atom=scalar_coarray, stat=status)

  !ERROR: missing mandatory 'atom=' argument
  call atomic_ref(val)

  !ERROR: missing mandatory 'atom=' argument
  call atomic_ref(value=val, stat=status)

  !ERROR: too many actual arguments for intrinsic 'atomic_ref'
  call atomic_ref(val, scalar_coarray, status, extra_arg)

  !ERROR: repeated keyword argument to intrinsic 'atomic_ref'
  call atomic_ref(value=val, value=repeated_val, atom=scalar_coarray, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_ref'
  call atomic_ref(value=val, atom=scalar_coarray, atom=repeated_atom, stat=status)

  !ERROR: repeated keyword argument to intrinsic 'atomic_ref'
  call atomic_ref(value=val, atom=scalar_coarray, stat=status, stat=repeated_status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_ref'
  call atomic_ref(values=val, atom=scalar_coarray, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_ref'
  call atomic_ref(value=val, atomic=scalar_coarray, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_ref'
  call atomic_ref(value=val, atom=scalar_coarray, status=status)

  !ERROR: keyword argument to intrinsic 'atomic_ref' was supplied positionally by an earlier actual argument
  call atomic_ref(val, value=repeated_val, scalar_coarray)

  !ERROR: keyword argument to intrinsic 'atomic_ref' was supplied positionally by an earlier actual argument
  call atomic_ref(val, scalar_coarray, atom=repeated_atom)

  !ERROR: keyword argument to intrinsic 'atomic_ref' was supplied positionally by an earlier actual argument
  call atomic_ref(val, scalar_coarray, status, stat=repeated_status)

end program test_atomic_ref
