! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in atomic_add() subroutine based on the
! statement specification in section 16.9.20 of the Fortran 2018 standard.

program test_atomic_add
  use iso_fortran_env, only : atomic_int_kind
  implicit none(external, type)

  integer(kind=atomic_int_kind) atom_object[*], atom_array(2)[*], quantity, array(1), coarray[*], non_coarray
  integer non_atom_object[*], non_scalar(1), status, stat_array(1), coindexed[*]
  logical non_integer

  !___ standard-conforming calls with required arguments _______

  call atomic_add(atom_object, quantity)
  call atomic_add(atom_object[1], quantity)
  call atomic_add(atom_array(1), quantity)
  call atomic_add(atom_array(1)[1], quantity)
  call atomic_add(atom_object, array(1))
  call atomic_add(atom_object, coarray[1])
  call atomic_add(atom=atom_object, value=quantity)
  call atomic_add(value=quantity, atom=atom_object)

  !___ standard-conforming calls with all arguments ____________
  call atomic_add(atom_object, quantity, status)
  call atomic_add(atom_object, quantity, stat_array(1))
  call atomic_add(atom=atom_object, value=quantity, stat=status)
  call atomic_add(stat=status, value=quantity, atom=atom_object)

  !___ non-standard-conforming calls _______

  ! atom must be of kind atomic_int_kind
  ! ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_add(non_atom_object, quantity)

  ! atom must be a coarray
  ! ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_add'
  call atomic_add(non_coarray, quantity)

  ! atom must be a scalar variable
  ! ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_add'
  call atomic_add(atom_array, quantity)

  ! atom has an unknown keyword argument
  ! ERROR: unknown keyword argument to intrinsic 'atomic_add'
  call atomic_add(atoms=atom_object, value=quantity)

  ! atom has an argument mismatch
  ! ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_add(atom=non_atom_object, value=quantity)

  ! value must be an integer
  ! ERROR: Actual argument for 'value=' has bad type 'LOGICAL(4)'
  call atomic_add(atom_object, non_integer)

  ! value must be an integer scalar
  ! ERROR: 'value=' argument has unacceptable rank 1
  call atomic_add(atom_object, array)

  ! value has an unknown keyword argument
  ! ERROR: unknown keyword argument to intrinsic 'atomic_add'
  call atomic_add(atom_object, values=quantity)

  ! value has an argument mismatch
  ! ERROR: Actual argument for 'value=' has bad type 'LOGICAL(4)'
  call atomic_add(atom_object, value=non_integer)

  ! stat must be an integer
  ! ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call atomic_add(atom_object, quantity, non_integer)

  ! stat must be an integer scalar
  ! ERROR: 'stat=' argument has unacceptable rank 1
  call atomic_add(atom_object, quantity, non_scalar)

  ! stat is an intent(out) argument
  ! ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' is not definable
  ! ERROR: '8_4' is not a variable or pointer
  call atomic_add(atom_object, quantity, 8)

  ! stat has an unknown keyword argument
  ! ERROR: unknown keyword argument to intrinsic 'atomic_add'
  call atomic_add(atom_object, quantity, statuses=status)

  ! stat has an argument mismatch
  ! ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call atomic_add(atom_object, quantity, stat=non_integer)

  ! stat must not be coindexed
  ! ERROR: 'stat' argument to 'atomic_add' may not be a coindexed object
  call atomic_add(atom_object, quantity, coindexed[1])

  ! Too many arguments
  ! ERROR: too many actual arguments for intrinsic 'atomic_add'
  call atomic_add(atom_object, quantity, status, stat_array(1))

  ! Repeated atom keyword
  ! ERROR: repeated keyword argument to intrinsic 'atomic_add'
  call atomic_add(atom=atom_object, atom=atom_array(1), value=quantity)

  ! Repeated value keyword
  ! ERROR: repeated keyword argument to intrinsic 'atomic_add'
  call atomic_add(atom=atom_object, value=quantity, value=array(1))

  ! Repeated stat keyword
  ! ERROR: repeated keyword argument to intrinsic 'atomic_add'
  call atomic_add(atom=atom_object, value=quantity, stat=status, stat=stat_array(1))

end program test_atomic_add
