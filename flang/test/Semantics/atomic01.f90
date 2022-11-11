! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in atomic_add() subroutine based on the
! statement specification in section 16.9.20 of the Fortran 2018 standard.

program test_atomic_add
  use iso_fortran_env, only : atomic_int_kind
  implicit none

  integer(kind=atomic_int_kind) atom_object[*], atom_array(2)[*], quantity, array(1), coarray[*], non_coarray
  integer non_atom_object[*], non_atom, non_scalar(1), status, stat_array(1), coindexed[*]
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
  call atomic_add(non_atom_object, quantity)

  ! atom must be a coarray
  call atomic_add(non_coarray, quantity)

  ! atom must be a scalar variable
  call atomic_add(atom_array, quantity)

  ! atom has an unknown keyword argument
  call atomic_add(atoms=atom_object, value=quantity)

  ! atom has an argument mismatch
  call atomic_add(atom=non_atom_object, value=quantity)

  ! value must be an integer
  call atomic_add(atom_object, non_integer)

  ! value must be an integer scalar
  call atomic_add(atom_object, array)

  ! value must be of kind atomic_int_kind
  call atomic_add(atom_object, non_atom)

  ! value has an unknown keyword argument
  call atomic_add(atom_object, values=quantity)

  ! value has an argument mismatch
  call atomic_add(atom_object, value=non_integer)

  ! stat must be an integer
  call atomic_add(atom_object, quantity, non_integer)

  ! stat must be an integer scalar
  call atomic_add(atom_object, quantity, non_scalar)

  ! stat is an intent(out) argument
  call atomic_add(atom_object, quantity, 8)

  ! stat has an unknown keyword argument
  call atomic_add(atom_object, quantity, statuses=status)

  ! stat has an argument mismatch
  call atomic_add(atom_object, quantity, stat=non_integer)

  ! stat must not be coindexed
  call atomic_add(atom_object, quantity, coindexed[1])

  ! Too many arguments
  call atomic_add(atom_object, quantity, status, stat_array(1))

  ! Repeated atom keyword
  call atomic_add(atom=atom_object, atom=atom_array(1), value=quantity)

  ! Repeated value keyword
  call atomic_add(atom=atom_object, value=quantity, value=array(1))

  ! Repeated stat keyword
  call atomic_add(atom=atom_object, value=quantity, stat=status, stat=stat_array(1))

end program test_atomic_add
