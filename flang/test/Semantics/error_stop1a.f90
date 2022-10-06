! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in error stop statements based on the
! statement specification in section 11.4 of the Fortran 2018 standard.

program test_error_stop
  implicit none

  integer int_code, int_array(1), int_coarray[*], array_coarray(1)[*]
  integer(kind=1) non_default_int_kind
  character(len=128) char_code, char_array(1), char_coarray[*], non_logical
  character(kind=4, len=128) non_default_char_kind
  logical bool, logical_array(1), logical_coarray[*], non_integer, non_character

  !___ standard-conforming statements ____________________________
  error stop

  !___ standard-conforming statements with stop-code ______________
  error stop int_code
  error stop 5
  error stop (5)
  error stop ((5 + 8) * 2)
  error stop char_code
  error stop 'c'
  error stop ('c')
  error stop ('program failed')
  error stop int_array(1)
  error stop char_array(1)
  error stop int_coarray
  error stop int_coarray[1]
  error stop char_coarray
  error stop char_coarray[1]
  error stop array_coarray(1)
  error stop array_coarray(1)[1]

  !___ standard-conforming statements with stop-code and quiet= ___
  error stop int_code, quiet=bool
  error stop int_code, quiet=logical_array(1)
  error stop int_code, quiet=logical_coarray
  error stop int_code, quiet=logical_coarray[1]
  error stop int_code, quiet=.true.
  error stop (int_code), quiet=.false.

  !___ non-standard-conforming statements _________________________

  ! unknown stop-code
  !ERROR: expected execution part construct
  error stop code=int_code

  ! missing 'quiet='
  !ERROR: expected execution part construct
  error stop int_code, bool

  ! incorrect spelling for 'quiet='
  !ERROR: expected execution part construct
  error stop int_code, quiets=bool

  ! missing scalar-logical-expr for quiet=
  !ERROR: expected execution part construct
  error stop int_code, quiet

  ! superfluous stop-code
  !ERROR: expected execution part construct
  error stop int_code, char_code

  ! repeated quiet=
  !ERROR: expected execution part construct
  error stop int_code, quiet=bool, quiet=.true.

  ! superfluous stop-code
  !ERROR: expected execution part construct
  error stop int_code, char_code, quiet=bool

  ! superfluous integer
  !ERROR: expected execution part construct
  error stop int_code, quiet=bool, 5

  ! quiet= appears without stop-code
  !ERROR: expected execution part construct
  error stop quiet=bool

  ! incorrect syntax
  !ERROR: expected execution part construct
  error stop ()

  ! incorrect syntax
  !ERROR: expected execution part construct
  error stop (2, quiet=.true.)

end program test_error_stop
