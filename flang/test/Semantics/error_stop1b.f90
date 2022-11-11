! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in error stop statements based on the
! statement specification in section 11.4 of the Fortran 2018 standard.
! The errors in this test would be hidden by the errors in
! the test error_stop01a.f90 if they were included in that file,
! and are thus tested here.

program test_error_stop
  implicit none

  integer int_code, int_array(1), int_coarray[*], array_coarray(1)[*]
  integer(kind=1) non_default_int_kind
  character(len=128) char_code, char_array(1), char_coarray[*], non_logical
  character(kind=4, len=128) non_default_char_kind
  logical bool, logical_array(1), logical_coarray[*], non_integer, non_character

  !___ non-standard-conforming statements _________________________

  !ERROR: Stop code must be of INTEGER or CHARACTER type
  error stop non_integer

  !ERROR: Stop code must be of INTEGER or CHARACTER type
  error stop non_character

  !ERROR: INTEGER stop code must be of default kind
  error stop non_default_int_kind

  !ERROR: CHARACTER stop code must be of default kind
  error stop non_default_char_kind

  !ERROR: Must be a scalar value, but is a rank-1 array
  error stop char_array

  !ERROR: Must be a scalar value, but is a rank-1 array
  error stop array_coarray[1]

  !ERROR: Must have LOGICAL type, but is CHARACTER(KIND=1,LEN=128_8)
  error stop int_code, quiet=non_logical

  !ERROR: Must be a scalar value, but is a rank-1 array
  error stop int_code, quiet=logical_array

end program test_error_stop
