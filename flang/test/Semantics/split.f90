! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in split() subroutine calls
! Based on Fortran 2023 standard requirements

program test_split_errors
  implicit none

  character(20) :: string
  character(5) :: set
  integer :: pos
  logical :: back

  ! Valid declarations for testing
  integer :: int_scalar
  real :: real_scalar
  character(10) :: string_array(5)
  character(5) :: set_array(5)
  character(len=20, kind=2) :: string_k2
  character(len=5, kind=2) :: set_k2
  character(len=20, kind=4) :: string_k4
  character(len=5, kind=4) :: set_k4

  !========================================================================
  ! Valid calls (reference)
  !========================================================================

  call split(string, set, pos)
  call split(string, set, pos, back)
  call split("hello world", " ", pos)
  call split("hello world", " ", pos, .false.)

  ! Valid calls with different character kinds
  call split(string_k2, set_k2, pos)
  call split(string_k2, set_k2, pos, back)
  call split(string_k4, set_k4, pos)
  call split(string_k4, set_k4, pos, back)

  !========================================================================
  ! Wrong types for STRING argument
  !========================================================================

  !ERROR: Actual argument for 'string=' has bad type 'INTEGER(4)'
  call split(int_scalar, set, pos)

  !ERROR: Actual argument for 'string=' has bad type 'REAL(4)'
  call split(real_scalar, set, pos)

  !========================================================================
  ! Wrong rank for STRING (must be scalar)
  !========================================================================

  !ERROR: 'string=' argument has unacceptable rank 1
  call split(string_array, set, pos)

  !========================================================================
  ! Wrong types for SET argument
  !========================================================================

  !ERROR: Actual argument for 'set=' has bad type 'INTEGER(4)'
  call split(string, int_scalar, pos)

  !ERROR: Actual argument for 'set=' has bad type 'REAL(4)'
  call split(string, real_scalar, pos)

  !========================================================================
  ! Wrong types for POS argument
  !========================================================================

  !ERROR: Actual argument for 'pos=' has bad type 'REAL(4)'
  call split(string, set, real_scalar)

  !========================================================================
  ! Wrong types for BACK argument
  !========================================================================

  !ERROR: Actual argument for 'back=' has bad type 'INTEGER(4)'
  call split(string, set, pos, int_scalar)

  !========================================================================
  ! Character kind mismatches between STRING and SET
  !========================================================================

  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=1,LEN=5_8)'
  call split(string_k2, set, pos)

  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=2,LEN=5_8)'
  call split(string, set_k2, pos)

  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=1,LEN=5_8)'
  call split(string_k4, set, pos)

  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=4,LEN=5_8)'
  call split(string, set_k4, pos)

  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=4,LEN=5_8)'
  call split(string_k2, set_k4, pos)

end program test_split_errors
