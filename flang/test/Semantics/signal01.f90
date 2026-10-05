! RUN: %python %S/test_errors.py %s %flang_fc1

! Tests for semantic checking of the SIGNAL intrinsic.

subroutine signal_handler(sig)
  integer :: sig
end subroutine

integer function signal_handler_function(sig)
  integer :: sig
  signal_handler_function = 0
end function

subroutine test_handler
  external :: signal_handler
  integer, external :: signal_handler_function

  integer :: integer_handler
  real :: real_handler
  logical :: logical_handler
  complex :: complex_handler
  character :: character_handler

  ! Valid procedure handlers.
  call signal(8, signal_handler)
  call signal(8, signal_handler_function)

  ! Valid scalar INTEGER handlers.
  call signal(8, 0)
  call signal(8, integer_handler)

  !ERROR: 'handler=' argument to SIGNAL() must be a procedure or a scalar INTEGER
  call signal(8, real_handler)

  !ERROR: 'handler=' argument to SIGNAL() must be a procedure or a scalar INTEGER
  call signal(8, logical_handler)

  !ERROR: 'handler=' argument to SIGNAL() must be a procedure or a scalar INTEGER
  call signal(8, complex_handler)

  !ERROR: 'handler=' argument to SIGNAL() must be a procedure or a scalar INTEGER
  call signal(8, character_handler)
end subroutine

subroutine test_implicit_real_handler
  ! No IMPLICIT NONE here intentionally.
  ! HANDLER_FFL begins with H and is therefore implicitly REAL.

  !ERROR: 'handler=' argument to SIGNAL() must be a procedure or a scalar INTEGER
  call signal(8, handler_ffl)
end subroutine

subroutine test_number
  external :: signal_handler

  integer :: number
  integer :: number_array(2)
  real :: real_number
  logical :: logical_number
  character :: character_number

  ! Valid scalar INTEGER number.
  call signal(8, signal_handler)
  call signal(number, signal_handler)

  ! Invalid type.
  !ERROR: Actual argument for 'number=' has bad type 'REAL(4)'
  call signal(real_number, signal_handler)

  !ERROR: Actual argument for 'number=' has bad type 'LOGICAL(4)'
  call signal(logical_number, signal_handler)

  !ERROR: Actual argument for 'number=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call signal(character_number, signal_handler)

  ! Invalid rank.
  !ERROR: 'number=' argument has unacceptable rank 1
  call signal(number_array, signal_handler)
end subroutine

subroutine test_status
  external :: signal_handler

  integer :: status
  integer :: status_array(2)
  integer, parameter :: status_parameter = 0
  real :: real_status
  logical :: logical_status

  ! STATUS is optional.
  call signal(8, signal_handler)

  ! Valid scalar INTEGER STATUS.
  call signal(8, signal_handler, status)

  ! Invalid type.
  !ERROR: Actual argument for 'status=' has bad type 'REAL(4)'
  call signal(8, signal_handler, real_status)

  !ERROR: Actual argument for 'status=' has bad type 'LOGICAL(4)'
  call signal(8, signal_handler, logical_status)

  ! Invalid rank.
  !ERROR: 'status=' argument has unacceptable rank 1
  call signal(8, signal_handler, status_array)

  ! STATUS has INTENT(OUT), so the actual argument must be definable.
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'status=' is not definable
  !BECAUSE: '0_4' is not a variable or pointer
  call signal(8, signal_handler, 0)

  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'status=' is not definable
  !BECAUSE: '0_4' is not a variable or pointer
  call signal(8, signal_handler, status_parameter)
end subroutine

