! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in form team statements
! This subtest contains syntactic tests that prevent the main tests from being emitted.

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team
  integer :: team_number

  ! Syntactically invalid invocations.
  !ERROR: expected '('
  FORM TEAM (team_number, 0)
  !ERROR: expected '('
  FORM TEAM (team_number, team, STAT=0)
  !ERROR: expected '('
  FORM TEAM (team_number, team, ERRMSG='')
end subroutine
