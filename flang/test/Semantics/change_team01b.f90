! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in change team statements
! This subtest contains syntactic tests that prevent the main tests from being emitted.

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team

  ! If a construct name appears on the CHANGE TEAM statement of the construct, the same name must also appear on the END TEAM construct.
  block
  construct: change team (team)
  ! ERROR: CHANGE TEAM construct name required but missing
  end team
  end block
  ! If a construct name appears on an END TEAM statement, the same construct name must appear on the corresponding CHANGE TEAM statement.
  block
  change team (team)
  ! ERROR: CHANGE TEAM construct name unexpected
  end team construct
  end block
  block
  construct1: change team (team)
  ! ERROR: CHANGE TEAM construct name mismatch
  end team construct2
  end block
end subroutine


