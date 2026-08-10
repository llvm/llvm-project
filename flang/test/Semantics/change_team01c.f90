! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in change team statements.
! This subtest contains tests for unimplemented errors.

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team
  integer, codimension[*] :: selector

  ! A branch to an END TEAM statement is permitted only from within the corresponding CHANGE TEAM construct.
  change team (team)
    if (.true.) then
      end team
    end if
  end team

  ! A RETURN statement may not appear in a CHANGE TEAM construct.
  change team (team)
    ! ERROR: TBD
    return
  end team

  ! On each image, the team variable specified in the CHANGE TEAM statement cannot become undefined or redefined during execution of the construct.
  ! ERROR: TBD
  change team (team)
    team = get_team(INITIAL_TEAM)
  end team
end subroutine


