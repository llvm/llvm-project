! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in team_number() function calls

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team

  ! correct calls, should produce no errors
  team = get_team()
  print *, team_number()
  print *, team_number(team)
  print *, team_number(team=team)

  ! call with too many arguments
  !ERROR: too many actual arguments for intrinsic 'team_number'
  print *, team_number(1, 3)

  ! keyword argument with incorrect type
  !ERROR: Actual argument for 'team=' has bad type 'REAL(4)'
  print *, team_number(team=3.1415)

end subroutine
