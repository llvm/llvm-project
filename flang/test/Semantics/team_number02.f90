! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in team_number() function calls

program team_number_tests
  use iso_fortran_env, only : team_type
  implicit none

  type(team_type) home, league(2)
  integer n, non_team_type
  character non_integer

  !___ standard-conforming statement with no optional arguments present ___
  n = team_number()

  !___ standard-conforming statements with team argument present ___
  n = team_number(home)
  n = team_number(team=home)
  n = team_number(league(1))

  !___ non-conforming statements ___
  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  n = team_number(non_team_type)

  ! non-scalar team_type argument
  !ERROR: 'team=' argument has unacceptable rank 1
  n = team_number(team=league)

  ! incorrectly typed argument
  !ERROR: Actual argument for 'team=' has bad type 'REAL(4)'
  n = team_number(3.4)

  !ERROR: too many actual arguments for intrinsic 'team_number'
  n = team_number(home, league(1))

  !ERROR: repeated keyword argument to intrinsic 'team_number'
  n = team_number(team=home, team=league(1))

  ! keyword argument with incorrect type
  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  n = team_number(team=non_team_type)

  ! incorrect keyword argument name but valid type
  !ERROR: unknown keyword argument to intrinsic 'team_number'
  n = team_number(my_team=home)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types CHARACTER(KIND=1) and INTEGER(4)
  non_integer = team_number(home)

end program team_number_tests
