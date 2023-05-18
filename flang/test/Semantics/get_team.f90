! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in get_team(), as defined in
! section 16.9.85 of the Fortran 2018 standard

program get_team_test
  use iso_fortran_env, only: team_type, initial_team, current_team, parent_team
  implicit none

  integer n, array(1), coarray[*]
  type(team_type) :: result_team
  logical wrong_result_type, non_integer

  !___ standard-conforming statement with no optional arguments present ___
  result_team = get_team()

  !___ standard-conforming statements with optional level argument present ___
  result_team = get_team(-1)
  result_team = get_team(-2)
  result_team = get_team(-3)
  result_team = get_team(initial_team)
  result_team = get_team(current_team)
  result_team = get_team(parent_team)
  result_team = get_team(n)
  result_team = get_team(array(1))
  result_team = get_team(array(n))
  result_team = get_team(coarray[1])
  result_team = get_team(level=initial_team)
  result_team = get_team(level=n)

  !___ non-conforming statements ___
  !ERROR: 'level=' argument has unacceptable rank 1
  result_team = get_team(array)

  !ERROR: Actual argument for 'level=' has bad type 'LOGICAL(4)'
  result_team = get_team(non_integer)

  !ERROR: Actual argument for 'level=' has bad type 'REAL(4)'
  result_team = get_team(3.4)

  !ERROR: too many actual arguments for intrinsic 'get_team'
  result_team = get_team(current_team, parent_team)

  !ERROR: Actual argument for 'level=' has bad type 'REAL(4)'
  result_team = get_team(level=3.4)

  !ERROR: unknown keyword argument to intrinsic 'get_team'
  result_team = get_team(levels=initial_team)

  !ERROR: repeated keyword argument to intrinsic 'get_team'
  result_team = get_team(level=initial_team, level=parent_team)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types LOGICAL(4) and TYPE(__builtin_team_type)
  wrong_result_type = get_team()

end program get_team_test
