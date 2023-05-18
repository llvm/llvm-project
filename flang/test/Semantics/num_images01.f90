! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in num_images() function calls

subroutine test
  use iso_fortran_env, only: team_type
  implicit none

  type(team_type) my_team

  ! correct calls, should produce no errors
  print *, num_images()
  print *, num_images(team_number=1)
  print *, num_images(1)
  print *, num_images(my_team)
  print *, num_images(team=my_team)

  ! incorrectly typed argument
  ! the error is seen as too many arguments to the num_images() call with no arguments
  !ERROR: too many actual arguments for intrinsic 'num_images'
  print *, num_images(3.4)

  ! call with too many arguments
  !ERROR: too many actual arguments for intrinsic 'num_images'
  print *, num_images(1, 1)

  ! keyword argument with incorrect type
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  print *, num_images(team_number=3.4)

  ! incorrect keyword argument
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  print *, num_images(team_numbers=1)

  !ERROR: unknown keyword argument to intrinsic 'num_images'
  print *, num_images(teams=my_team)

end subroutine
