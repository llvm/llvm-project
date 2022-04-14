! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in num_images() function calls

program num_images_with_team_type
  use iso_fortran_env, only : team_type
  implicit none

  type(team_type) home, league(2)
  integer n
  integer :: standard_initial_value = -1
  integer coindexed[*]
  integer array(1)

  !___ standard-conforming statement with no optional arguments present ___
  n = num_images()

  !___ standard-conforming statements with team_number argument present ___
  n = num_images(-1)
  n = num_images(team_number = -1)
  n = num_images(team_number = standard_initial_value)
  n = num_images(standard_initial_value)
  n = num_images(coindexed[1])

  !___ standard-conforming statements with team_type argument present (not yet supported) ___

  !ERROR: too many actual arguments for intrinsic 'num_images'
  n = num_images(home)

  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(team=home)

  !___ non-conforming statements ___

  ! non-scalar integer argument 
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(team_number=array)

  ! non-scalar team_type argument 
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(team=league)

  ! incorrectly typed argument
  !ERROR: too many actual arguments for intrinsic 'num_images'
  n = num_images(3.4)

  !ERROR: too many actual arguments for intrinsic 'num_images'
  n = num_images(1, -1)

  !ERROR: too many actual arguments for intrinsic 'num_images'
  n = num_images(home, standard_initial_value)

  ! keyword argument with incorrect type
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(team_number=1.1)

  ! incorrect keyword argument name but valid type (type number)
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(team_num=-1)

  ! incorrect keyword argument name but valid type (team_type)
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(my_team=home)

  ! correct keyword argument name but mismatched type
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(team=-1)

  ! correct keyword argument name but mismatched type
  !ERROR: unknown keyword argument to intrinsic 'num_images'
  n = num_images(team_number=home)

end program num_images_with_team_type
