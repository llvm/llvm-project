! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in image_status(), as defined in
! section 16.9.98 of the Fortran 2018 standard

program test_image_status
  use iso_fortran_env, only : team_type, stat_failed_image, stat_stopped_image
  implicit none

  type(team_type) home, league(2)
  integer n, image_num, array(5), coindexed[*], non_array_result, array_2d(10, 10), not_team_type
  integer, parameter :: array_with_negative(2) = [-2, 1]
  integer, parameter :: array_with_zero(2) = [1, 0]
  integer, parameter :: constant_integer = 2, constant_negative = -4, constant_zero = 0
  integer, allocatable :: result_array(:), result_array_2d(:,:), wrong_rank_result(:)
  logical wrong_arg_type_logical
  real wrong_arg_type_real
  character wrong_result_type

  !___ standard-conforming statements ___
  n = image_status(1)
  n = image_status(constant_integer)
  n = image_status(image_num)
  n = image_status(array(1))
  n = image_status(coindexed[1])
  n = image_status(image=1)
  result_array = image_status(array)
  result_array_2d = image_status(array_2d)

  n = image_status(2, home)
  n = image_status(2, league(1))
  n = image_status(image=2, team=home)
  n = image_status(team=home, image=2)

  if (image_status(1) .eq. stat_failed_image .or. image_status(1) .eq. stat_stopped_image) then
     error stop
  else if (image_status(1) .eq. 0) then
     continue
  end if

  !___ non-conforming statements ___

  !ERROR: 'image=' argument for intrinsic 'image_status' must be a positive value, but is -1
  n = image_status(-1)

  !ERROR: 'image=' argument for intrinsic 'image_status' must be a positive value, but is 0
  n = image_status(0)

  !ERROR: 'image=' argument for intrinsic 'image_status' must be a positive value, but is -4
  n = image_status(constant_negative)

  !ERROR: 'image=' argument for intrinsic 'image_status' must be a positive value, but is 0
  n = image_status(constant_zero)

  !ERROR: 'team=' argument has unacceptable rank 1
  n = image_status(1, team=league)

  !ERROR: Actual argument for 'image=' has bad type 'REAL(4)'
  n = image_status(3.4)

  !ERROR: Actual argument for 'image=' has bad type 'LOGICAL(4)'
  n = image_status(wrong_arg_type_logical)

  !ERROR: Actual argument for 'image=' has bad type 'REAL(4)'
  n = image_status(wrong_arg_type_real)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  n = image_status(1, not_team_type)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  n = image_status(1, 1)

  !ERROR: Actual argument for 'image=' has bad type 'REAL(4)'
  n = image_status(image=3.4)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  n = image_status(1, team=1)

  !ERROR: too many actual arguments for intrinsic 'image_status'
  n = image_status(1, home, 2)

  !ERROR: repeated keyword argument to intrinsic 'image_status'
  n = image_status(image=1, image=2)

  !ERROR: repeated keyword argument to intrinsic 'image_status'
  n = image_status(image=1, team=home, team=league(1))

  !ERROR: unknown keyword argument to intrinsic 'image_status'
  n = image_status(images=1)

  !ERROR: unknown keyword argument to intrinsic 'image_status'
  n = image_status(1, my_team=home)

  !ERROR: 'image=' argument for intrinsic 'image_status' must contain all positive values
  result_array = image_status(image=array_with_negative)

  !ERROR: 'image=' argument for intrinsic 'image_status' must contain all positive values
  result_array = image_status(image=[-2, 1])

  !ERROR: 'image=' argument for intrinsic 'image_status' must contain all positive values
  result_array = image_status(image=array_with_zero)

  !ERROR: 'image=' argument for intrinsic 'image_status' must contain all positive values
  result_array = image_status(image=[1, 0])

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
  non_array_result = image_status(image=array)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches rank 1 array of INTEGER(4) and rank 2 array of INTEGER(4)
  wrong_rank_result = image_status(array_2d)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types CHARACTER(KIND=1) and INTEGER(4)
  wrong_result_type = image_status(1)

end program test_image_status
