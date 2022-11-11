! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in stopped_images() function calls
! as defined in 16.9.183 in the Fortran 2018 standard

program stopped_images_test
  use iso_fortran_env, only: team_type
  use iso_c_binding, only: c_int32_t, c_int64_t
  implicit none

  type(team_type) home, league(2)
  integer n, i, array(1), non_constant
  integer, allocatable :: stopped(:)
  integer, allocatable :: wrong_rank(:,:)
  logical non_integer, non_team
  character, allocatable :: wrong_result(:)

  !___ standard-conforming statement with no optional arguments present ___
  stopped = stopped_images()

  !___ standard-conforming statements with optional team argument present ___
  stopped = stopped_images(home)
  stopped = stopped_images(team=home)
  stopped = stopped_images(league(1))

  !___ standard-conforming statements with optional kind argument present ___
  stopped = stopped_images(kind=c_int32_t)

  !___ standard-conforming statements with both optional arguments present ___
  stopped = stopped_images(home, c_int32_t)
  stopped = stopped_images(team=home, kind=c_int32_t)
  stopped = stopped_images(kind=c_int32_t, team=home)

  !___ non-conforming statements ___

  !ERROR: Actual argument for 'team=' has bad type 'LOGICAL(4)'
  stopped = stopped_images(non_team)

  !ERROR: 'team=' argument has unacceptable rank 1
  stopped = stopped_images(league)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  stopped = stopped_images(team=-1)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  stopped = stopped_images(team=i, kind=c_int32_t)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  stopped = stopped_images(i, c_int32_t)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  stopped = stopped_images(c_int32_t)

  !ERROR: repeated keyword argument to intrinsic 'stopped_images'
  stopped = stopped_images(team=home, team=league(1))

  !ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
  stopped = stopped_images(kind=non_constant)

  !ERROR: Actual argument for 'kind=' has bad type 'LOGICAL(4)'
  stopped = stopped_images(home, non_integer)

  !ERROR: Actual argument for 'kind=' has bad type 'LOGICAL(4)'
  stopped = stopped_images(kind=non_integer)

  !ERROR: 'kind=' argument has unacceptable rank 1
  stopped = stopped_images(kind=array)

  !ERROR: repeated keyword argument to intrinsic 'stopped_images'
  stopped = stopped_images(kind=c_int32_t, kind=c_int64_t)

  !ERROR: too many actual arguments for intrinsic 'stopped_images'
  stopped = stopped_images(home, c_int32_t, 3)

  !ERROR: Actual argument for 'team=' has bad type 'REAL(4)'
  stopped = stopped_images(3.4)

  !ERROR: unknown keyword argument to intrinsic 'stopped_images'
  stopped = stopped_images(kinds=c_int32_t)

  !ERROR: unknown keyword argument to intrinsic 'stopped_images'
  stopped = stopped_images(home, kinds=c_int32_t)

  !ERROR: unknown keyword argument to intrinsic 'stopped_images'
  stopped = stopped_images(my_team=home)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
  n = stopped_images()

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches rank 2 array of INTEGER(4) and rank 1 array of INTEGER(4)
  wrong_rank = stopped_images()

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types CHARACTER(KIND=1) and INTEGER(4)
  wrong_result = stopped_images()

end program stopped_images_test
