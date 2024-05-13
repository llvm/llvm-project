! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in failed_images() function calls

program failed_images_test
  use iso_fortran_env, only: team_type
  use iso_c_binding, only: c_int32_t
  implicit none

  type(team_type) home, league(2)
  integer n, i, array(1)
  integer, allocatable :: failure(:)
  integer, allocatable :: wrong_rank(:,:)
  logical non_integer, non_team
  character, allocatable :: wrong_result(:)

  !___ standard-conforming statement with no optional arguments present ___
  failure = failed_images()

  !___ standard-conforming statements with optional team argument present ___
  failure = failed_images(home)
  failure = failed_images(team=home)
  failure = failed_images(league(1))

  !___ standard-conforming statements with optional kind argument present ___
  failure = failed_images(kind=c_int32_t)

  !___ standard-conforming statements with both optional arguments present ___
  failure = failed_images(home, c_int32_t)
  failure = failed_images(team=home, kind=c_int32_t)
  failure = failed_images(kind=c_int32_t, team=home)

  !___ non-conforming statements ___

  !ERROR: Actual argument for 'team=' has bad type 'LOGICAL(4)'
  failure = failed_images(non_team)

  ! non-scalar team_type argument
  !ERROR: 'team=' argument has unacceptable rank 1
  failure = failed_images(league)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  failure = failed_images(team=-1)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  failure = failed_images(team=i, kind=c_int32_t)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  failure = failed_images(i, c_int32_t)

  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  failure = failed_images(c_int32_t)

  ! non constant
  !ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
  failure = failed_images(kind=i)

  ! non integer
  !ERROR: Actual argument for 'kind=' has bad type 'LOGICAL(4)'
  failure = failed_images(home, non_integer)
  !ERROR: Actual argument for 'kind=' has bad type 'LOGICAL(4)'
  failure = failed_images(kind=non_integer)

  ! non-scalar
  !ERROR: 'kind=' argument has unacceptable rank 1
  failure = failed_images(kind=array)

  !ERROR: too many actual arguments for intrinsic 'failed_images'
  failure = failed_images(home, c_int32_t, 3)

  !ERROR: Actual argument for 'team=' has bad type 'REAL(4)'
  failure = failed_images(3.4)

  !ERROR: unknown keyword argument to intrinsic 'failed_images'
  failure = failed_images(kinds=c_int32_t)

  !ERROR: unknown keyword argument to intrinsic 'failed_images'
  failure = failed_images(home, kinds=c_int32_t)

  !ERROR: unknown keyword argument to intrinsic 'failed_images'
  failure = failed_images(my_team=home)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
  n = failed_images()

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches rank 2 array of INTEGER(4) and rank 1 array of INTEGER(4)
  wrong_rank = failed_images()

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types CHARACTER(KIND=1) and INTEGER(4)
  wrong_result = failed_images()

end program failed_images_test
