! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in this_image() function calls

program this_image_tests
  use iso_fortran_env, only : team_type
  implicit none

  !ERROR: Coarray 'team_coarray' may not have type TEAM_TYPE, C_PTR, or C_FUNPTR
  type(team_type) team_coarray[*]
  type(team_type) home, league(2)
  integer n, i, array(1), non_coarray(1), co_array[*]
  integer, allocatable :: images(:)
  logical non_integer

  !___ standard-conforming statement with no optional arguments present ___
  n = this_image()

  !___ standard-conforming statements with team argument present ___
  n = this_image(home)
  n = this_image(team=home)
  n = this_image(league(1))

  !___ standard-conforming statements with coarray argument present ___
  images = this_image(co_array)
  images = this_image(coarray=co_array)

  !___ standard-conforming statements with coarray and team arguments present ___
  images = this_image(co_array, home)
  images = this_image(co_array, team=home)
  images = this_image(team_coarray, team=home)
  images = this_image(team_coarray[1], team=home)
  images = this_image(coarray=co_array, team=home)
  images = this_image(team=home, coarray=co_array)

  !___ standard-conforming statements with coarray and dim arguments present ___
  n = this_image(co_array, i)
  n = this_image(co_array, dim=i)
  n = this_image(coarray=co_array, dim=i)
  n = this_image(dim=i, coarray=co_array)

  !___ standard-conforming statements with all arguments present ___
  n = this_image(co_array, i, home)
  n = this_image(co_array, i, team=home)
  n = this_image(co_array, dim=i, team=home)
  n = this_image(co_array, team=home, dim=i)

  n = this_image(coarray=co_array, dim=i, team=home)
  n = this_image(coarray=co_array, team=home, dim=i)

  n = this_image(dim=i, coarray=co_array, team=home)
  n = this_image(dim=i, team=home, coarray=co_array)

  n = this_image(team=home, dim=i, coarray=co_array)
  n = this_image(team=home, coarray=co_array, dim=i)

  !___ non-conforming statements ___

  !ERROR: TBD
  n = this_image(co_array)

  !ERROR: missing mandatory 'dim=' argument
  n = this_image(i)

  !ERROR: missing mandatory 'coarray=' argument
  n = this_image(dim=i)

  !ERROR: Actual argument for 'dim=' has bad type 'team_type'
  n = this_image(i, home)

  !ERROR: missing mandatory 'dim=' argument
  n = this_image(i, team=home)

  !ERROR: TBD
  n = this_image(coarray=co_array, dim=2)

  !ERROR: missing mandatory 'coarray=' argument
  n = this_image(dim=i, team=home)

  !ERROR: missing mandatory 'coarray=' argument
  n = this_image(team=home, dim=i)

  ! Doesn't produce an error
  n = this_image(coarray=co_array, i)

  !ERROR: No explicit type declared for 'team'
  images = this_image(coarray=co_array, team)

  ! non-scalar team_type argument 
  !ERROR: missing mandatory 'coarray=' argument
  n = this_image(team=league)

  ! incorrectly typed argument
  !ERROR: missing mandatory 'dim=' argument
  n = this_image(3.4)

  !ERROR: too many actual arguments for intrinsic 'this_image'
  n = this_image(co_array, i, home, 0)

  ! keyword argument with incorrect type
  !ERROR: missing mandatory 'dim=' argument
  images = this_image(coarray=non_coarray)

  ! incorrect keyword argument name but valid type (type number)
  !ERROR: unknown keyword argument to intrinsic 'this_image'
  images = this_image(co_array=co_array)

  ! incorrect keyword argument name but valid type (team_type)
  !ERROR: unknown keyword argument to intrinsic 'this_image'
  n = this_image(my_team=home)

  ! correct keyword argument name but mismatched type
  !ERROR: Actual argument for 'team=' has bad type 'INTEGER(4)'
  n = this_image(co_array, i, team=-1)

  !ERROR: 'dim=' argument has unacceptable rank 1
  n = this_image(co_array, array )

  !ERROR: unknown keyword argument to intrinsic 'this_image'
  n = this_image(co_array, dims=i)

  !ERROR: Actual argument for 'dim=' has bad type 'LOGICAL(4)'
  n = this_image(co_array, non_integer)

  ! A this_image reference with a coarray argument of team type shall also have a team argument
  ! Doesn't produce an error
  images = this_image(team_coarray)
end program this_image_tests
