! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ishftc() function calls

program test_ishftc
  use iso_fortran_env, only: int8, int16, int32, int64
  implicit none

  integer :: n
  integer, allocatable :: array_result(:)
  integer, parameter :: const_arr1(2) = [-3,3]
  integer, parameter :: const_arr2(2) = [3,0]
  integer(kind=8), parameter :: const_arr3(2) = [0,4]
  integer(kind=int8), parameter :: const_arr4(2) = [0,4]
  integer(kind=int16), parameter :: const_arr5(2) = [0,4]
  integer(kind=int32), parameter :: const_arr6(2) = [0,4]
  integer(kind=int64), parameter :: const_arr7(2) = [0,4]

  n = ishftc(3, 2, 3)
  array_result = ishftc([3,3], [2,2], [3,3])

  !ERROR: 'size=' argument for intrinsic 'ishftc' must be a positive value, but is -3
  n = ishftc(3, 2, -3)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must be a positive value, but is 0
  n = ishftc(3, 2, 0)
  !ERROR: The absolute value of the 'shift=' argument for intrinsic 'ishftc' must be less than or equal to the 'size=' argument
  n = ishftc(3, 2, 1)
  !ERROR: The absolute value of the 'shift=' argument for intrinsic 'ishftc' must be less than or equal to the 'size=' argument
  n = ishftc(3, -2, 1)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [2,2], [-3,3])
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [2,2], [-3,-3])
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [-2,-2], const_arr1)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [-2,-2], const_arr2)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [-2,-2], const_arr3)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [-2,-2], const_arr4)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [-2,-2], const_arr5)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [-2,-2], const_arr6)
  !ERROR: 'size=' argument for intrinsic 'ishftc' must contain all positive values
  array_result = ishftc([3,3], [-2,-2], const_arr7)

end program test_ishftc
