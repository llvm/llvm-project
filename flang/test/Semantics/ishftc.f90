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

  !ERROR: SIZE=-3 count for ishftc is not positive
  n = ishftc(3, 2, -3)
  !ERROR: SIZE=0 count for ishftc is not positive
  n = ishftc(3, 2, 0)
  !ERROR: SHIFT=2 count for ishftc is greater in magnitude than SIZE=1
  n = ishftc(3, 2, 1)
  !ERROR: SHIFT=-2 count for ishftc is greater in magnitude than SIZE=1
  n = ishftc(3, -2, 1)
  !ERROR: SHIFT=4 count for ishftc is greater in magnitude than SIZE=3
  array_result = ishftc(666, [(j,integer::j=1,5)], 3)
  !ERROR: SHIFT=4 count for ishftc is greater in magnitude than SIZE=3
  array_result = ishftc(666, 4, [(j,integer::j=10,3,-1)])
  !ERROR: SIZE=-3 count for ishftc is not positive
  array_result = ishftc([3,3], [2,2], [-3,3])
  !ERROR: SIZE=-3 count for ishftc is not positive
  array_result = ishftc([3,3], [2,2], [-3,-3])
  !ERROR: SIZE=-3 count for ishftc is not positive
  array_result = ishftc([3,3], [-2,-2], const_arr1)
  !ERROR: SIZE=0 count for ishftc is not positive
  array_result = ishftc([3,3], [-2,-2], const_arr2)
  !ERROR: SIZE=0 count for ishftc is not positive
  array_result = ishftc([3,3], [-2,-2], const_arr3)
  !ERROR: SIZE=0 count for ishftc is not positive
  array_result = ishftc([3,3], [-2,-2], const_arr4)
  !ERROR: SIZE=0 count for ishftc is not positive
  array_result = ishftc([3,3], [-2,-2], const_arr5)
  !ERROR: SIZE=0 count for ishftc is not positive
  array_result = ishftc([3,3], [-2,-2], const_arr6)
  !ERROR: SIZE=0 count for ishftc is not positive
  array_result = ishftc([3,3], [-2,-2], const_arr7)
  array_result = ishftc([(j,integer::j=1,0)], 10, 9) ! ok because empty
end program test_ishftc
