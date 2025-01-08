! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in image_index() function references
! based on the 16.9.107 section of the Fortran 2023 standard

program image_index_test
  use iso_c_binding, only: c_int32_t
  use iso_fortran_env, only: team_type
  implicit none

  integer n, array(1), non_coarray, scalar, team_num
  integer scalar_coarray[*], array_coarray(1)[*], coarray_corank3[10, 0:9, 0:*], repeated_coarray[*]
  integer subscripts_corank1(1), subscripts_corank3(3), repeated_sub(1), multi_rank_array(3,3)
  integer, parameter :: const_subscripts_corank1(1) = [1]
  logical non_integer_array(1)
  type(team_type) :: home, league(2), wrong_result_type

  !___ non-conforming statements ___

  !ERROR: missing mandatory 'coarray=' argument
  n = image_index()

  !ERROR: missing mandatory 'sub=' argument
  n = image_index(scalar_coarray)

  !ERROR: 'sub=' argument has unacceptable rank 2
  n = image_index(scalar_coarray, multi_rank_array)

  !ERROR: The size of 'SUB=' (1) for intrinsic 'image_index' must be equal to the corank of 'COARRAY=' (3)
  n = image_index(coarray_corank3, subscripts_corank1, league(1))

  !ERROR: The size of 'SUB=' (1) for intrinsic 'image_index' must be equal to the corank of 'COARRAY=' (3)
  n = image_index(coarray_corank3, const_subscripts_corank1, league(1))

  !ERROR: The size of 'SUB=' (1) for intrinsic 'image_index' must be equal to the corank of 'COARRAY=' (3)
  n = image_index(coarray_corank3, [1], league(1))

  !ERROR: The size of 'SUB=' (6) for intrinsic 'image_index' must be equal to the corank of 'COARRAY=' (3)
  n = image_index(coarray_corank3, [1,2,3,4,5,6])

  !ERROR: missing mandatory 'coarray=' argument
  n = image_index(sub=[1])

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(team=home)

  !ERROR: 'coarray=' argument must have corank > 0 for intrinsic 'image_index'
  n = image_index(non_coarray, [1])

  !ERROR: Actual argument for 'sub=' has bad type 'LOGICAL(4)'
  n = image_index(array_coarray, [.true.])

  !ERROR: Actual argument for 'sub=' has bad type 'LOGICAL(4)'
  n = image_index(array_coarray, non_integer_array)

  !ERROR: 'sub=' argument has unacceptable rank 0
  n = image_index(array_coarray, scalar)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, subscripts_corank1, team=league)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, [1], team=team_num)

  !ERROR: too many actual arguments for intrinsic 'image_index'
  n = image_index(array_coarray, [1], home, team_num)

  !ERROR: too many actual arguments for intrinsic 'image_index'
  n = image_index(array_coarray, [1], home, team_num)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(array_coarray, [1], team=home, team=league(1))

  !ERROR: repeated keyword argument to intrinsic 'image_index'
  n = image_index(coarray=scalar_coarray, sub=[1], coarray=repeated_coarray)

  !ERROR: keyword argument to intrinsic 'image_index' was supplied positionally by an earlier actual argument
  n = image_index(scalar_coarray, [1], coarray=repeated_coarray)

  !ERROR: repeated keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, sub=subscripts_corank1, sub=repeated_sub)

  !ERROR: keyword argument to intrinsic 'image_index' was supplied positionally by an earlier actual argument
  n = image_index(scalar_coarray, subscripts_corank1, sub=repeated_sub)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, [1], team_number=array)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, [1], team_number=home)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(array_coarray, [1], team=home, team_number=team_num)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(c=scalar_coarray, [1])

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, subscripts=[1])

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, [1], team_num=team_num)

  !ERROR: unknown keyword argument to intrinsic 'image_index'
  n = image_index(scalar_coarray, [1], teams=home)

  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(team_type) and INTEGER(4)
  wrong_result_type = image_index(scalar_coarray, subscripts_corank1)

end program image_index_test
