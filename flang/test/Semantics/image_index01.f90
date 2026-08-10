! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure standard-conforming image_index function references are
! accepted, based on the 16.9.107 section of the Fortran 2023 standard

program image_index_test
  use iso_fortran_env, only: team_type
  implicit none

  integer n, array(1), team_num
  integer scalar_coarray[*], array_coarray(1)[*], coarray_corank3[10, 0:9, 0:*]
  integer subscripts_corank1(1), subscripts_corank3(3)
  type(team_type) :: home, league(2)

  !___ standard-conforming statements - IMAGE_INDEX(COARRAY, SUB) ___
  n = image_index(scalar_coarray, [1])
  n = image_index(scalar_coarray, subscripts_corank1)
  n = image_index(array_coarray, [1])
  n = image_index(array_coarray, subscripts_corank1)
  n = image_index(coarray=scalar_coarray, sub=subscripts_corank1)
  n = image_index(coarray_corank3, subscripts_corank3)
  n = image_index(sub=subscripts_corank1, coarray=scalar_coarray)

  !___ standard-conforming statements - IMAGE_INDEX(COARRAY, SUB, TEAM) ___
  n = image_index(scalar_coarray, [1], home)
  n = image_index(scalar_coarray, subscripts_corank1, league(1))
  n = image_index(array_coarray, [1], home)
  n = image_index(array_coarray, subscripts_corank1, league(1))
  n = image_index(coarray_corank3, subscripts_corank3, league(1))
  n = image_index(coarray=scalar_coarray, sub=subscripts_corank1, team=home)
  n = image_index(team=home, sub=[1], coarray=scalar_coarray)

  !___ standard-conforming statements - IMAGE_INDEX(COARRAY, SUB, TEAM_NUMBER) ___
  n = image_index(scalar_coarray, [1], team_num)
  n = image_index(scalar_coarray, subscripts_corank1, team_number=team_num)
  n = image_index(array_coarray, [1], team_num)
  n = image_index(array_coarray, subscripts_corank1, array(1))
  n = image_index(coarray_corank3, subscripts_corank3, team_num)
  n = image_index(coarray=scalar_coarray, sub=subscripts_corank1, team_number=team_num)
  n = image_index(team_number=team_num, sub=[1], coarray=scalar_coarray)

end program image_index_test
