! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in this_image() function calls

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: oregon, coteam[*]
  integer :: coscalar[*], coarray(3)[*]
  save :: coteam, coscalar, coarray

  ! correct calls, should produce no errors
  print *, this_image()
  print *, this_image(coarray)
  print *, this_image(coscalar,1)
  print *, this_image(coarray,1)

  !ERROR: 'coarray=' argument must have corank > 0 for intrinsic 'this_image'
  print *, this_image(array,1)

  ! TODO: More complete testing requires implementation of team_type
  ! actual arguments in flang/lib/Evaluate/intrinsics.cpp

end subroutine
