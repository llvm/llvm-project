! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in this_image() function calls

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team
  !ERROR: Coarray 'coteam' may not have type TEAM_TYPE, C_PTR, or C_FUNPTR
  type(team_type) :: coteam[*]
  integer :: coscalar[*], coarray(3)[*]
  save :: coteam, coscalar, coarray
  real coarray1[*], coarray2[2,*], coarray3[2,3,*]
  integer indices(3)

  ! correct calls, should produce no errors
  team = get_team()
  print *, this_image()
  print *, this_image(team)
  print *, this_image(coarray)
  print *, this_image(coarray, team)
  print *, this_image(coarray, 1)
  print *, this_image(coarray, 1, team)
  print *, this_image(coarray(1))
  print *, this_image(coarray(1), team)
  print *, this_image(coarray(1), 1)
  print *, this_image(coarray(1), 1, team)
  print *, this_image(coscalar)
  print *, this_image(coscalar, team)
  print *, this_image(coscalar, 1)
  print *, this_image(coscalar, 1, team)

  !ERROR: 'coarray=' argument must have corank > 0 for intrinsic 'this_image'
  print *, this_image(array,1)

  print *, team_number()
  print *, team_number(team)

  indices(1:1) = this_image(coarray1) ! ok
  indices(1:2) = this_image(coarray2) ! ok
  indices(1:3) = this_image(coarray3) ! ok
  !ERROR: Dimension 1 of left-hand side has extent 2, but right-hand side has extent 1
  indices(1:2) = this_image(coarray1)
  !ERROR: Dimension 1 of left-hand side has extent 3, but right-hand side has extent 2
  indices(1:3) = this_image(coarray2)
  !ERROR: Dimension 1 of left-hand side has extent 1, but right-hand side has extent 3
  indices(1:1) = this_image(coarray3)

end subroutine
