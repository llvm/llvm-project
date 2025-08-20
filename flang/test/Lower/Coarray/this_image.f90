! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env
  integer :: i
  type(team_type) :: team

  ! CHECK: fir.call @_QMprifPprif_this_image_no_coarray
  i = this_image()

  ! CHECK: fir.call @_QMprifPprif_this_image_no_coarray
  i = this_image(TEAM=team)

end program
