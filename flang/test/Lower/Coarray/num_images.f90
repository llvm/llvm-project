! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env
  integer :: i
  integer :: team_number 
  type(team_type) :: team

  ! CHECK: fir.call @_QMprifPprif_num_images
  i = num_images()

  ! CHECK: fir.call @_QMprifPprif_num_images_with_team_number
  i = num_images(TEAM_NUMBER=team_number)

  ! CHECK: fir.call @_QMprifPprif_num_images_with_team
  i = num_images(TEAM=team)

end program
