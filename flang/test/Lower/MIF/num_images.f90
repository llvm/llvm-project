! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env
  integer :: i
  integer :: team_number 
  type(team_type) :: team

  ! CHECK: mif.num_images : () -> i32
  i = num_images()

  ! CHECK: mif.num_images team_number %[[TEAM_NUMBER:.*]] : (i32) -> i32
  i = num_images(TEAM_NUMBER=team_number)

  ! CHECK: mif.num_images team %[[TEAM:.*]]#0 : ({{.*}}) -> i32
  i = num_images(TEAM=team)

end program
