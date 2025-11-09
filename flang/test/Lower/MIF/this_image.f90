! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env
  integer :: i
  type(team_type) :: team

  ! CHECK: mif.this_image : () -> i32 
  i = this_image()

  ! CHECK: mif.this_image team %[[TEAM:.*]] : ({{.*}}) -> i32
  i = this_image(TEAM=team)

end program
