! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env, only: team_type
  integer :: i, j(2)
  type(team_type) :: team
  integer :: a[2,*]

  ! CHECK: mif.this_image : () -> i32
  i = this_image()

  ! CHECK: mif.this_image team %[[TEAM:.*]] : ({{.*}}) -> i32
  i = this_image(TEAM=team)

  ! CHECK: mif.this_image coarray %[[A:.*]] : (!fir.box<i32, corank:2>) -> !fir.box<!fir.array<?xi64>>
  j = this_image(COARRAY=a)
  
  ! CHECK: mif.this_image coarray %[[A:.*]] dim %[[DIM:.*]] : (!fir.box<i32, corank:2>, i32) -> i64
  j = this_image(COARRAY=a, DIM=1)
end program
