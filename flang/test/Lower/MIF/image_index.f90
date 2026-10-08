! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env, only: team_type
  integer(kind=4) :: sub(3) = (/1, 4, 2/) 
  integer(kind=8) :: sub2(3) = (/1, 4, 2/) 
  integer(kind=4) :: a[2,3:5,*], idx
  type(team_type) :: team
  integer :: team_number

  ! CHECK: mif.image_index coarray %[[COARRAY:.*]] sub %[[SUB:.*]] : (!fir.box<i32, corank:3>, !fir.box<!fir.array<3xi64>>) -> i32
  idx = image_index(a, SUB=sub)

  ! CHECK: mif.image_index coarray %[[COARRAY:.*]] sub %[[SUB2:.*]] : (!fir.box<i32, corank:3>, !fir.box<!fir.array<3xi64>>) -> i32
  idx = image_index(a, SUB=sub2)

  ! CHECK: mif.image_index coarray %[[COARRAY:.*]] sub %[[SUB:.*]] team %[[TEAM:.*]]#0 : (!fir.box<i32, corank:3>,
  ! !fir.box<!fir.array<3xi64>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}}>>) -> i32
  idx = image_index(a, SUB=sub, TEAM=team)

  ! CHECK: mif.image_index coarray %[[COARRAY:.*]] sub %[[SUB:.*]] team_number %[[TEAM_NUMBER:.*]]: (!fir.box<i32, corank:3>, !fir.box<!fir.array<3xi64>>, i32) -> i32
  idx = image_index(a, SUB=sub, TEAM_NUMBER=team_number)

end program
