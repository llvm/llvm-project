! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  use iso_fortran_env, only: team_type
  integer(kind=4) :: sub(3) = (/1, 4, 2/) 
  integer(kind=8) :: sub2(3) = (/1, 4, 2/) 
  integer(kind=4) :: a[2,3:5,*], idx
  type(team_type) :: team
  integer :: team_number

  ! CHECK: mif.image_index coarray %[[COARRAY:.*]]#0 sub %[[SUB:.*]] : (!fir.ref<i32>, !fir.box<!fir.array<3xi32>>) -> i32
  idx = image_index(a, SUB=sub)

  ! CHECK: mif.image_index coarray %[[COARRAY:.*]]#0 sub %[[SUB2:.*]] : (!fir.ref<i32>, !fir.box<!fir.array<3xi64>>) -> i32
  idx = image_index(a, SUB=sub2)

  ! CHECK: mif.image_index coarray %[[COARRAY:.*]]#0 sub %[[SUB2:.*]] team %[[TEAM:.*]]#0 : (!fir.ref<i32>, !fir.box<!fir.array<3xi32>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
  idx = image_index(a, SUB=sub, TEAM=team)

end program
