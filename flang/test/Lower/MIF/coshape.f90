! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  integer :: res(3)
  integer(kind=8) :: res2(3)
  integer :: res3(3)
  integer :: a[2,3:5,*]

  ! COSHAPE without KIND returns default integer kind (4 = i32).
  ! CHECK: mif.coshape coarray %[[COARRAY:.*]] : (!fir.box<i32, corank:3>) -> !fir.box<!fir.array<?xi32>>
  res = coshape(a)

  ! Assignment to integer(kind=8) widens; COSHAPE result type is still i32.
  ! CHECK: mif.coshape coarray %[[COARRAY:.*]] : (!fir.box<i32, corank:3>) -> !fir.box<!fir.array<?xi32>>
  res2 = coshape(a)

  ! Explicit KIND=8 yields i64 elements.
  ! CHECK: mif.coshape coarray %[[COARRAY:.*]] : (!fir.box<i32, corank:3>) -> !fir.box<!fir.array<?xi64>>
  res3 = coshape(a, kind=8)

end program
