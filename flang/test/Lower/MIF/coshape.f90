! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  integer :: res(3)
  integer(kind=8) :: res2(3)
  integer :: a[2,3:5,*]

  ! CHECK: mif.coshape coarray %[[COARRAY:.*]] : (!fir.box<i32, corank:3>) -> !fir.box<!fir.array<?xi64>>
  res = coshape(a)

  ! CHECK: mif.coshape coarray %[[COARRAY:.*]] : (!fir.box<i32, corank:3>) -> !fir.box<!fir.array<?xi64>>
  res2 = coshape(a)

end program
