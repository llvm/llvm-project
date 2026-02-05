! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  integer :: res1(3), res2
  integer, allocatable :: a[:,:,:]

  allocate(a[2,3:5,*])

  ! CHECK: mif.lcobound coarray %[[COARRAY:.*]] : (!fir.heap<i32>) -> !fir.box<!fir.array<?xi64>>
  res1 = lcobound(a)

  ! CHECK: mif.lcobound coarray %[[COARRAY:.*]] dim %[[C2:.*]] : (!fir.heap<i32>, i32) -> i32
  res2 = lcobound(a, DIM=2)

  ! CHECK: mif.ucobound coarray %[[COARRAY:.*]] : (!fir.heap<i32>) -> !fir.box<!fir.array<?xi64>>
  res1 = ucobound(a)

  ! CHECK: mif.ucobound coarray %[[COARRAY:.*]] dim %[[C2:.*]] : (!fir.heap<i32>, i32) -> i32
  res2 = ucobound(a, DIM=2)

end program
