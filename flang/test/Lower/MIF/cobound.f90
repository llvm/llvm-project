! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test
  integer :: res1(3), res2
  integer, allocatable :: a[:,:,:]

  allocate(a[2,3:5,*])

  ! CHECK: mif.lcobound coarray %[[COARRAY:.*]] dim %[[C1_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32 
  ! CHECK: mif.lcobound coarray %[[COARRAY]] dim %[[C2_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32 
  ! CHECK: mif.lcobound coarray %[[COARRAY]] dim %[[C3_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32 
  res1 = lcobound(a)

  ! CHECK: mif.lcobound coarray %[[COARRAY:.*]] dim %[[C2_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
  res2 = lcobound(a, DIM=2)

  ! CHECK: mif.ucobound coarray %[[COARRAY:.*]] dim %[[C1_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32 
  ! CHECK: mif.ucobound coarray %[[COARRAY]] dim %[[C2_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32 
  ! CHECK: mif.ucobound coarray %[[COARRAY]] dim %[[C3_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32 
  res1 = ucobound(a)

  ! CHECK: mif.ucobound coarray %[[COARRAY:.*]] dim %[[C2_I32:.*]] : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
  res2 = ucobound(a, DIM=2)

end program
