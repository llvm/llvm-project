! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: dprod_test
subroutine dprod_test (x, y, z)
  real :: x,y
  double precision :: z
  z = dprod(x,y)
  ! CHECK-DAG: %[[x:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[y:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[a:.*]] = fir.convert %[[x]] : (f32) -> f64
  ! CHECK-DAG: %[[b:.*]] = fir.convert %[[y]] : (f32) -> f64
  ! CHECK: %[[res:.*]] = arith.mulf %[[a]], %[[b]]
  ! CHECK: fir.store %[[res]] to %arg2
end subroutine
