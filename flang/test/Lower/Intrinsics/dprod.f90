! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: dprod_test
subroutine dprod_test (x, y, z)
  real :: x,y
  double precision :: z
  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %arg0 dummy_scope %[[DS]] {{.*}}
  ! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %arg1 dummy_scope %[[DS]] {{.*}}
  ! CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %arg2 dummy_scope %[[DS]] {{.*}}
  ! CHECK-DAG: %[[x:.*]] = fir.load %[[X_DECL]]#0
  ! CHECK-DAG: %[[y:.*]] = fir.load %[[Y_DECL]]#0
  ! CHECK-DAG: %[[a:.*]] = fir.convert %[[x]] : (f32) -> f64
  ! CHECK-DAG: %[[b:.*]] = fir.convert %[[y]] : (f32) -> f64
  ! CHECK: %[[res:.*]] = arith.mulf %[[a]], %[[b]]
  ! CHECK: hlfir.assign %[[res]] to %[[Z_DECL]]#0
  z = dprod(x,y)
end subroutine
