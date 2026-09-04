! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

function test_real4(y, x)
  real(4) :: x, y, test_real4
  test_real4 = atan2d(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real4
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f32
! CHECK: %[[factor:.*]] = arith.constant 57.2957763 : f32
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[factor]] fastmath<contract> : f32
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f32, !fir.ref<f32>

function test_real8(y, x)
  real(8) :: x, y, test_real8
  test_real8 = atan2d(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real8
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f64
! CHECK: %[[factor:.*]] = arith.constant 57.295779513082323 : f64
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[factor]] fastmath<contract> : f64
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f64, !fir.ref<f64>
