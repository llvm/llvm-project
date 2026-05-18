! REQUIRES: flang-supports-f128-math
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

function test_real4(y, x)
  real(4) :: x, y, test_real4
  test_real4 = atan2pi(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real4
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f32
! CHECK: %[[inv_pi:.*]] = arith.constant 0.318309873 : f32
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[inv_pi]] fastmath<contract> : f32
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f32, !fir.ref<f32>

function test_real8(y, x)
  real(8) :: x, y, test_real8
  test_real8 = atan2pi(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real8
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f64
! CHECK: %[[inv_pi:.*]] = arith.constant 0.31830988618379069 : f64
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[inv_pi]] fastmath<contract> : f64
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f64, !fir.ref<f64>

function test_real16(y, x)
  real(16) :: x, y, test_real16
  test_real16 = atan2pi(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real16
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f128
! CHECK: %[[inv_pi:.*]] = arith.constant 0.3183098861837906715377675267450{{.*}} : f128
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[inv_pi]] fastmath<contract> : f128
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f128, !fir.ref<f128>
