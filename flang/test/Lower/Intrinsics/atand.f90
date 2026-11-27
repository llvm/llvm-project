! REQUIRES: flang-supports-f128-math
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
! RUN: %flang_fc1 -mllvm --math-runtime=precise -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-PRECISE"

function test_real4(x)
  real :: x, test_real4
  test_real4 = atand(x)
end function

! CHECK-LABEL: func.func @_QPtest_real4
! CHECK-PRECISE: %[[atan:.*]] = fir.call @atanf({{%[A-Za-z0-9._]+}}) fastmath<contract> : (f32) -> f32
! CHECK-FAST: %[[atan:.*]] = math.atan %{{.*}} fastmath<contract> : f32
! CHECK: %[[factor:.*]] = arith.constant 57.2957763 : f32
! CHECK: %[[res:.*]] = arith.mulf %[[atan]], %[[factor]] fastmath<contract> : f32
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f32, !fir.ref<f32>

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = atand(x)
end function

! CHECK-LABEL: func.func @_QPtest_real8
! CHECK-PRECISE: %[[atan:.*]] = fir.call @atan({{%[A-Za-z0-9._]+}}) fastmath<contract> : (f64) -> f64
! CHECK-FAST: %[[atan:.*]] = math.atan %{{.*}} fastmath<contract> : f64
! CHECK: %[[factor:.*]] = arith.constant 57.295779513082323 : f64
! CHECK: %[[res:.*]] = arith.mulf %[[atan]], %[[factor]] fastmath<contract> : f64
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f64, !fir.ref<f64>

function test_real16(x)
  real(16) :: x, test_real16
  test_real16 = atand(x)
end function

! CHECK-LABEL: func.func @_QPtest_real16
! CHECK: %[[atan:.*]] = fir.call @_FortranAAtanF128({{.*}}) fastmath<contract> : (f128) -> f128
! CHECK: %[[factor:.*]] = arith.constant 57.295779513082320876798154814105{{.*}} : f128
! CHECK: %[[res:.*]] = arith.mulf %[[atan]], %[[factor]] fastmath<contract> : f128
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f128, !fir.ref<f128>

function test_real4_yx(y, x)
  real(4) :: x, y, test_real4
  test_real4 = atand(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real4_yx
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f32
! CHECK: %[[factor:.*]] = arith.constant 57.2957763 : f32
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[factor]] fastmath<contract> : f32
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f32, !fir.ref<f32>

function test_real8_yx(y, x)
  real(8) :: x, y, test_real8
  test_real8 = atand(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real8_yx
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f64
! CHECK: %[[factor:.*]] = arith.constant 57.295779513082323 : f64
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[factor]] fastmath<contract> : f64
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f64, !fir.ref<f64>

function test_real16_yx(y, x)
  real(16) :: x, y, test_real16
  test_real16 = atand(y, x)
end function

! CHECK-LABEL: func.func @_QPtest_real16_yx
! CHECK: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}} fastmath<contract> : f128
! CHECK: %[[factor:.*]] = arith.constant 57.295779513082320876798154814105{{.*}} : f128
! CHECK: %[[res:.*]] = arith.mulf %[[atan2]], %[[factor]] fastmath<contract> : f128
! CHECK: hlfir.assign %[[res]] to %{{.*}} : f128, !fir.ref<f128>
