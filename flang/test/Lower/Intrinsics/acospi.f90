! REQUIRES: flang-supports-f128-math
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
! RUN: %flang_fc1 -emit-hlfir -mllvm --math-runtime=precise %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-PRECISE"

function test_real4(x)
  real :: x, test_real4
  test_real4 = acospi(x)
end function

! CHECK-LABEL: func.func @_QPtest_real4(
! CHECK-PRECISE: %[[acos:.*]] = fir.call @acosf({{%[A-Za-z0-9._]+}}) fastmath<contract> : (f32) -> f32
! CHECK-FAST: %[[acos:.*]] = math.acos %{{.*}} fastmath<contract> : f32
! CHECK: %[[inv_pi:.*]] = arith.constant 0.318309873 : f32
! CHECK: %{{.*}} = arith.mulf %[[acos]], %[[inv_pi]] fastmath<contract> : f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = acospi(x)
end function

! CHECK-LABEL: func.func @_QPtest_real8(
! CHECK-PRECISE: %[[acos:.*]] = fir.call @acos({{%[A-Za-z0-9._]+}}) fastmath<contract> : (f64) -> f64
! CHECK-FAST: %[[acos:.*]] = math.acos %{{.*}} fastmath<contract> : f64
! CHECK: %[[inv_pi:.*]] = arith.constant 0.31830988618379069 : f64
! CHECK: %{{.*}} = arith.mulf %[[acos]], %[[inv_pi]] fastmath<contract> : f64

function test_real16(x)
  real(16) :: x, test_real16
  test_real16 = acospi(x)
end function

! CHECK-LABEL: func.func @_QPtest_real16(
! CHECK: %[[acos:.*]] = fir.call @_FortranAAcosF128({{.*}}) fastmath<contract> : (f128) -> f128
! CHECK: %[[inv_pi:.*]] = arith.constant 0.3183098861837906715377675267450{{.*}} : f128
! CHECK: %{{.*}} = arith.mulf %[[acos]], %[[inv_pi]] fastmath<contract> : f128
