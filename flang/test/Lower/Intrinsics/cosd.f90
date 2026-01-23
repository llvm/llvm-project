! REQUIRES: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-PRECISE"
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"

function test_real4(x)
  real :: x, test_real4
  test_real4 = cosd(x)
end function

! CHECK-LABEL: @_QPtest_real4
! CHECK: %[[factor:.*]] = arith.constant 0.0174532924 : f32
! CHECK: %[[arg:.*]] = arith.mulf %{{[A-Za-z0-9._]+}}, %[[factor]] fastmath<contract> : f32
! CHECK-PRECISE: %{{.*}} = fir.call @cosf(%[[arg]]) fastmath<contract> : (f32) -> f32
! CHECK-FAST: %{{.*}} = math.cos %[[arg]] fastmath<contract> : f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = cosd(x)
end function

! CHECK-LABEL: @_QPtest_real8
! CHECK: %[[factor:.*]] = arith.constant 0.017453292519943295 : f64
! CHECK: %[[arg:.*]] = arith.mulf %{{[A-Za-z0-9._]+}}, %[[factor]] fastmath<contract> : f64
! CHECK-PRECISE: %{{.*}} = fir.call @cos(%[[arg]]) fastmath<contract> : (f64) -> f64
! CHECK-FAST: %{{.*}} = math.cos %[[arg]] fastmath<contract> : f64

function test_real16(x)
  real(16) :: x, test_real16
  test_real16 = cosd(x)
end function

! CHECK-LABEL: @_QPtest_real16
! CHECK: %[[factor:.*]] = arith.constant 0.0174532925199432957692369076848861{{.*}} : f128
! CHECK: %[[arg:.*]] = arith.mulf %{{[A-Za-z0-9._]+}}, %[[factor]] fastmath<contract> : f128
! CHECK: %[[result:.*]] = fir.call @_FortranACosF128({{.*}}) fastmath<contract> : (f128) -> f128
