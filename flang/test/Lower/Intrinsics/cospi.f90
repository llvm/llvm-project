! REQUIRES: flang-supports-f128-math
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK"

function test_real4(x)
  real :: x, test_real4
  test_real4 = cospi(x)
end function

! CHECK-LABEL: @_QPtest_real4
! CHECK: %[[factor:.*]] = arith.constant 3.14159274 : f32
! CHECK: %[[mul:.*]] = arith.mulf %{{.*}}, %[[factor]] fastmath<contract> : f32
! CHECK: %[[cos:.*]] = math.cos %[[mul]] fastmath<contract> : f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = cospi(x)
end function

! CHECK-LABEL: @_QPtest_real8
! CHECK: %[[dfactor:.*]] = arith.constant 3.1415926535897931 : f64
! CHECK: %[[mul:.*]] = arith.mulf %{{.*}}, %[[dfactor]] fastmath<contract> : f64
! CHECK: %[[cos:.*]] = math.cos %[[mul]] fastmath<contract> : f64

function test_real16(x)
  real(16) :: x, test_real16
  test_real16 = cospi(x)
end function

! CHECK-LABEL: @_QPtest_real16
! CHECK: %[[factor:.*]] = arith.constant 3.141592653589793238462643383279{{.*}} : f128
! CHECK: %[[mul:.*]] = arith.mulf %{{.*}}, %[[factor]] fastmath<contract> : f128
! CHECK: %[[cos:.*]] = fir.call @_FortranACosF128(%[[mul]]) fastmath<contract> : (f128) -> f128
