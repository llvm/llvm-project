! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK"

function test_real4(x)
  real :: x, test_real4
  test_real4 = acosd(x)
end function

! CHECK-LABEL: @_QPtest_real4
! CHECK: %[[dfactor:.*]] = arith.constant 57.295779513082323 : f64
! CHECK: %[[result:.*]] = math.acos %{{.*}} fastmath<contract> : f32
! CHECK: %[[factor:.*]] = fir.convert %[[dfactor]] : (f64) -> f32
! CHECK: %[[arg:.*]] = arith.mulf %[[result]], %[[factor]] fastmath<contract> : f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = acosd(x)
end function

! CHECK-LABEL: @_QPtest_real8
! CHECK: %[[dfactor:.*]] = arith.constant 57.295779513082323 : f64
! CHECK: %[[result:.*]] = math.acos %{{.*}} fastmath<contract> : f64
! CHECK: %[[arg:.*]] = arith.mulf %[[result]], %[[dfactor]] fastmath<contract> : f64
