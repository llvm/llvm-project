! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK"

function test_real4(x)
  real :: x, test_real4
  test_real4 = tanpi(x)
end function

! CHECK-LABEL: @_QPtest_real4
! CHECK: %[[dfactor:.*]] = arith.constant 3.1415926535897931 : f64
! CHECK: %[[factor:.*]] = fir.convert %[[dfactor]] : (f64) -> f32
! CHECK: %[[mul:.*]] = arith.mulf %{{.*}}, %[[factor]] fastmath<contract> : f32
! CHECK: %[[tan:.*]] = math.tan %[[mul]] fastmath<contract> : f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = tanpi(x)
end function

! CHECK-LABEL: @_QPtest_real8
! CHECK: %[[dfactor:.*]] = arith.constant 3.1415926535897931 : f64
! CHECK: %[[mul:.*]] = arith.mulf %{{.*}}, %[[dfactor]] fastmath<contract> : f64
! CHECK: %[[tan:.*]] = math.tan %[[mul]] fastmath<contract> : f64
