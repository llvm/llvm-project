! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"


function test_real4(y,x)
  real(4) :: x, y, test_real4
  test_real4 = atan2d(y,x)
end function

! CHECK-LABEL: @_QPtest_real4
! CHECK-FAST: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}}: f32
! CHECK: %[[dfactor:.*]] = arith.constant 57.295779513082323 : f64
! CHECK: %[[factor:.*]] = fir.convert %[[dfactor]] : (f64) -> f32
! CHECK: %{{.*}} = arith.mulf %[[atan2]], %[[factor]] fastmath<contract> : f32

function test_real8(y,x)
  real(8) :: x, y, test_real8
  test_real8 = atan2d(y,x)
end function

! CHECK-LABEL: @_QPtest_real8
! CHECK-FAST: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}}: f64
! CHECK: %[[factor:.*]] = arith.constant 57.295779513082323 : f64
! CHECK: %{{.*}} = arith.mulf %[[atan2]], %[[factor]] fastmath<contract> : f64
