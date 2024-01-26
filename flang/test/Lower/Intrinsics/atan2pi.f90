! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"


function test_real4(y,x)
  real(4) :: x, y, test_real4
  test_real4 = atan2pi(y,x)
end function

! CHECK-LABEL: @_QPtest_real4
! CHECK: %[[terminationCheck:.*]] = arith.andi %[[YEq0:.*]], %[[XEq0:.*]] : i1
! CHECK: fir.if %[[terminationCheck]]
! CHECK-FAST: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}}: f32
! CHECK: %[[dpi:.*]] = arith.constant 3.1415926535897931 : f64
! CHECK: %[[pi:.*]] = fir.convert %[[dpi]] : (f64) -> f32
! CHECK: %{{.*}} = arith.divf %[[atan2]], %[[pi]] fastmath<contract> : f32

function test_real8(y,x)
  real(8) :: x, y, test_real8
  test_real8 = atan2pi(y,x)
end function

! CHECK-LABEL: @_QPtest_real8
! CHECK: %[[terminationCheck:.*]] = arith.andi %[[YEq0:.*]], %[[XEq0:.*]] : i1
! CHECK: fir.if %[[terminationCheck]]
! CHECK-FAST: %[[atan2:.*]] = math.atan2 %{{.*}}, %{{.*}}: f64
! CHECK: %[[pi:.*]] = arith.constant 3.1415926535897931 : f64
! CHECK: %{{.*}} = arith.divf %[[atan2]], %[[pi]] fastmath<contract> : f64
