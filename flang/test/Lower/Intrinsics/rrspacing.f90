! REQUIRES: flang-supports-f128-math
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPrrspacing_test2(
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.ref<f128>{{.*}}) -> f128
real*16 function rrspacing_test2(x)
  real*16 :: x
  rrspacing_test2 = rrspacing(x)
! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg0]]{{.*}}{uniq_name = "_QFrrspacing_test2Ex"}
! CHECK: %[[a1:.*]] = fir.load %[[x]]#0 : !fir.ref<f128>
! CHECK: %{{.*}} = fir.call @_FortranARRSpacing16(%[[a1]]) {{.*}}: (f128) -> f128
end function
