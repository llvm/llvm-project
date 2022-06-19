! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPrrspacing_test2(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f128>{{.*}}) -> f128
real*16 function rrspacing_test2(x)
  real*16 :: x
  rrspacing_test2 = rrspacing(x)
! CHECK: %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f128>
! CHECK: %{{.*}} = fir.call @_FortranARRSpacing16(%[[a1]]) : (f128) -> f128
end function
