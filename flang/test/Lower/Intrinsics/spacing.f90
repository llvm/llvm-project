! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPspacing_test(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f32>{{.*}}) -> f32
real*4 function spacing_test(x)
  real*4 :: x
  spacing_test = spacing(x)
! CHECK: %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f32>
! CHECK: %{{.*}} = fir.call @_FortranASpacing4(%[[a1]]) : (f32) -> f32
end function

! CHECK-LABEL: func @_QPspacing_test2(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f80>{{.*}}) -> f80
real*10 function spacing_test2(x)
  real*10 :: x
  spacing_test2 = spacing(x)
! CHECK: %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f80>
! CHECK: %{{.*}} = fir.call @_FortranASpacing10(%[[a1]]) : (f80) -> f80
end function
