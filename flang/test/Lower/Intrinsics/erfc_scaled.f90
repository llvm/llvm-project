! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPerfc_scaled4(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f32>{{.*}}) -> f32
function erfc_scaled4(x)
  real(kind=4) :: erfc_scaled4
  real(kind=4) :: x
  erfc_scaled4 = erfc_scaled(x);
! CHECK: %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f32>
! CHECK: %{{.*}} = fir.call @_FortranAErfcScaled4(%[[a1]]) {{.*}}: (f32) -> f32
end function erfc_scaled4


! CHECK-LABEL: func @_QPerfc_scaled8(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f64>{{.*}}) -> f64
function erfc_scaled8(x)
  real(kind=8) :: erfc_scaled8
  real(kind=8) :: x
  erfc_scaled8 = erfc_scaled(x);
! CHECK: %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f64>
! CHECK: %{{.*}} = fir.call @_FortranAErfcScaled8(%[[a1]]) {{.*}}: (f64) -> f64
end function erfc_scaled8
