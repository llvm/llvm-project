! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPerfc_scaled4(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f32>{{.*}}) -> f32
function erfc_scaled4(x)
  real(kind=4) :: erfc_scaled4
  real(kind=4) :: x
  erfc_scaled4 = erfc_scaled(x);
! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[RET_ALLOC:.*]] = fir.alloca f32
! CHECK: %[[RET_DECL:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[x]] dummy_scope %[[DS]]
! CHECK: %[[a1:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
! CHECK: %[[VAL:.*]] = fir.call @_FortranAErfcScaled4(%[[a1]]) {{.*}}: (f32) -> f32
! CHECK: hlfir.assign %[[VAL]] to %[[RET_DECL]]#0
end function erfc_scaled4


! CHECK-LABEL: func.func @_QPerfc_scaled8(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f64>{{.*}}) -> f64
function erfc_scaled8(x)
  real(kind=8) :: erfc_scaled8
  real(kind=8) :: x
  erfc_scaled8 = erfc_scaled(x);
! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[RET_ALLOC:.*]] = fir.alloca f64
! CHECK: %[[RET_DECL:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[x]] dummy_scope %[[DS]]
! CHECK: %[[a1:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f64>
! CHECK: %[[VAL:.*]] = fir.call @_FortranAErfcScaled8(%[[a1]]) {{.*}}: (f64) -> f64
! CHECK: hlfir.assign %[[VAL]] to %[[RET_DECL]]#0
end function erfc_scaled8


! CHECK-LABEL: func.func @_QPderfc_scaled8(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f64>{{.*}}) -> f64
function derfc_scaled8(x)
  real(kind=8) :: derfc_scaled8
  real(kind=8) :: x
  derfc_scaled8 = derfc_scaled(x);
! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[RET_ALLOC:.*]] = fir.alloca f64
! CHECK: %[[RET_DECL:.*]]:2 = hlfir.declare %[[RET_ALLOC]]
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[x]] dummy_scope %[[DS]]
! CHECK: %[[a1:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f64>
! CHECK: %[[VAL:.*]] = fir.call @_FortranAErfcScaled8(%[[a1]]) {{.*}}: (f64) -> f64
! CHECK: hlfir.assign %[[VAL]] to %[[RET_DECL]]#0
end function derfc_scaled8
