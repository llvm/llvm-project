! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPdim_testr(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<f32>{{.*}}) {
subroutine dim_testr(x, y, z)
! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DS]] {{.*}}
! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[DS]] {{.*}}
! CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[DS]] {{.*}}
! CHECK: %[[VAL_3:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
! CHECK: %[[VAL_4:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f32>
! CHECK: %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
! CHECK: %[[VAL_6:.*]] = arith.subf %[[VAL_3]], %[[VAL_4]] {{.*}}: f32
! CHECK: %[[VAL_7:.*]] = arith.cmpf ogt, %[[VAL_6]], %[[VAL_5]] {{.*}} : f32
! CHECK: %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_6]], %[[VAL_5]] : f32
! CHECK: hlfir.assign %[[VAL_8]] to %[[Z_DECL]]#0 : f32, !fir.ref<f32>
! CHECK: return
  real :: x, y, z
  z = dim(x, y)
end subroutine

! CHECK-LABEL: func @_QPdim_testi(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dim_testi(i, j, k)
! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DS]] {{.*}}
! CHECK: %[[J_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[DS]] {{.*}}
! CHECK: %[[K_DECL:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[DS]] {{.*}}
! CHECK: %[[VAL_3:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[VAL_4:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK: %[[VAL_6:.*]] = arith.subi %[[VAL_3]], %[[VAL_4]] : i32
! CHECK: %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_5]] : i32
! CHECK: %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_6]], %[[VAL_5]] : i32
! CHECK: hlfir.assign %[[VAL_8]] to %[[K_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: return
  integer :: i, j, k
  k = dim(i, j)
end subroutine
