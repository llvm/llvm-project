! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPdim_testr(
! CHECK-SAME:                     %[[VAL_0:[a-z]+[0-9]]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<f32>{{.*}}) {
subroutine dim_testr(x, y, z)
! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<f32>
! CHECK: %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
! CHECK: %[[VAL_6:.*]] = arith.subf %[[VAL_3]], %[[VAL_4]] {{.*}}: f32
! CHECK: %[[VAL_7:.*]] = arith.cmpf ogt, %[[VAL_6]], %[[VAL_5]] : f32
! CHECK: %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_6]], %[[VAL_5]] : f32
! CHECK: fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK: return
  real :: x, y, z
  z = dim(x, y)
end subroutine

! CHECK-LABEL: func @_QPdim_testi(
! CHECK-SAME: %[[VAL_0:[a-z]+[0-9]]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}) {
subroutine dim_testi(i, j, k)
! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK: %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK: %[[VAL_6:.*]] = arith.subi %[[VAL_3]], %[[VAL_4]] : i32
! CHECK: %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_5]] : i32
! CHECK: %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_6]], %[[VAL_5]] : i32
! CHECK: fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK: return
  integer :: i, j, k
  k = dim(i, j)
end subroutine

