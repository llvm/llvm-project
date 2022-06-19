! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPaint_test(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine aint_test(a, b)
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK: %[[VAL_3:.*]] = fir.call @llvm.trunc.f32(%[[VAL_2]]) : (f32) -> f32
! CHECK: fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK: return
  real :: a, b
  b = aint(a)
end subroutine
