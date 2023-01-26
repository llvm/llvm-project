! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPaint_test(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine aint_test(a, b)
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK: %[[VAL_3:.*]] = fir.call @llvm.trunc.f32(%[[VAL_2]]) {{.*}}: (f32) -> f32
! CHECK: fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK: return
  real :: a, b
  b = aint(a)
end subroutine

! CHECK-LABEL: func.func @_QPaint_test_real8(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<f64> {fir.bindc_name = "a"},
! CHECK-SAME:                                %[[VAL_1:.*]]: !fir.ref<f64> {fir.bindc_name = "b"}) {
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f64>
! CHECK:         %[[VAL_3:.*]] = fir.call @llvm.trunc.f64(%[[VAL_2]]) {{.*}}: (f64) -> f64
! CHECK:         fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f64>
! CHECK:         return
! CHECK:       }

subroutine aint_test_real8(a, b)
  real(8) :: a, b
  b = aint(a)
end subroutine

! CHECK-LABEL: func.func @_QPaint_test_real10(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<f80> {fir.bindc_name = "a"},
! CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<f80> {fir.bindc_name = "b"}) {
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f80>
! CHECK:         %[[VAL_3:.*]] = fir.call @llvm.trunc.f80(%[[VAL_2]]) {{.*}}: (f80) -> f80
! CHECK:         fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f80>
! CHECK:         return
! CHECK:       }

subroutine aint_test_real10(a, b)
  real(10) :: a, b
  b = aint(a)
end subroutine

! TODO: wait until fp128 is supported well in llvm.trunc
!subroutine aint_test_real16(a, b)
!  real(16) :: a, b
!  b = aint(a)
!end subroutine
