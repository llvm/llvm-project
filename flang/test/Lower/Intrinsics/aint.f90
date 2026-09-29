! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}

! CHECK-LABEL: func @_QPaint_test(
! CHECK-SAME: %[[VAL_0_b:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1_b:.*]]: !fir.ref<f32>{{.*}}) {
subroutine aint_test(a, b)
! CHECK: %[[VAL_0:.*]]:2 = hlfir.declare %[[VAL_0_b]]
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_1_b]]
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]]#0 : !fir.ref<f32>
! CHECK: %[[VAL_3:.*]] = fir.call @llvm.trunc.f32(%[[VAL_2]]) {{.*}}: (f32) -> f32
! CHECK: hlfir.assign %[[VAL_3]] to %[[VAL_1]]#0 : f32, !fir.ref<f32>
! CHECK: return
  real :: a, b
  b = aint(a)
end subroutine

! CHECK-LABEL: func.func @_QPaint_test_real8(
! CHECK:   fir.call @llvm.trunc.f64({{.*}}) {{.*}}: (f64) -> f64

subroutine aint_test_real8(a, b)
  real(8) :: a, b
  b = aint(a)
end subroutine

! CHECK-KIND10-LABEL: func.func @_QPaint_test_real10(
! CHECK-KIND10:   fir.call @llvm.trunc.f80({{.*}}) {{.*}}: (f80) -> f80
subroutine aint_test_real10(a, b)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind10) :: a, b
  b = aint(a)
end subroutine

! TODO: wait until fp128 is supported well in llvm.trunc
!subroutine aint_test_real16(a, b)
!  real(16) :: a, b
!  b = aint(a)
!end subroutine
