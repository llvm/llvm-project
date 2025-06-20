!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! Check that the presence tests are done outside of the atomic update
! construct.

!CHECK: %[[IS_PRESENT:[a-z0-9]+]] = fir.is_present
!CHECK: %[[IF_VAL:[a-z0-9]+]] = fir.if %[[IS_PRESENT]] -> (f32) {
!CHECK:   fir.result {{.*}} : f32
!CHECK: } else {
!CHECK:   fir.result {{.*}} : f32
!CHECK: }
!CHECK: omp.atomic.update {{.*}} : !fir.ref<f32> {
!CHECK: ^bb0(%[[ARG:[a-z0-9]+]]: f32):
!CHECK:   %[[V10:[a-z0-9]+]] = arith.cmpf ogt, %[[ARG]], %[[IF_VAL]]
!CHECK:   %[[V11:[a-z0-9]+]] = arith.select %[[V10]], %[[ARG]], %[[IF_VAL]]
!CHECK:   omp.yield(%[[V11]] : f32)
!CHECK: }

subroutine f00(a, x, y)
  real :: a
  real, optional :: x, y
  !$omp atomic update
  a = max(x, a, y)
end
