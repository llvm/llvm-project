! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME: @[[MIN_RED_I32_NAME:.*]] : i32 init {
!CHECK: ^bb0(%{{.*}}: i32):
!CHECK:  %[[C0_1:.*]] = arith.constant 2147483647 : i32
!CHECK:  omp.yield(%[[C0_1]] : i32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:  %[[RES:.*]] = arith.minsi %[[ARG0]], %[[ARG1]] : i32
!CHECK:  omp.yield(%[[RES]] : i32)
!CHECK: }

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME: @[[ADD_RED_F32_NAME:.*]] : f32 init {
!CHECK: ^bb0(%{{.*}}: f32):
!CHECK:   %[[C0_1:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:   omp.yield(%[[C0_1]] : f32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
!CHECK:   %[[RES:.*]] = arith.addf %[[ARG0]], %[[ARG1]] {{.*}} : f32
!CHECK:   omp.yield(%[[RES]] : f32)
!CHECK: }

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME: @[[ADD_RED_I32_NAME:.*]] : i32 init {
!CHECK: ^bb0(%{{.*}}: i32):
!CHECK:  %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:  omp.yield(%[[C0_1]] : i32)
!CHECK: } combiner {
!CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:  %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
!CHECK:  omp.yield(%[[RES]] : i32)
!CHECK: }

!CHECK-LABEL: func.func @_QPmultiple_reduction
!CHECK:      %[[X_REF:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_reductionEx"}
!CHECK:      %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]] {uniq_name = "_QFmultiple_reductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[Y_REF:.*]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFmultiple_reductionEy"}
!CHECK:      %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_REF]] {uniq_name = "_QFmultiple_reductionEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:      %[[Z_REF:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFmultiple_reductionEz"}
!CHECK:      %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z_REF]] {uniq_name = "_QFmultiple_reductionEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      omp.wsloop reduction(
!CHECK-SAME: @[[ADD_RED_I32_NAME]] %[[X_DECL]]#0 -> %[[PRV_X:.+]] : !fir.ref<i32>,
!CHECK-SAME: @[[ADD_RED_F32_NAME]] %[[Y_DECL]]#0 -> %[[PRV_Y:.+]] : !fir.ref<f32>,
!CHECK-SAME: @[[MIN_RED_I32_NAME]] %[[Z_DECL]]#0 -> %[[PRV_Z:.+]] : !fir.ref<i32>) {
!CHECK-NEXT:   omp.loop_nest {{.*}} {
!CHECK:          %[[PRV_X_DECL:.+]]:2 = hlfir.declare %[[PRV_X]] {{.*}} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:          %[[PRV_Y_DECL:.+]]:2 = hlfir.declare %[[PRV_Y]] {{.*}} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:          %[[PRV_Z_DECL:.+]]:2 = hlfir.declare %[[PRV_Z]] {{.*}} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:          %[[LPRV_X:.+]] = fir.load %[[PRV_X_DECL]]#0 : !fir.ref<i32>
!CHECK:          %[[RES_X:.+]] = arith.addi %[[LPRV_X]], %{{.+}} : i32
!CHECK:          hlfir.assign %[[RES_X]] to %[[PRV_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:          %[[LPRV_Y:.+]] = fir.load %[[PRV_Y_DECL]]#0 : !fir.ref<f32>
!CHECK:          %[[RES_Y:.+]] = arith.addf %[[LPRV_Y]], %{{.+}} : f32
!CHECK:          hlfir.assign %[[RES_Y]] to %[[PRV_Y_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:          %[[LPRV_Z:.+]] = fir.load %[[PRV_Z_DECL]]#0 : !fir.ref<i32>
!CHECK:          %[[RES_Z:.+]] = arith.select %{{.+}}, %[[LPRV_Z]], %{{.+}} : i32
!CHECK:          hlfir.assign %[[RES_Z]] to %[[PRV_Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:          omp.yield
!CHECK:        }
!CHECK:        omp.terminator
!CHECK:      }
!CHECK:      return
subroutine multiple_reduction(v)
  implicit none
  integer, intent(in) :: v(:)
  integer :: i
  integer :: x
  real :: y
  integer:: z
  x = 0
  y = 0.0
  z = 10

  !$omp do reduction(+:x,y) reduction(min:z)
  do i=1, 100
    x = x + v(i)
    y = y + 1.5 * v(i)
    z = min(z, v(i))
  end do
  !$omp end do
end subroutine
