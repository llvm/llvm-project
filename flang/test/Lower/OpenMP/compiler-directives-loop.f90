!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Check that we generate proper body of the do-construct.

!CHECK: omp.loop_nest (%[[ARG1:arg[0-9]+]]) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32_1) {
!CHECK:   %[[V0:[0-9]+]]:2 = hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   hlfir.assign %[[ARG1]] to %[[V0]]#0 : i32, !fir.ref<i32>
!CHECK:   %[[V1:[0-9]+]] = fir.load %[[V0]]#0 : !fir.ref<i32>
!CHECK:   %[[V2:[0-9]+]] = fir.convert %[[V1]] : (i32) -> f32
!CHECK:   %[[V3:[0-9]+]] = fir.load %[[V0]]#0 : !fir.ref<i32>
!CHECK:   %[[V4:[0-9]+]] = fir.convert %[[V3]] : (i32) -> i64
!CHECK:   %[[V5:[0-9]+]] = hlfir.designate %3#0 (%[[V4]])  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
!CHECK:   hlfir.assign %[[V2]] to %[[V5]] : f32, !fir.ref<f32>
!CHECK:   omp.yield
!CHECK: }

program omp_cdir_codegen
  implicit none
  integer, parameter :: n = 10
  real :: a(n)
  integer :: i

!$omp parallel do
!dir$ unroll
  do i = 1, n
    a(i) = real(i)
  end do
!$omp end parallel do

  print *, 'a(1)=', a(1), ' a(n)=', a(n)
end program omp_cdir_codegen
