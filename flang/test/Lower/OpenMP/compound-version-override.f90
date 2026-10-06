! RUN: bbc -fopenmp -fopenmp-version=45 -emit-hlfir %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=45 %s -o - 2>&1 | FileCheck %s

!CHECK: warning: IF clause is not allowed on DO SIMD directive in OpenMP v4.5, try -fopenmp-version=50

!CHECK: %[[A:[0-9]+]]:2 = hlfir.declare %{{.*}} uniq_name("_QFcompoundEa")
!CHECK: omp.wsloop {
!CHECK:   omp.simd if(%true) linear(%[[V0:[0-9]+]]#0 : !fir.ref<i32> = %c1_i32_0 : i32) linear_var_types([i32]) {
!CHECK:     omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%{{[0-9]+}}) inclusive step (%c1_i32_0) {
!CHECK:       hlfir.assign %arg2 to %[[V0]]#0 : i32, !fir.ref<i32>
!CHECK:       %[[V1:[0-9]+]] = fir.load %[[V0]]#0 : !fir.ref<i32>
!CHECK:       %[[V2:[0-9]+]] = fir.convert %[[V1]] : (i32) -> i64
!CHECK:       %[[V3:[0-9]+]] = hlfir.designate %[[A]]#0 (%[[V2]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
!CHECK:       %[[V4:[0-9]+]] = fir.load %[[V3]] : !fir.ref<f32>
!CHECK:       %cst = arith.constant 1.000000e+00 : f32
!CHECK:       %[[V5:[0-9]+]] = arith.addf %[[V4]], %cst fastmath<contract> : f32
!CHECK:       %[[V6:[0-9]+]] = fir.load %[[V0]]#0 : !fir.ref<i32>
!CHECK:       %[[V7:[0-9]+]] = fir.convert %[[V6]] : (i32) -> i64
!CHECK:       %[[V8:[0-9]+]] = hlfir.designate %[[A]]#0 (%[[V7]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
!CHECK:       hlfir.assign %[[V5]] to %[[V8]] : f32, !fir.ref<f32>
!CHECK:       omp.yield
!CHECK:     }
!CHECK:   } {omp.composite}
!CHECK: } {omp.composite}

subroutine compound(a, n)
  integer :: n, i
  real :: a(n)
  !$omp do simd if(.true.)
  do i = 1, n
    a(i) = a(i) + 1.0
  end do
  !$omp end do simd
end subroutine
