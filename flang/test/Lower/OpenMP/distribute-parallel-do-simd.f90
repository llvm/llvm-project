! This test checks lowering of OpenMP DISTRIBUTE PARALLEL DO SIMD composite
! constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_num_threads(
subroutine distribute_parallel_do_simd_num_threads()
  !$omp teams

  ! CHECK:      omp.parallel num_threads({{.*}}) private({{.*}}) {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd num_threads(10)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams
end subroutine distribute_parallel_do_simd_num_threads

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_dist_schedule(
subroutine distribute_parallel_do_simd_dist_schedule()
  !$omp teams

  ! CHECK:      omp.parallel private({{.*}}) {
  ! CHECK:      omp.distribute dist_schedule_static dist_schedule_chunk_size({{.*}}) {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd dist_schedule(static, 4)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams
end subroutine distribute_parallel_do_simd_dist_schedule

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_schedule(
subroutine distribute_parallel_do_simd_schedule()
  !$omp teams

  ! CHECK:      omp.parallel private({{.*}}) {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop schedule(static = {{.*}}) {
  ! CHECK-NEXT: omp.simd {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd schedule(static, 4)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams
end subroutine distribute_parallel_do_simd_schedule

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_simdlen(
subroutine distribute_parallel_do_simd_simdlen()
  !$omp teams

  ! CHECK:      omp.parallel private({{.*}}) {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd simdlen(4) {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd simdlen(4)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams
end subroutine distribute_parallel_do_simd_simdlen

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_private(
subroutine distribute_parallel_do_simd_private()
  ! CHECK: %[[INDEX_ALLOC:.*]] = fir.alloca i32
  ! CHECK: %[[INDEX:.*]]:2 = hlfir.declare %[[INDEX_ALLOC]]
  ! CHECK: %[[X_ALLOC:.*]] = fir.alloca i64
  ! CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOC]]
  integer(8) :: x

  ! CHECK: omp.teams {
  !$omp teams

  ! CHECK:      omp.parallel private(@{{.*}} %[[X]]#0 -> %[[X_ARG:[^,]+]],
  ! CHECK-SAME:                      @{{.*}} %[[INDEX]]#0 -> %[[INDEX_ARG:.*]] : !fir.ref<i64>, !fir.ref<i32>) {
  ! CHECK:      %[[X_PRIV:.*]]:2 = hlfir.declare %[[X_ARG]]
  ! CHECK:      %[[INDEX_PRIV:.*]]:2 = hlfir.declare %[[INDEX_ARG]]
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd private(x)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams
end subroutine distribute_parallel_do_simd_private
