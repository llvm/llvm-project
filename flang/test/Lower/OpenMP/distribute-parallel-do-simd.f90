! This test checks lowering of OpenMP DISTRIBUTE PARALLEL DO SIMD composite
! constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_num_threads(
subroutine distribute_parallel_do_simd_num_threads()
  !$omp teams

  ! CHECK:      omp.parallel num_threads({{.*}}) {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd private({{.*}}) {
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

  ! CHECK:      omp.parallel  {
  ! CHECK:      omp.distribute dist_schedule_static dist_schedule_chunk_size({{.*}}) {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd private({{.*}}) {
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

  ! CHECK:      omp.parallel {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop schedule(static = {{.*}}) {
  ! CHECK-NEXT: omp.simd private({{.*}}) {
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

  ! CHECK:      omp.parallel {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd simdlen(4) private({{.*}}) {
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

  ! CHECK:      omp.parallel {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.simd private(@{{.*}} %[[X]]#0 -> %[[X_ARG:[^:]+]], {{.*}}) {
  ! CHECK-NEXT: omp.loop_nest
  ! CHECK:      %[[X_PRIV:.*]]:2 = hlfir.declare %[[X_ARG]]
  !$omp distribute parallel do simd private(x)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams
end subroutine distribute_parallel_do_simd_private

! CHECK-LABEL:   func.func @_QPlastprivate_cond_in_composite_construct
subroutine lastprivate_cond_in_composite_construct(x_min, x_max, y_min, y_max)
implicit none
integer :: x_min,x_max,y_min,y_max
integer :: i,j

! CHECK:           omp.target kernel_type(spmd) {{.*}} {
! CHECK:             %[[X_MAX_MAPPED:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}x_max"}
! CHECK:             omp.teams {
! CHECK:               omp.parallel {
! CHECK:                 omp.distribute {
! CHECK:                   omp.wsloop {
! CHECK:                     omp.simd private({{.*}}) {
! CHECK:                       omp.loop_nest (%[[I_IV:.*]], %[[J_IV:.*]]) : i32 = ({{.*}}) to ({{.*}}) inclusive step ({{.*}}) collapse(2) {
! CHECK:                        %[[I_IV_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_cond_in_composite_constructEi"} {{.*}})
! CHECK:                        %[[J_IV_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_cond_in_composite_constructEj"} {{.*}})
! CHECK:                        hlfir.assign %[[I_IV]] to %[[I_IV_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:                        hlfir.assign %[[J_IV]] to %[[J_IV_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:                        omp.yield

!$omp target teams distribute parallel do simd collapse(2) private(y_max)
  do i=x_min,x_max
    do j=y_min,y_max
    enddo
  enddo
end subroutine

! CHECK-LABEL:   func.func @_QPtarget_teams_distribute_parallel_do_simd_linear
subroutine target_teams_distribute_parallel_do_simd_linear()
  implicit none
  integer :: iv

  ! CHECK: omp.target
  ! CHECK: %[[IV:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtarget_teams_distribute_parallel_do_simd_linearEiv"}
  ! CHECK: omp.simd private(@_QFtarget_teams_distribute_parallel_do_simd_linearEiv_private_i32 %[[IV]]#0 -> %{{.*}} : !fir.ref<i32>)
  !$omp target teams distribute parallel do simd
  do iv = 1, 10
  end do
end subroutine
