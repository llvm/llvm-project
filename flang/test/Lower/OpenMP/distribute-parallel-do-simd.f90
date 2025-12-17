! This test checks lowering of OpenMP DISTRIBUTE PARALLEL DO SIMD composite
! constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

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
  ! CHECK-NEXT: omp.simd private(@{{.*}} %[[X]]#0 -> %[[X_ARG:[^,]+]],
  ! CHECK-SAME:                  @{{.*}} %[[INDEX]]#0 -> %[[INDEX_ARG:.*]] : !fir.ref<i64>, !fir.ref<i32>) {
  ! CHECK-NEXT: omp.loop_nest
  ! CHECK:      %[[X_PRIV:.*]]:2 = hlfir.declare %[[X_ARG]]
  ! CHECK:      %[[INDEX_PRIV:.*]]:2 = hlfir.declare %[[INDEX_ARG]]
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

! CHECK:           omp.target {{.*}} {
! CHECK:             %[[X_MAX_MAPPED:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}x_max"}
! CHECK:             omp.teams {
! CHECK:               omp.parallel {
! CHECK:                 omp.distribute {
! CHECK:                   omp.wsloop {
! CHECK:                     omp.simd private({{.*}}) {
! CHECK:                       omp.loop_nest (%[[I_IV:.*]], %[[J_IV:.*]]) : i32 = ({{.*}}) to ({{.*}}) inclusive step ({{.*}}) collapse(2) {
! CHECK:                         %[[Y_MAX_PRIV:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}y_max"}

! CHECK:                         %[[I_UB:.*]] = fir.load %[[X_MAX_MAPPED]]#0 : !fir.ref<i32>
! CHECK:                         %[[I_STEP:.*]] = arith.constant 1 : i32
! CHECK:                         %[[J_UB:.*]] = fir.load %[[Y_MAX_PRIV]]#0 : !fir.ref<i32>
! CHECK:                         %[[J_STEP:.*]] = arith.constant 1 : i32

! CHECK:                         %[[VAL_55:.*]] = arith.addi %[[I_IV]], %[[I_STEP]] : i32
! CHECK:                         %[[VAL_56:.*]] = arith.constant 0 : i32
! CHECK:                         %[[VAL_57:.*]] = arith.cmpi slt, %[[I_STEP]], %[[VAL_56]] : i32
! CHECK:                         %[[VAL_58:.*]] = arith.cmpi slt, %[[VAL_55]], %[[I_UB]] : i32
! CHECK:                         %[[VAL_59:.*]] = arith.cmpi sgt, %[[VAL_55]], %[[I_UB]] : i32
! CHECK:                         %[[VAL_60:.*]] = arith.select %[[VAL_57]], %[[VAL_58]], %[[VAL_59]] : i1

! CHECK:                         %[[VAL_61:.*]] = arith.addi %[[J_IV]], %[[J_STEP]] : i32
! CHECK:                         %[[VAL_62:.*]] = arith.constant 0 : i32
! CHECK:                         %[[VAL_63:.*]] = arith.cmpi slt, %[[J_STEP]], %[[VAL_62]] : i32
! CHECK:                         %[[VAL_64:.*]] = arith.cmpi slt, %[[VAL_61]], %[[J_UB]] : i32
! CHECK:                         %[[VAL_65:.*]] = arith.cmpi sgt, %[[VAL_61]], %[[J_UB]] : i32
! CHECK:                         %[[VAL_66:.*]] = arith.select %[[VAL_63]], %[[VAL_64]], %[[VAL_65]] : i1

! CHECK:                         %[[LASTPRIV_CMP:.*]] = arith.andi %[[VAL_60]], %[[VAL_66]] : i1
! CHECK:                         fir.if %[[LASTPRIV_CMP]] {

!$omp target teams distribute parallel do simd collapse(2) private(y_max)
  do i=x_min,x_max
    do j=y_min,y_max
    enddo
  enddo
end subroutine
