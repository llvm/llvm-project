! This test checks lowering of OpenMP DISTRIBUTE PARALLEL DO composite
! constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_num_threads(
subroutine distribute_parallel_do_num_threads()
  !$omp teams

  ! CHECK:      omp.parallel num_threads({{.*}}) private({{.*}}) {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do num_threads(10)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams
end subroutine distribute_parallel_do_num_threads

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_dist_schedule(
subroutine distribute_parallel_do_dist_schedule()
  !$omp teams

  ! CHECK:      omp.parallel private({{.*}}) {
  ! CHECK:      omp.distribute dist_schedule_static dist_schedule_chunk_size({{.*}}) {
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do dist_schedule(static, 4)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams
end subroutine distribute_parallel_do_dist_schedule

! Temporaries evaluated outside omp.parallel must be destroyed once, after it.
! CHECK-LABEL: func.func @_QPdistribute_parallel_do_temp_cleanup(
subroutine distribute_parallel_do_temp_cleanup(n)
  integer :: n
  interface
    function get(n) result(r)
      integer, intent(in) :: n
      integer, allocatable :: r(:)
    end function
  end interface
  !$omp teams

  ! CHECK:      omp.teams {
  ! CHECK:      fir.call @_QPget(
  ! CHECK:      %[[DIST_TMP:.*]] = hlfir.as_expr
  ! CHECK:      fir.call @_QPget(
  ! CHECK:      %[[NT_TMP:.*]] = hlfir.as_expr
  ! CHECK:      omp.parallel num_threads({{.*}}) private({{.*}}) {
  ! CHECK-NOT:  hlfir.destroy
  ! CHECK:      omp.distribute dist_schedule_static dist_schedule_chunk_size({{.*}}) {
  ! CHECK-NOT:  hlfir.destroy
  ! CHECK:      omp.terminator
  ! CHECK-NEXT: } {omp.composite}
  ! CHECK:      hlfir.destroy %[[NT_TMP]] :
  ! CHECK:      hlfir.destroy %[[DIST_TMP]] :
  ! CHECK-NEXT: omp.terminator
  !$omp distribute parallel do num_threads(size(get(n))) dist_schedule(static, size(get(n)))
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams
end subroutine distribute_parallel_do_temp_cleanup

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_schedule(
subroutine distribute_parallel_do_schedule()
  !$omp teams

  ! CHECK:      omp.parallel private({{.*}}) {
  ! CHECK:      omp.distribute {
  ! CHECK-NEXT: omp.wsloop schedule(runtime) {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do schedule(runtime)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams
end subroutine distribute_parallel_do_schedule

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_private(
subroutine distribute_parallel_do_private()
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
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do private(x)
  do index_ = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams
end subroutine distribute_parallel_do_private
