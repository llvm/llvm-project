! Tests that a `do concurrent reduce(min:...)` on a scalar maps the reduction
! variable as `tofrom ByRef` (not `ByCopy`) when targeting a device. This is
! needed so the reduced result is written back from the device to the host.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine min_reduce(arr, n, min_val)
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: arr(n)
    real :: min_val
    integer :: i

    do concurrent (i=1:n) reduce(min:min_val)
        min_val = min(min_val, arr(i))
    end do
end subroutine min_reduce

! CHECK-DAG: omp.declare_reduction @[[RED_SYM:.*\.omp]] : f32 init

! CHECK-LABEL: func.func @_QPmin_reduce

! CHECK: %[[MIN_VAL_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} {uniq_name = "_QFmin_reduceEmin_val"}

! Verify the reduction variable is mapped tofrom + ByRef (not implicit + ByCopy).
! CHECK: %[[MIN_VAL_MAP:.*]] = omp.map.info var_ptr(%[[MIN_VAL_DECL]]#1
! CHECK-SAME: map_clauses(implicit, tofrom) capture(ByRef)
! CHECK-SAME: -> !fir.ref<f32> {name = "_QFmin_reduceEmin_val"}

! CHECK: omp.target
! CHECK-SAME: map_entries({{.*}}%[[MIN_VAL_MAP]] -> %[[MIN_VAL_ARG:[[:alnum:]]+]]{{.*}})

! CHECK: %[[MIN_VAL_DEV:.*]]:2 = hlfir.declare %[[MIN_VAL_ARG]] {{.*}} "_QFmin_reduceEmin_val"
! CHECK: omp.teams reduction(@[[RED_SYM]] %[[MIN_VAL_DEV]]#0 -> %[[RED_TEAMS:.*]] : !fir.ref<f32>) {
! CHECK:   omp.parallel {
! CHECK:     omp.distribute {
! CHECK:       omp.wsloop reduction(@[[RED_SYM]] %[[RED_TEAMS]] -> %[[RED_WS:.*]] : !fir.ref<f32>) {
! CHECK:         omp.loop_nest
! CHECK:       } {omp.composite}
! CHECK:     } {omp.composite}
! CHECK:   } {omp.composite}
! CHECK: }
