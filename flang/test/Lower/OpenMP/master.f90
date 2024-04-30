!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!===============================================================================
! parallel construct with function call which has master construct internally
!===============================================================================
!CHECK-LABEL: func @_QPomp_master
subroutine omp_master()

!CHECK: omp.master  {
!$omp master

    !CHECK: fir.call @_QPmaster() {{.*}}: () -> ()
    call master()

!CHECK: omp.terminator
!$omp end master

end subroutine omp_master

!CHECK-LABEL: func @_QPparallel_function_master
subroutine parallel_function_master()

!CHECK: omp.parallel {
!$omp parallel

    !CHECK: fir.call @_QPfoo() {{.*}}: () -> ()
    call foo()

!CHECK: omp.terminator
!$omp end parallel

end subroutine parallel_function_master

!===============================================================================
! master construct nested inside parallel construct
!===============================================================================

!CHECK-LABEL: func @_QPomp_parallel_master
subroutine omp_parallel_master()

!CHECK: omp.parallel {
!$omp parallel
    !CHECK: fir.call @_QPparallel() {{.*}}: () -> ()
    call parallel()

!CHECK: omp.master {
!$omp master

    !CHECK: fir.call @_QPparallel_master() {{.*}}: () -> ()
    call parallel_master()

!CHECK: omp.terminator
!$omp end master

!CHECK: omp.terminator
!$omp end parallel

end subroutine omp_parallel_master

!===============================================================================
! master construct nested inside parallel construct with conditional flow
!===============================================================================

!CHECK-LABEL: func @_QPomp_master_parallel
subroutine omp_master_parallel()
    integer :: alpha, beta, gama
    alpha = 4
    beta = 5
    gama = 6

!CHECK: omp.master {
!$omp master

    !CHECK: %{{.*}} = fir.load %{{.*}}
    !CHECK: %{{.*}} = fir.load %{{.*}}
    !CHECK: %[[RESULT:.*]] = arith.cmpi sge, %{{.*}}, %{{.*}}
    !CHECK: fir.if %[[RESULT]] {
    if (alpha .ge. gama) then

!CHECK: omp.parallel {
!$omp parallel
        !CHECK: fir.call @_QPinside_if_parallel() {{.*}}: () -> ()
        call inside_if_parallel()

!CHECK: omp.terminator
!$omp end parallel

        !CHECK: %{{.*}} = fir.load %{{.*}}
        !CHECK: %{{.*}} = fir.load %{{.*}}
        !CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}}
        !CHECK: hlfir.assign %{{.*}} to %{{.*}}#0 : i32, !fir.ref<i32>
        beta = alpha + gama
    end if
    !CHECK: else

!CHECK: omp.terminator
!$omp end master

end subroutine omp_master_parallel
