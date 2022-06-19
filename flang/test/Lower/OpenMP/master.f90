!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --cfg-conversion | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="OMPDialect"

!===============================================================================
! parallel construct with function call which has master construct internally
!===============================================================================
!FIRDialect-LABEL: func @_QPomp_master
subroutine omp_master()

!OMPDialect: omp.master  {
!$omp master

    !FIRDialect: fir.call @_QPmaster() : () -> ()
    call master()

!OMPDialect: omp.terminator
!$omp end master

end subroutine omp_master

!FIRDialect-LABEL: func @_QPparallel_function_master
subroutine parallel_function_master()

!OMPDialect: omp.parallel {
!$omp parallel

    !FIRDialect: fir.call @_QPfoo() : () -> ()
    call foo()

!OMPDialect: omp.terminator
!$omp end parallel

end subroutine parallel_function_master

!===============================================================================
! master construct nested inside parallel construct
!===============================================================================

!FIRDialect-LABEL: func @_QPomp_parallel_master
subroutine omp_parallel_master()

!OMPDialect: omp.parallel {
!$omp parallel
    !FIRDialect: fir.call @_QPparallel() : () -> ()
    call parallel()

!OMPDialect: omp.master {
!$omp master

    !FIRDialect: fir.call @_QPparallel_master() : () -> ()
    call parallel_master()

!OMPDialect: omp.terminator
!$omp end master

!OMPDialect: omp.terminator
!$omp end parallel

end subroutine omp_parallel_master

!===============================================================================
! master construct nested inside parallel construct with conditional flow
!===============================================================================

!FIRDialect-LABEL: func @_QPomp_master_parallel
subroutine omp_master_parallel()
    integer :: alpha, beta, gama
    alpha = 4
    beta = 5
    gama = 6

!OMPDialect: omp.master {
!$omp master

    !FIRDialect: %{{.*}} = fir.load %{{.*}}
    !FIRDialect: %{{.*}} = fir.load %{{.*}}
    !FIRDialect: %[[RESULT:.*]] = arith.cmpi sge, %{{.*}}, %{{.*}}
    !FIRDialect: fir.if %[[RESULT]] {
    if (alpha .ge. gama) then

!OMPDialect: omp.parallel {
!$omp parallel
        !FIRDialect: fir.call @_QPinside_if_parallel() : () -> ()
        call inside_if_parallel()

!OMPDialect: omp.terminator
!$omp end parallel

        !FIRDialect: %{{.*}} = fir.load %{{.*}}
        !FIRDialect: %{{.*}} = fir.load %{{.*}}
        !FIRDialect: %{{.*}} = arith.addi %{{.*}}, %{{.*}}
        !FIRDialect: fir.store %{{.*}} to %{{.*}}
        beta = alpha + gama
    end if
    !FIRDialect: else

!OMPDialect: omp.terminator
!$omp end master

end subroutine omp_master_parallel
