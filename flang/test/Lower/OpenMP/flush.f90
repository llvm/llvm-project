! This test checks lowering of OpenMP Flush Directive.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL:  func.func @_QPflush_standalone
!CHECK-SAME: %[[ARG_A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}, %[[ARG_B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}, %[[ARG_C:.*]]: !fir.ref<i32> {fir.bindc_name = "c"})
subroutine flush_standalone(a, b, c)
    integer, intent(inout) :: a, b, c

!CHECK:    %[[A:.*]]:2 = hlfir.declare %[[ARG_A]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFflush_standaloneEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[B:.*]]:2 = hlfir.declare %[[ARG_B]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFflush_standaloneEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C:.*]]:2 = hlfir.declare %[[ARG_C]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFflush_standaloneEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.flush(%[[A]]#1, %[[B]]#1, %[[C]]#1 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.flush
!$omp flush(a,b,c)
!$omp flush

end subroutine flush_standalone

!CHECK-LABEL: func.func @_QPflush_parallel
!CHECK-SAME: %[[ARG_A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}, %[[ARG_B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}, %[[ARG_C:.*]]: !fir.ref<i32> {fir.bindc_name = "c"})
subroutine flush_parallel(a, b, c)
    integer, intent(inout) :: a, b, c
!CHECK:    %[[A:.*]]:2 = hlfir.declare %[[ARG_A]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFflush_parallelEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[B:.*]]:2 = hlfir.declare %[[ARG_B]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFflush_parallelEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C:.*]]:2 = hlfir.declare %[[ARG_C]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFflush_parallelEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!$omp parallel
!CHECK:    omp.parallel
!CHECK:      omp.flush(%[[A]]#1, %[[B]]#1, %[[C]]#1 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
!CHECK:      omp.flush
!$omp flush(a,b,c)
!$omp flush

!CHECK:      %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
!CHECK:      %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
!CHECK:      %[[C_VAL:.*]] = arith.addi %3, %4 : i32
!CHECK:      hlfir.assign %[[C_VAL]] to %[[C]]#0 : i32, !fir.ref<i32>
    c = a + b

!CHECK: omp.terminator
!$omp END parallel

end subroutine flush_parallel
