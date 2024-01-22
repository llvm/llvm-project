!Test data-sharing attribute clauses for the `target` directive.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPomp_target_private()
subroutine omp_target_private
    implicit none
    integer :: x(1)

!$omp target private(x)
    x(1) = 42
!$omp end target
!CHECK: omp.target {
!CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
!CHECK-DAG:    %[[PRIV_ALLOC:.*]] = fir.alloca !fir.array<1xi32> {bindc_name = "x",
!CHECK-SAME:     pinned, uniq_name = "_QFomp_target_privateEx"}
!CHECK-NEXT:   %[[SHAPE:.*]] = fir.shape %[[C1]] : (index) -> !fir.shape<1>
!CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]](%[[SHAPE]])
!CHECK-SAME:     {uniq_name = "_QFomp_target_privateEx"} :
!CHECK-SAME:     (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) ->
!CHECK-SAME:     (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
!CHECK-DAG:    %[[C42:.*]] = arith.constant 42 : i32
!CHECK-DAG:    %[[C1_2:.*]] = arith.constant 1 : index
!CHECK-NEXT:   %[[PRIV_BINDING:.*]] = hlfir.designate %[[PRIV_DECL]]#0 (%[[C1_2]])
!CHECK-SAME:     : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
!CHECK-NEXT:   hlfir.assign %[[C42]] to %[[PRIV_BINDING]] : i32, !fir.ref<i32>
!CHECK-NEXT:   omp.terminator
!CHECK-NEXT: }

end subroutine omp_target_private
