! Simple test for lowering of OpenMP Threadprivate Directive with HLFIR.

!RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: @_QPsub
!CHECK:    %[[ADDR:.*]] = fir.address_of(@_QFsubEa) : !fir.ref<i32>
!CHECK:    %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]] {uniq_name = "_QFsubEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[TP:.*]] = omp.threadprivate %[[DECL]]#1 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:    %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP:.*]] {uniq_name = "_QFsubEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.parallel   {
!CHECK:      %[[TP_PARALLEL:.*]] = omp.threadprivate %[[DECL]]#1 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:      %[[TP_PARALLEL_DECL:.*]]:2 = hlfir.declare %[[TP_PARALLEL]] {uniq_name = "_QFsubEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[TP_VAL:.*]] = fir.load %[[TP_PARALLEL_DECL]]#0 : !fir.ref<i32>
!CHECK:      %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[TP_VAL]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
!CHECK:      omp.terminator

!CHECK:  fir.global internal @_QFsubEa : i32

subroutine sub()
  integer, save:: a
  !$omp threadprivate(a)
  !$omp parallel
    print *, a
  !$omp end parallel
end subroutine

