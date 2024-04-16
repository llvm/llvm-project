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

subroutine sub()
  integer, save:: a
  !$omp threadprivate(a)
  !$omp parallel
    print *, a
  !$omp end parallel
end subroutine

!CHECK-LABEL: @_QPsub2() {
!CHECK:   %[[STACK:.*]] = fir.call @llvm.stacksave.p0() fastmath<contract> : () -> !fir.ref<i8>
!CHECK:   %[[A:.*]] = fir.address_of(@_QFsub2B1Ea) : !fir.ref<i32>
!CHECK:   %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {uniq_name = "_QFsub2B1Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   %[[TP_A:.*]] = omp.threadprivate %[[A_DECL]]#1 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:   %[[TP_A_DECL:.*]]:2 = hlfir.declare %[[TP_A]] {uniq_name = "_QFsub2B1Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   fir.call @llvm.stackrestore.p0(%[[STACK]]) fastmath<contract> : (!fir.ref<i8>) -> ()

subroutine sub2()
  BLOCK
    integer, save :: a
    !$omp threadprivate(a)
  END BLOCK
end subroutine

!CHECK:  fir.global internal @_QFsubEa : i32
!CHECK:  fir.global internal @_QFsub2B1Ea : i32

