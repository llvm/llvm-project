! This test checks lowering of OpenMP declare reduction Directive.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

subroutine declare_red()
  integer :: my_var
!CHECK: omp.declare_reduction @my_red : i32 init {
!CHECK: ^bb0(%[[OMP_ORIG_ARG_I:.*]]: i32):
!CHECK:    %[[OMP_PRIV:.*]] = fir.alloca i32
!CHECK:    %[[OMP_ORIG:.*]] = fir.alloca i32
!CHECK:    fir.store %[[OMP_ORIG_ARG_I]] to %[[OMP_ORIG]] : !fir.ref<i32>
!CHECK:    %[[OMP_ORIG_DECL:.*]]:2 = hlfir.declare %[[OMP_ORIG]] {uniq_name = "omp_orig"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    fir.store %[[OMP_ORIG_ARG_I]] to %[[OMP_PRIV]] : !fir.ref<i32>
!CHECK:    %[[OMP_PRIV_DECL:.*]]:2 = hlfir.declare %[[OMP_PRIV]] {uniq_name = "omp_priv"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[CONST_0:.*]] = arith.constant 0 : i32
!CHECK:    omp.yield(%[[CONST_0]] : i32)
!CHECK: } combiner {
!CHECK:  ^bb0(%[[LHS_ARG:.*]]: i32, %[[RHS_ARG:.*]]: i32):
!CHECK:    %[[OMP_OUT:.*]] = fir.alloca i32
!CHECK:    %[[OMP_IN:.*]] = fir.alloca i32
!CHECK:    fir.store %[[RHS_ARG]] to %[[OMP_IN]] : !fir.ref<i32>
!CHECK:    %[[OMP_IN_DECL:.*]]:2 = hlfir.declare %[[OMP_IN]] {uniq_name = "omp_in"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    fir.store %[[LHS_ARG]] to %[[OMP_OUT]] : !fir.ref<i32>
!CHECK:    %[[OMP_OUT_DECL:.*]]:2 = hlfir.declare %[[OMP_OUT]] {uniq_name = "omp_out"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[OMP_OUT_VAL:.*]] = fir.load %[[OMP_OUT_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[OMP_IN_VAL:.*]] = fir.load %[[OMP_IN_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[SUM:.*]] = arith.addi %[[OMP_OUT_VAL]], %[[OMP_IN_VAL]] : i32
!CHECK:    omp.yield(%[[SUM]] : i32)
!CHECK: }

  !$omp declare reduction (my_red : integer : omp_out = omp_out + omp_in) initializer (omp_priv = 0)
  my_var = 0
end subroutine declare_red
