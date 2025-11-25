! This test checks lowering of OpenMP declare reduction Directive, with combiner
! via a subroutine call.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

subroutine combine_me(out, in)
  integer out, in
  out = out + in
end subroutine combine_me

function func(x, n)
  integer func
  integer x(n)
  integer res
  interface
     subroutine combine_me(out, in)
       integer out, in
     end subroutine combine_me
  end interface
!CHECK:  omp.declare_reduction @red_add : i32 init {
!CHECK: ^bb0(%[[OMP_ORIG_ARG_I:.*]]: i32):
!CHECK:    %[[OMP_PRIV:.*]] = fir.alloca i32
!CHECK:    %[[OMP_ORIG:.*]] = fir.alloca i32
!CHECK:    fir.store %[[OMP_ORIG_ARG_I]] to %[[OMP_ORIG]] : !fir.ref<i32>
!CHECK:    %[[OMP_ORIG_DECL:.*]]:2 = hlfir.declare %[[OMP_ORIG]] {uniq_name = "omp_orig"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    fir.store %[[OMP_ORIG_ARG_I]] to %[[OMP_PRIV]] : !fir.ref<i32>
!CHECK:    %[[OMP_PRIV_DECL:.*]]:2 = hlfir.declare %[[OMP_PRIV]] {uniq_name = "omp_priv"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[CONST_0:.*]] = arith.constant 0 : i32
!CHECK:    omp.yield(%[[CONST_0]] : i32)
!CHECK:  } combiner {
!CHECK:  ^bb0(%[[LHS_ARG:.*]]: i32, %[[RHS_ARG:.*]]: i32):
!CHECK:    %[[OMP_OUT:.*]] = fir.alloca i32
!CHECK:    %[[OMP_IN:.*]] = fir.alloca i32
!CHECK:    fir.store %[[RHS_ARG]] to %[[OMP_IN]] : !fir.ref<i32>
!CHECK:    %[[OMP_IN_DECL:.*]]:2 = hlfir.declare %[[OMP_IN]] {uniq_name = "omp_in"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    fir.store %[[LHS_ARG]] to %[[OMP_OUT]] : !fir.ref<i32>
!CHECK:    %[[OMP_OUT_DECL:.*]]:2 = hlfir.declare %[[OMP_OUT]] {uniq_name = "omp_out"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    fir.call @_QPcombine_me(%[[OMP_OUT_DECL]]#0, %[[OMP_IN_DECL]]#0) fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> ()
!CHECK:    %[[OMP_OUT_VAL:.*]] = fir.load %[[OMP_OUT_DECL]]#0 : !fir.ref<i32>
!CHECK:    omp.yield(%[[OMP_OUT_VAL]] : i32)
!CHECK:  }
!CHECK:  func.func @_QPcombine_me(%[[OUT:.*]]: !fir.ref<i32> {fir.bindc_name = "out"}, %[[IN:.*]]: !fir.ref<i32> {fir.bindc_name = "in"}) {
!CHECK:    %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
!CHECK:    %[[IN_DECL:.*]]:2 = hlfir.declare %[[IN]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFcombine_meEin"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[OUT_DECL:.*]]:2 = hlfir.declare %[[OUT]] dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFcombine_meEout"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[OUT_VAL:.*]] = fir.load %[[OUT_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[IN_VAL:.*]] = fir.load %[[IN_DECL]]#0 : !fir.ref<i32>
!CHECK:    %[[SUM:.*]] = arith.addi %[[OUT_VAL]], %[[IN_VAL]] : i32
!CHECK:    hlfir.assign %[[SUM]] to %[[OUT_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:    return
!CHECK:  }
!$omp declare reduction(red_add:integer(4):combine_me(omp_out,omp_in)) initializer(omp_priv=0)
  res=0
!$omp simd reduction(red_add:res)
  do i=1,n
     res=res+x(i)
  enddo
  func=res
end function func

