! This test checks lowering of OpenMP declare reduction Directive, with initialization
! via a subroutine. This functionality is currently not implemented.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

subroutine initme(x,n)
  integer x,n
  x=0
end subroutine initme

function func(x, n, init)
  integer func
  integer x(n)
  integer res
  interface
     subroutine initme(x,n)
       integer x,n
     end subroutine initme
  end interface
!CHECK:  omp.declare_reduction @red_add : i32 init {
!CHECK: ^bb0(%[[OMP_ORIG_ARG_I:.*]]: i32):
!CHECK:    %[[OMP_PRIV:.*]] = fir.alloca i32
!CHECK:    %[[OMP_ORIG:.*]] = fir.alloca i32
!CHECK:    fir.store %[[OMP_ORIG_ARG_I]] to %[[OMP_ORIG]] : !fir.ref<i32>
!CHECK:    %[[OMP_ORIG_DECL:.*]]:2 = hlfir.declare %[[OMP_ORIG]] {uniq_name = "omp_orig"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    fir.store %[[OMP_ORIG_ARG_I]] to %[[OMP_PRIV]] : !fir.ref<i32>
!CHECK:    %[[OMP_PRIV_DECL:.*]]:2 = hlfir.declare %[[OMP_PRIV]] {uniq_name = "omp_priv"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    fir.call @_QPinitme(%[[OMP_PRIV_DECL]]#0, %[[OMP_ORIG_DECL]]#0) fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> ()
!CHECK:    %[[OMP_PRIV_VAL:.*]] = fir.load %[[OMP_PRIV_DECL]]#0 : !fir.ref<i32>
!CHECK:    omp.yield(%[[OMP_PRIV_VAL]] : i32)
!CHECK:  } combiner {
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
!CHECK:  }
!CHECK:  func.func @_QPinitme(%[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}, %[[N:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
!CHECK:    %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
!CHECK:    %[[N_DECL:.*]]:2 = hlfir.declare %[[N]] dummy_scope %[[SCOPE]] arg 2 {uniq_name = "_QFinitmeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] dummy_scope %[[OMP_OUT]] arg 1 {uniq_name = "_QFinitmeEx"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[CONST_0:.*]] = arith.constant 0 : i32
!CHECK:    hlfir.assign %[[CONST_0]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:    return
!CHECK:  }
!$omp declare reduction(red_add:integer(4):omp_out=omp_out+omp_in) initializer(initme(omp_priv,omp_orig))
  res=init
!$omp simd reduction(red_add:res)
  do i=1,n
     res=res+x(i)
  enddo
  func=res
end function func
