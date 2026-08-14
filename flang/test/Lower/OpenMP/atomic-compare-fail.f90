! This test checks lowering of the OpenMP `fail` clause on a (non-capturing)
! atomic compare construct. The fail memory order is attached as the
! `fail_memory_order` attribute on omp.atomic.compare.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPfail_acquire
subroutine fail_acquire(x, e, d)
  integer :: x, e, d
  !$omp atomic compare fail(acquire)
  if (x == e) x = d
end
! CHECK:   %[[E_DECL:.*]]:2 = hlfir.declare %arg1 {{.*}}Ee"
! CHECK:   %[[E_VAL:.*]] = fir.load %[[E_DECL]]#0 : !fir.ref<i32>
! CHECK: omp.atomic.compare memory_order(relaxed) %{{.*}} : !fir.ref<i32> {
! CHECK: ^bb0(%[[XVAL:.*]]: i32):
! CHECK:   arith.cmpi eq, %[[XVAL]], %[[E_VAL]] : i32
! CHECK:   omp.yield
! CHECK: } {fail_memory_order = #omp<memoryorderkind acquire>}

! CHECK-LABEL: func.func @_QPfail_relaxed
subroutine fail_relaxed(x, e, d)
  integer :: x, e, d
  !$omp atomic compare fail(relaxed)
  if (x == e) x = d
end
! CHECK: } {fail_memory_order = #omp<memoryorderkind relaxed>}

! CHECK-LABEL: func.func @_QPfail_seqcst
subroutine fail_seqcst(x, e, d)
  integer :: x, e, d
  !$omp atomic compare fail(seq_cst)
  if (x == e) x = d
end
! CHECK: } {fail_memory_order = #omp<memoryorderkind seq_cst>}

! CHECK-LABEL: func.func @_QPseqcst_fail_relaxed
subroutine seqcst_fail_relaxed(x, e, d)
  integer :: x, e, d
  !$omp atomic seq_cst compare fail(relaxed)
  if (x == e) x = d
end
! CHECK: omp.atomic.compare memory_order(seq_cst) %{{.*}} : !fir.ref<i32> {
! CHECK: } {fail_memory_order = #omp<memoryorderkind relaxed>}
