! This test checks lowering of OpenMP allocate Directive to HLFIR.
! Verifies code generation for default (no align, null allocator) case.
! omp.allocate_free must be emitted at the exit (before return).

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program main
  integer :: x, y
  !$omp allocate(x, y)
end program

! CHECK: %[[X_ALLOC:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ALLOC]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[Y_ALLOC:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_ALLOC]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: omp.allocate_dir(%[[X_DECL]]#0, %[[Y_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: omp.allocate_free(%[[X_DECL]]#0, %[[Y_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: return
