! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! Per the OpenMP spec, an in_reduction list item on a target construct is
! implicitly data-mapped. The lowering must not rely on the variable being
! referenced inside the target body to discover that map: here `i` only
! appears in the in_reduction clause and is never read or written inside
! the region. Verify that an omp.map.info for `i` is still emitted and
! flows into the omp.target's map_entries.

!CHECK-LABEL: func.func @_QPomp_target_in_reduction_unused()
!CHECK:       %[[IDECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFomp_target_in_reduction_unusedEi"}
!CHECK:       %[[IMAP:.*]] = omp.map.info var_ptr(%[[IDECL]]#1 : !fir.ref<i32>, i32) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<i32> {name = "i"}
!CHECK:       omp.target kernel_type(generic) in_reduction(@{{[^ ]+}} %[[IDECL]]#0 : !fir.ref<i32>)
!CHECK-SAME:    map_entries(%[[IMAP]] -> %{{[^ ]+}} : !fir.ref<i32>)

subroutine omp_target_in_reduction_unused()
  interface
    subroutine sub()
    end subroutine
  end interface
  integer i
  i = 0
  !$omp target in_reduction(+:i)
  call sub()
  !$omp end target
end subroutine omp_target_in_reduction_unused
