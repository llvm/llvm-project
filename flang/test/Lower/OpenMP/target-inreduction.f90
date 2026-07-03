! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! Verify that in_reduction on a target construct is lowered to an
! omp.target with both an in_reduction clause and an implicit map_entries
! entry for the same variable. The in_reduction clause does not define an
! entry block argument: inside the target body the variable is accessed
! through its map_entries block argument. The implicit map also captures the
! original pointer into the target region so the MLIR -> LLVM IR translation
! can pass it to __kmpc_task_reduction_get_th_data.

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME:  @[[RED_I32_NAME:.*]] : i32 init {

!CHECK-LABEL: func.func @_QPomp_target_in_reduction()
!CHECK:       %[[IDECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFomp_target_in_reductionEi"}
!CHECK:       %[[IMAP:.*]] = omp.map.info var_ptr(%[[IDECL]]#1 : !fir.ref<i32>, i32) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<i32> {name = "i"}
!CHECK:       omp.target kernel_type(generic) in_reduction(@[[RED_I32_NAME]] %[[IDECL]]#0 : !fir.ref<i32>)
!CHECK-SAME:    map_entries(%[[IMAP]] -> %[[MAPARG:[^ ]+]] : !fir.ref<i32>)
!CHECK:         hlfir.declare %[[MAPARG]]
!CHECK:         omp.terminator
!CHECK:       }

subroutine omp_target_in_reduction()
  integer i
  i = 0
  !$omp target in_reduction(+:i)
  i = i + 1
  !$omp end target
end subroutine omp_target_in_reduction
