!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: @_QPomp_taskgroup
subroutine omp_taskgroup
use omp_lib
integer :: allocated_x
!CHECK: %[[ALLOC_X_REF:.*]] = fir.alloca i32 {bindc_name = "allocated_x", uniq_name = "_QFomp_taskgroupEallocated_x"}
!CHECK-NEXT: %[[ALLOC_X_DECL:.*]]:2 = hlfir.declare %[[ALLOC_X_REF]] {uniq_name = "_QFomp_taskgroupEallocated_x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[C1:.*]] = arith.constant 1 : i32

!CHECK: omp.taskgroup  allocate(%[[C1]] : i32 -> %[[ALLOC_X_DECL]]#1 : !fir.ref<i32>)
!$omp taskgroup allocate(omp_high_bw_mem_alloc: allocated_x)
!$omp task
!CHECK: fir.call @_QPwork() {{.*}}: () -> ()
   call work()
!CHECK: omp.terminator
!$omp end task
!CHECK: omp.terminator
!$omp end taskgroup
end subroutine
