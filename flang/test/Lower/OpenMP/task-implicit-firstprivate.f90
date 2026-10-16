! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[B_FIRSTPRIVATE:.*firstprivate.*]] : !fir.box<!fir.array<2xi32>>

! CHECK-LABEL: func @_QPtest_nested_task_firstprivate
! CHECK:         omp.parallel
! CHECK:           omp.task private(@[[B_FIRSTPRIVATE]] %{{.*}} -> %[[OUTER_TASK_B:.*]] :
! CHECK:             %[[OUTER_TASK_B_DECL:.*]]:2 = hlfir.declare %[[OUTER_TASK_B]]
! CHECK:             omp.task private(@[[B_FIRSTPRIVATE]] %[[OUTER_TASK_B_DECL]]#0 -> %[[INNER_TASK_B:.*]] :
! CHECK:               hlfir.declare %[[INNER_TASK_B]]
! CHECK:               omp.terminator
! CHECK:             omp.terminator
! CHECK:           omp.terminator
subroutine test_nested_task_firstprivate
  integer :: b(2)
  b = 1
  !$omp parallel private(b)
    b = 2
    !$omp task
      !$omp task firstprivate(b)
      !$omp end task
    !$omp end task
  !$omp end parallel
end subroutine
