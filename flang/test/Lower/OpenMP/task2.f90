!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPomp_task_nested_allocatable_firstprivate
subroutine omp_task_nested_allocatable_firstprivate
  integer, allocatable :: a(:)

  allocate(a(7))
  a = 10

!CHECK:       %[[A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>,
!CHECK-SAME:    uniq_name = "_QFomp_task_nested_allocatable_firstprivateEa"} :
!CHECK-SAME:    (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) ->
!CHECK-SAME:    (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK:       omp.task {
  !$omp task default(firstprivate)
!CHECK:         omp.task {
!CHECK:           %[[PRIV_A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>,
!CHECK-SAME:        uniq_name = "_QFomp_task_nested_allocatable_firstprivateEa"} :
!CHECK-SAME:        (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) ->
!CHECK-SAME:        (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!CHECK:           %[[TEMP:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK:           hlfir.assign %[[TEMP]] to %[[PRIV_A]]#0 realloc :
!CHECK-SAME:        !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    !$omp task default(firstprivate)
      a = 2
!CHECK:         }
    !$omp end task
!CHECK:       }
  !$omp end task
end subroutine
