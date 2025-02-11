!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s


!CHECK-LABEL: omp.private
!CHECK-SAME:      {type = firstprivate} @[[PRIVATIZER:.*]] : !fir.box<!fir.heap<!fir.array<?xi32>>> init {
!CHECK:         fir.if
!CHECK:       } copy {
!CHECK:         fir.if
!CHECK:       } dealloc {
!CHECK:         fir.if
!CHECK:       }

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
!CHECK:         omp.task private(@[[PRIVATIZER]] %[[A]]#0 -> %[[A_ARG:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
!CHECK:           %[[PRIV_A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>,
!CHECK-SAME:        uniq_name = "_QFomp_task_nested_allocatable_firstprivateEa"} :
!CHECK-SAME:        (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) ->
!CHECK-SAME:        (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
    !$omp task default(firstprivate)
      a = 2
!CHECK:         }
    !$omp end task
!CHECK:       }
  !$omp end task
end subroutine
