! This test checks lowering of OpenMP Threadprivate Directive.
! Test for allocatable and pointer variables.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module test
  integer, pointer :: x(:), m
  real, allocatable :: y(:), n

  !$omp threadprivate(x, y, m, n)

!CHECK-DAG: fir.global @_QMtestEm : !fir.box<!fir.ptr<i32>> {
!CHECK-DAG: fir.global @_QMtestEn : !fir.box<!fir.heap<f32>> {
!CHECK-DAG: fir.global @_QMtestEx : !fir.box<!fir.ptr<!fir.array<?xi32>>> {
!CHECK-DAG: fir.global @_QMtestEy : !fir.box<!fir.heap<!fir.array<?xf32>>> {

contains
  subroutine sub()
!CHECK-DAG:  %[[M:.*]] = fir.address_of(@_QMtestEm) : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %[[M_DECL:.*]]:2 = hlfir.declare %[[M]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEm"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-DAG:  %[[OMP_M:.*]] = omp.threadprivate %[[M_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %[[OMP_M_DECL:.*]]:2 = hlfir.declare %[[OMP_M]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEm"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-DAG:  %[[N:.*]] = fir.address_of(@_QMtestEn) : !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  %[[N_DECL:.*]]:2 = hlfir.declare %[[N]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtestEn"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
!CHECK-DAG:  %[[OMP_N:.*]] = omp.threadprivate %[[N_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  %[[OMP_N_DECL:.*]]:2 = hlfir.declare %[[OMP_N]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtestEn"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
!CHECK-DAG:  %[[X:.*]] = fir.address_of(@_QMtestEx) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
!CHECK-DAG:  %[[OMP_X:.*]] = omp.threadprivate %[[X_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %[[OMP_X_DECL:.*]]:2 = hlfir.declare %[[OMP_X]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
!CHECK-DAG:  %[[Y:.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtestEy"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
!CHECK-DAG:  %[[OMP_Y:.*]] = omp.threadprivate %[[Y_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %[[OMP_Y_DECL:.*]]:2 = hlfir.declare %[[OMP_Y]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtestEy"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_X_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Y_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_M_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_N_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
    print *, x, y, m, n

    !$omp parallel
!CHECK-DAG:  %[[X_PVT:.*]] = omp.threadprivate %[[X_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %[[X_PVT_DECL:.*]]:2 = hlfir.declare %[[X_PVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
!CHECK-DAG:  %[[Y_PVT:.*]] = omp.threadprivate %[[Y_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %[[Y_PVT_DECL:.*]]:2 = hlfir.declare %[[Y_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtestEy"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
!CHECK-DAG:  %[[Z_PVT:.*]] = omp.threadprivate %[[M_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %[[Z_PVT_DECL:.*]]:2 = hlfir.declare %[[Z_PVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEm"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-DAG:  %[[N_PVT:.*]] = omp.threadprivate %[[N_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  %[[N_PVT_DECL:.*]]:2 = hlfir.declare %[[N_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtestEn"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
!CHECK-DAG:  %{{.*}} = fir.load %[[X_PVT_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[Y_PVT_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[Z_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[N_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
      print *, x, y, m, n
    !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_X_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Y_DECL]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_M_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_N_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
    print *, x, y, m, n
  end
end
