! This test checks lowering of OpenMP Threadprivate Directive.
! Test for allocatable and pointer variables.

!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

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
!CHECK-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QMtestEm) : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QMtestEn) : !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEx) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>
    print *, x, y, m, n

    !$omp parallel
!CHECK-DAG:    [[ADDR54:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:    [[ADDR55:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:    [[ADDR56:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:    [[ADDR57:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR56]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR57]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR54]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR55]] : !fir.ref<!fir.box<!fir.heap<f32>>>
      print *, x, y, m, n
    !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>
    print *, x, y, m, n
  end
end
