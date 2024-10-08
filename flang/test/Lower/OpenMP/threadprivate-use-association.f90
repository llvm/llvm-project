! This test checks lowering of OpenMP Threadprivate Directive.
! Test for threadprivate variable in use association.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-DAG: fir.global common @blk_(dense<0> : vector<24xi8>) {alignment = 4 : i64} : !fir.array<24xi8>
!CHECK-DAG: fir.global @_QMtestEy : f32 {

module test
  integer :: x
  real :: y, z(5)
  common /blk/ x, z

  !$omp threadprivate(y, /blk/)

contains
  subroutine sub()
! CHECK-LABEL: @_QMtestPsub
!CHECK-DAG:   [[ADDR0:%.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:   [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:   [[ADDR1:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!CHECK-DAG:   %[[DECY:.*]]:2 = hlfir.declare [[ADDR1]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:   [[NEWADDR1:%.*]] = omp.threadprivate %[[DECY]]#1 : !fir.ref<f32> -> !fir.ref<f32>

    !$omp parallel
!CHECK-DAG:    [[ADDR2:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[ADDR3:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR4:%.*]] = fir.coordinate_of [[ADDR3]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR5:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK-DAG:    %[[ADDR6:.*]]:2 = hlfir.declare [[ADDR5]] {uniq_name = "_QMtestEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:    [[NEWADDR2:%.*]] = omp.threadprivate %[[DECY]]#1 : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:    %[[ADDR7:.*]]:2 = hlfir.declare [[NEWADDR2]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:    [[ADDR8:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR9:%.*]] = fir.coordinate_of [[ADDR8]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR10:%.*]] = fir.convert [[ADDR9]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!CHECK-DAG:    %[[ADDR11:.*]]:2 = hlfir.declare [[ADDR10]](%{{.*}}) {uniq_name = "_QMtestEz"} : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xf32>>, !fir.ref<!fir.array<5xf32>>)
!CHECK-DAG:    %{{.*}} = fir.load %[[ADDR6]]#0 : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.load %[[ADDR7]]#0 : !fir.ref<f32>
!CHECK-DAG:    %{{.*}} = fir.embox %[[ADDR11]]#1(%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>
      print *, x, y, z
    !$omp end parallel
  end
end

program main
  use test
  integer :: x1
  real :: z1(5)
  common /blk/ x1, z1

  !$omp threadprivate(/blk/)

  call sub()

! CHECK-LABEL: @_QQmain()
!CHECK-DAG:    [[ADDR0:%.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[ADDR1:%.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[ADDR2:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!CHECK-DAG:    %[[ADDR3:.*]]:2 = hlfir.declare [[ADDR2]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:    [[NEWADDR2:%.*]] = omp.threadprivate %[[ADDR3]]#1 : !fir.ref<f32> -> !fir.ref<f32>

  !$omp parallel
!CHECK-DAG:    [[ADDR4:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[ADDR6:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR7:%.*]] = fir.coordinate_of [[ADDR6]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR8:%.*]] = fir.convert [[ADDR7]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK-DAG:    %[[DECX1:.*]]:2 = hlfir.declare [[ADDR8]] {uniq_name = "_QFEx1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:    [[ADDR5:%.*]] = omp.threadprivate %[[ADDR3]]#1 : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:    %[[DECY:.*]]:2 = hlfir.declare [[ADDR5]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:    [[ADDR9:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR10:%.*]] = fir.coordinate_of [[ADDR9]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR11:%.*]] = fir.convert [[ADDR10]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!CHECK-DAG:    %[[DECZ1:.*]]:2 = hlfir.declare [[ADDR11]](%{{.*}}) {uniq_name = "_QFEz1"} : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xf32>>, !fir.ref<!fir.array<5xf32>>)
!CHECK-DAG:    %{{.*}} = fir.load %[[DECX1]]#0 : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.load %[[DECY]]#0 : !fir.ref<f32>
!CHECK-DAG:    %{{.*}} = fir.embox %[[DECZ1]]#1(%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>
    print *, x1, y, z1
  !$omp end parallel

end
