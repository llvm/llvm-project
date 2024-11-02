! This test checks lowering of OpenMP Threadprivate Directive.
! Test for threadprivate variable in use association.

!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

!CHECK-DAG: fir.global common @_QBblk(dense<0> : vector<24xi8>) : !fir.array<24xi8>
!CHECK-DAG: fir.global @_QMtestEy : f32 {

module test
  integer :: x
  real :: y, z(5)
  common /blk/ x, z

  !$omp threadprivate(y, /blk/)

contains
  subroutine sub()
! CHECK-LABEL: @_QMtestPsub
!CHECK-DAG:   [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:   [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:   [[ADDR1:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!CHECK-DAG:   [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>

    !$omp parallel
!CHECK-DAG:    [[ADDR2:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[ADDR3:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:    [[ADDR4:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR5:%.*]] = fir.coordinate_of [[ADDR4]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR6:%.*]] = fir.convert [[ADDR5:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK-DAG:    [[ADDR7:%.*]] = fir.convert [[ADDR2]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR8:%.*]] = fir.coordinate_of [[ADDR7]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR9:%.*]] = fir.convert [[ADDR8:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR6]] : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR3]] : !fir.ref<f32>
!CHECK-DAG:    %{{.*}} = fir.embox [[ADDR9]](%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>
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
!CHECK-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QBblk) : !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
!CHECK-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<f32> -> !fir.ref<f32>

  !$omp parallel
!CHECK-DAG:    [[ADDR4:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.array<24xi8>> -> !fir.ref<!fir.array<24xi8>>
!CHECK-DAG:    [[ADDR5:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:    [[ADDR6:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR7:%.*]] = fir.coordinate_of [[ADDR6]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR8:%.*]] = fir.convert [[ADDR7:%.*]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK-DAG:    [[ADDR9:%.*]] = fir.convert [[ADDR4]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK-DAG:    [[ADDR10:%.*]] = fir.coordinate_of [[ADDR9]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR11:%.*]] = fir.convert [[ADDR10:%.*]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR8]] : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR5]] : !fir.ref<f32>
!CHECK-DAG:    %{{.*}} = fir.embox [[ADDR11]](%{{.*}}) : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf32>>
    print *, x1, y, z1
  !$omp end parallel

end
