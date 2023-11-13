! This test checks lowering of OpenMP Threadprivate Directive.
! Test for real, logical, complex, and derived type.

!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

module test
  type my_type
    integer :: t_i
    real :: t_arr(5)
  end type my_type
  real :: x
  complex :: y
  logical :: z
  type(my_type) :: t

  !$omp threadprivate(x, y, z, t)

!CHECK-DAG: fir.global @_QMtestEt : !fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}> {
!CHECK-DAG: fir.global @_QMtestEx : f32 {
!CHECK-DAG: fir.global @_QMtestEy : !fir.complex<4> {
!CHECK-DAG: fir.global @_QMtestEz : !fir.logical<4> {

contains
  subroutine sub()
!CHECK-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QMtestEt) : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!CHECK-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>> -> !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!CHECK-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QMtestEx) : !fir.ref<f32>
!CHECK-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!CHECK-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QMtestEz) : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.coordinate_of [[NEWADDR0]]
    print *, x, y, z, t%t_i

    !$omp parallel
!CHECK-DAG:    [[ADDR38:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>> -> !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!CHECK-DAG:    [[ADDR39:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:    [[ADDR40:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!CHECK-DAG:    [[ADDR41:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR39]] : !fir.ref<f32>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR40]] : !fir.ref<!fir.complex<4>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR41]] : !fir.ref<!fir.logical<4>>
!CHECK-DAG:    %{{.*}} = fir.coordinate_of [[ADDR38]]
      print *, x, y, z, t%t_i
    !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.coordinate_of [[NEWADDR0]]
    print *, x, y, z, t%t_i

  end
end
