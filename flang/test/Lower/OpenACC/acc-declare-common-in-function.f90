! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Verify that a COMMON block declared with OpenACC declare inside a function
! is lowered as a global declare (acc.global_ctor/dtor) rather than a
! structured declare.

program p
  implicit none
  real :: pi
  integer :: i
  common /COM/ pi
!$acc declare copyin(/COM/)
  data pi/0.0/

! CHECK-DAG: acc.global_ctor @{{.*}}_acc_ctor {
! CHECK-DAG: %[[ADDR0:.*]] = fir.address_of(@{{.*}}) {acc.declare = #acc.declare<dataClause = acc_copyin>} : {{.*}}
! CHECK-DAG: acc.declare_enter dataOperands(%{{.*}} : {{.*}})
! CHECK-DAG: acc.terminator
! CHECK-DAG: }

! CHECK-DAG: acc.global_dtor @{{.*}}_acc_dtor {
! CHECK-DAG: %[[ADDR1:.*]] = fir.address_of(@{{.*}}) {acc.declare = #acc.declare<dataClause = acc_copyin>} : !fir.ref<tuple<f32>>
! CHECK-DAG: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[ADDR1]] : !fir.ref<tuple<f32>>) -> !fir.ref<tuple<f32>> {dataClause = #acc<data_clause acc_copyin>, {{.*}}}
! CHECK-DAG: acc.declare_exit dataOperands(%[[GDP]] : !fir.ref<tuple<f32>>)
! CHECK-DAG: acc.delete accPtr(%[[GDP]] : !fir.ref<tuple<f32>>) {dataClause = #acc<data_clause acc_copyin>{{.*}}}
! CHECK-DAG: acc.terminator
! CHECK-DAG: }

contains

  subroutine s()
    implicit none
    real :: pi
    common /COM/ pi
!$acc declare copyin(/COM/)
  end subroutine s

end program p


