! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Verify that 'declare present' on a COMMON block inside a function is lowered
! through the structured declare region (acc.present + declare enter/exit),
! and does not stamp the fir.global common with acc.declare=acc_present.

program p
  implicit none
  real :: pi
  common /COM/ pi
contains
  subroutine s()
    implicit none
    real :: pi
    common /COM/ pi
!$acc declare present(/COM/)
! CHECK: fir.global common @com_(dense<0> : vector<4xi8>) {alignment = 4 : i64} : !fir.array<4xi8>
! CHECK-LABEL: func.func private @_QFPs()
! CHECK-DAG: hlfir.declare
! CHECK-DAG: %[[ADDR:.*]] = fir.address_of(@com_){{.*}} : !fir.ref<!fir.array<4xi8>>
! CHECK-DAG: %[[PRESENT:.*]] = acc.present varPtr(%[[ADDR]] : !fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<4xi8>> {name = "com"}
! CHECK-DAG: %[[TOK:.*]] = acc.declare_enter dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<4xi8>>)
! CHECK: acc.declare_exit token(%[[TOK]]) dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<4xi8>>)
  end subroutine s
end program p


