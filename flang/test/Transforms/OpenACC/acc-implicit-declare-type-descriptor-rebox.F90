!RUN: bbc -emit-fir -fopenacc %s -o - | fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-declare)" | FileCheck %s
! Test assumed-shape arguments to exercise fir.rebox with type descriptors

module mm
  type struct
    real :: member
  end type
contains
  subroutine acc_rout(struct_pointer_arr)
    !$acc routine
    ! Tests assumed-shape with dimensions changing (aka `fir.rebox`)
    type(struct), dimension(0:) :: struct_pointer_arr
    struct_pointer_arr(0)%member = 1.0
  end subroutine
end module

! CHECK-DAG: fir.rebox
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}member {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}struct {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}c{{.+}}struct {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}dt{{.+}}struct {acc.declare
