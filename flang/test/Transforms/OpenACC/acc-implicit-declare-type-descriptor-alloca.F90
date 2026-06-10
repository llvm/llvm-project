!RUN: %bbc -emit-fir -fopenacc %s -o - | fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-declare)" | FileCheck %s
! Test block construct with local struct to exercise fir.alloca with type descriptors

module mm
  type struct
    real :: member
  end type
end module

program main
  use mm

  ! Test block with local struct to test local creation via `fir.alloca`
  !$acc kernels
  block
    type(struct) :: local_struct
    local_struct%member = 3.0
  end block
  !$acc end kernels

end program

! CHECK-DAG: fir.alloca !fir.type<_QMmmTstruct{member:f32}>
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}member {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}struct {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}c{{.+}}struct {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}dt{{.+}}struct {acc.declare
