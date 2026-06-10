!RUN: %bbc -emit-fir -fopenacc %s -o - | fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-declare)" | FileCheck %s
! Test pointer association to exercise fir.embox with type descriptors

module mm
  type struct
    real :: member
  end type
end module

program main
  use mm
  type(struct), target :: static_struct
  type(struct), pointer :: struct_pointer

  ! Test pointer association (to test `fir.embox`)
  !$acc serial
  struct_pointer => static_struct
  !$acc end serial

end program

! CHECK-DAG: fir.embox
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}member {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}struct {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}c{{.+}}struct {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}dt{{.+}}struct {acc.declare
