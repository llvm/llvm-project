!RUN: %bbc -emit-fir -fopenacc %s -o - | fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-declare)" | FileCheck %s
! Test nullify to exercise fir.type_desc with type descriptors

module mm
  type p1
    real :: field
  end type

  type struct
    class(p1), pointer :: member
  end type
end module

program main
  use mm
  class(struct), pointer :: struct_pointer

  ! Test extracting type descriptor to pass to runtime `fir.type_desc`
  !$acc serial
  nullify(struct_pointer%member)
  !$acc end serial

end program

! CHECK-DAG: fir.type_desc
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}field {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}n{{.+}}p1
! CHECK-DAG: @_QMmmE{{.+}}c{{.+}}p1 {acc.declare
! CHECK-DAG: @_QMmmE{{.+}}dt{{.+}}p1 {acc.declare
