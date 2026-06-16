! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine lge_test
    character*3 :: c1(3)
    character*7 :: c2(3)
    ! CHECK: BeginExternalListOutput
    ! CHECK: hlfir.elemental
    ! CHECK: hlfir.cmpchar sge
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, lge(c1, c2)
    ! CHECK: BeginExternalListOutput
    ! CHECK: hlfir.elemental
    ! CHECK: hlfir.cmpchar sgt
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, lgt(c1, c2)
    ! CHECK: BeginExternalListOutput
    ! CHECK: hlfir.elemental
    ! CHECK: hlfir.cmpchar sle
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, lle(c1, c2)
    ! CHECK: BeginExternalListOutput
    ! CHECK: hlfir.elemental
    ! CHECK: hlfir.cmpchar slt
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, llt(c1, c2)
  end
