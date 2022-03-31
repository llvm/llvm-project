! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine lge_test
    character*3 :: c1(3)
    character*7 :: c2(3)
    ! c1(1) = 'a'; c1(2) = 'B'; c1(3) = 'c';
    ! c2(1) = 'A'; c2(2) = 'b'; c2(3) = 'c';
    ! CHECK: BeginExternalListOutput
    ! CHECK: fir.do_loop
    ! CHECK: CharacterCompareScalar1
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, lge(c1, c2)
    ! CHECK: BeginExternalListOutput
    ! CHECK: fir.do_loop
    ! CHECK: CharacterCompareScalar1
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, lgt(c1, c2)
    ! CHECK: BeginExternalListOutput
    ! CHECK: fir.do_loop
    ! CHECK: CharacterCompareScalar1
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, lle(c1, c2)
    ! CHECK: BeginExternalListOutput
    ! CHECK: fir.do_loop
    ! CHECK: CharacterCompareScalar1
    ! CHECK: OutputDescriptor
    ! CHECK: EndIoStatement
    print*, llt(c1, c2)
  end
  