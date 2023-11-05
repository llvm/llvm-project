! Test that character literals of different types do not clash.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine repro(c1, c4)
  character(kind=1,len=*) :: c1
  character(kind=4,len=*) :: c4
  print *, ""
  print *, 4_""
end subroutine
!CHECK-LABEL: func.func @_QPrepro
!CHECK:    fir.address_of(@_QQcl_) : !fir.ref<!fir.char<1,0>>
!CHECK:    fir.call @_FortranAioOutputAscii
!CHECK:    fir.address_of(@_QQcl4_) : !fir.ref<!fir.char<4,0>>
!CHECK:    fir.call @_FortranAioOutputDescriptor(

!CHECK-DAG:  fir.global linkonce @_QQcl_ constant : !fir.char<1,0>
!CHECK-DAG:  fir.global linkonce @_QQcl4_ constant : !fir.char<4,0>
