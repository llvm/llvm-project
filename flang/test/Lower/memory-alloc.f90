! RUN: bbc -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: func @_QMw0bPtest1(
! CHECK: %[[TWO:.*]] = arith.constant 2 : index
! CHECK: %[[HEAP:.*]] = fir.allocmem !fir.array<?x!fir.logical<4>>, %[[TWO]] {uniq_name = ".array.expr"}
! CHECK: fir.freemem %[[HEAP]] : !fir.heap<!fir.array<?x!fir.logical<4>>>

Module w0b
  Integer,Parameter :: a(*,*) = Reshape( [ 1,2,3,4 ], [ 2,2 ])
contains
  Subroutine test1(i,expect)
    Integer,Intent(In) :: i,expect(:)
    Logical :: ok = .True.
    If (Any(a(:,i)/=expect)) Then
      !Print *,'FAIL 1:',a(:,i),'/=',expect
      ok = .False.
    End If
  End Subroutine
End Module
