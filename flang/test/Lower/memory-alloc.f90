! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QMw0bPtest1(
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMw0bECa"}
! CHECK: %[[SLICE:.*]] = hlfir.designate %[[A]]#0 (%{{.*}}:%{{.*}}:%{{.*}}, %{{.*}})  shape %{{.*}} : (!fir.ref<!fir.array<2x2xi32>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<2xi32>>
! CHECK: %[[EXPR:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<2x!fir.logical<4>>
! CHECK: %[[ANY:.*]] = hlfir.any %[[EXPR]] : (!hlfir.expr<2x!fir.logical<4>>) -> !fir.logical<4>
! CHECK: hlfir.destroy %[[EXPR]] : !hlfir.expr<2x!fir.logical<4>>

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
