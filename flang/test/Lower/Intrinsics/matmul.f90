! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test matmul intrinsic

! CHECK-LABEL: func.func @_QPmatmul_test(
! CHECK-SAME: %[[X:.*]]: !fir.ref<!fir.array<3x1xf32>>{{.*}}, %[[Y:.*]]: !fir.ref<!fir.array<1x3xf32>>{{.*}}, %[[Z:.*]]: !fir.ref<!fir.array<3x3xf32>>{{.*}})
subroutine matmul_test(x,y,z)
  real :: x(3,1), y(1,3), z(3,3)
! CHECK-DAG:  %[[XDECL:.*]]:2 = hlfir.declare %[[X]]
! CHECK-DAG:  %[[YDECL:.*]]:2 = hlfir.declare %[[Y]]
! CHECK-DAG:  %[[ZDECL:.*]]:2 = hlfir.declare %[[Z]]
! CHECK:  %[[RESULT:.*]] = hlfir.matmul %[[XDECL]]#0 %[[YDECL]]#0 {{.*}} : (!fir.ref<!fir.array<3x1xf32>>, !fir.ref<!fir.array<1x3xf32>>) -> !hlfir.expr<3x3xf32>
! CHECK:  hlfir.assign %[[RESULT]] to %[[ZDECL]]#0 : !hlfir.expr<3x3xf32>, !fir.ref<!fir.array<3x3xf32>>
! CHECK:  hlfir.destroy %[[RESULT]] : !hlfir.expr<3x3xf32>
  z = matmul(x,y)
end subroutine

! CHECK-LABEL: func.func @_QPmatmul_test2(
! CHECK-SAME: %[[X_BOX:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}}, %[[Y_BOX:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}}, %[[Z_BOX:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}})
subroutine matmul_test2(X, Y, Z)
  logical :: X(:,:)
  logical :: Y(:)
  logical :: Z(:)
! CHECK-DAG:  %[[XDECL:.*]]:2 = hlfir.declare %[[X_BOX]]
! CHECK-DAG:  %[[YDECL:.*]]:2 = hlfir.declare %[[Y_BOX]]
! CHECK-DAG:  %[[ZDECL:.*]]:2 = hlfir.declare %[[Z_BOX]]
! CHECK:  %[[RESULT:.*]] = hlfir.matmul %[[XDECL]]#0 %[[YDECL]]#0 {{.*}} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> !hlfir.expr<?x!fir.logical<4>>
! CHECK:  hlfir.assign %[[RESULT]] to %[[ZDECL]]#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  hlfir.destroy %[[RESULT]] : !hlfir.expr<?x!fir.logical<4>>
  Z = matmul(X, Y)
end subroutine
