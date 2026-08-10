! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test logical intrinsic operation lowering to fir.

! CHECK-LABEL: func.func @_QPeqv0_test
LOGICAL(1) FUNCTION eqv0_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK: %[[X0:.*]]:2 = hlfir.declare %arg0
! CHECK: %[[X1:.*]]:2 = hlfir.declare %arg1
! CHECK: %[[V0:.*]] = fir.load %[[X0]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[V1:.*]] = fir.load %[[X1]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[EQV:.*]] = fir.eqv %[[V0]], %[[V1]] : !fir.logical<1>
! CHECK: hlfir.assign %[[EQV]]
eqv0_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPneqv1_test
LOGICAL(1) FUNCTION neqv1_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK: %[[X0:.*]]:2 = hlfir.declare %arg0
! CHECK: %[[X1:.*]]:2 = hlfir.declare %arg1
! CHECK: %[[V0:.*]] = fir.load %[[X0]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[V1:.*]] = fir.load %[[X1]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[NEQV:.*]] = fir.neqv %[[V0]], %[[V1]] : !fir.logical<1>
! CHECK: hlfir.assign %[[NEQV]]
neqv1_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPor2_test
LOGICAL(1) FUNCTION or2_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK: %[[X0:.*]]:2 = hlfir.declare %arg0
! CHECK: %[[X1:.*]]:2 = hlfir.declare %arg1
! CHECK: %[[V0:.*]] = fir.load %[[X0]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[V1:.*]] = fir.load %[[X1]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[OR:.*]] = fir.logical_or %[[V0]], %[[V1]] : !fir.logical<1>
! CHECK: hlfir.assign %[[OR]]
or2_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPand3_test
LOGICAL(1) FUNCTION and3_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK: %[[X0:.*]]:2 = hlfir.declare %arg0
! CHECK: %[[X1:.*]]:2 = hlfir.declare %arg1
! CHECK: %[[V0:.*]] = fir.load %[[X0]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[V1:.*]] = fir.load %[[X1]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %[[AND:.*]] = fir.logical_and %[[V0]], %[[V1]] : !fir.logical<1>
! CHECK: hlfir.assign %[[AND]]
and3_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPnot4_test
LOGICAL(1) FUNCTION not4_test(x0)
LOGICAL(1) :: x0
! CHECK: %[[X0:.*]]:2 = hlfir.declare %arg0
! CHECK: %[[V0:.*]] = fir.load %[[X0]]#0 : !fir.ref<!fir.logical<1>>
! CHECK: %true = arith.constant true
! CHECK: %[[B0:.*]] = fir.convert %[[V0]] : (!fir.logical<1>) -> i1
! CHECK: %[[NOT:.*]] = arith.xori %[[B0]], %true : i1
! CHECK: %[[RES:.*]] = fir.convert %[[NOT]] : (i1) -> !fir.logical<1>
! CHECK: hlfir.assign %[[RES]]
not4_test = .NOT. x0
END FUNCTION
