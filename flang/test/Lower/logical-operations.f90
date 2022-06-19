! RUN: bbc %s -o "-" | FileCheck %s

! Test logical intrinsic operation lowering to fir.

! CHECK-LABEL:eqv0_test
LOGICAL(1) FUNCTION eqv0_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = arith.cmpi eq, [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
eqv0_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:neqv1_test
LOGICAL(1) FUNCTION neqv1_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = arith.cmpi ne, [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
neqv1_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:or2_test
LOGICAL(1) FUNCTION or2_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = arith.ori [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
or2_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:and3_test
LOGICAL(1) FUNCTION and3_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = arith.andi [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
and3_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:not4_test
LOGICAL(1) FUNCTION not4_test(x0)
LOGICAL(1) :: x0
! CHECK:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK:[[reg3:%[0-9]+]] = arith.xori [[reg2]], %true
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<1>
not4_test = .NOT. x0
END FUNCTION
