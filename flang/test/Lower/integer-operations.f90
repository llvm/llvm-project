! RUN: bbc %s -o "-" | FileCheck %s

! Test integer intrinsic operation lowering to fir.

! CHECK-LABEL:eq0_test
LOGICAL FUNCTION eq0_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpi eq, [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq0_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:ne1_test
LOGICAL FUNCTION ne1_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpi ne, [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne1_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:lt2_test
LOGICAL FUNCTION lt2_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpi slt, [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt2_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:le3_test
LOGICAL FUNCTION le3_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpi sle, [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le3_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:gt4_test
LOGICAL FUNCTION gt4_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpi sgt, [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt4_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:ge5_test
LOGICAL FUNCTION ge5_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpi sge, [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge5_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:add6_test
INTEGER(4) FUNCTION add6_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg1]], [[reg2]] : i32
add6_test = x0 + x1
END FUNCTION

! CHECK-LABEL:sub7_test
INTEGER(4) FUNCTION sub7_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg1]], [[reg2]] : i32
sub7_test = x0 - x1
END FUNCTION

! CHECK-LABEL:mult8_test
INTEGER(4) FUNCTION mult8_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg1]], [[reg2]] : i32
mult8_test = x0 * x1
END FUNCTION

! CHECK-LABEL:div9_test
INTEGER(4) FUNCTION div9_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:arith.divsi [[reg1]], [[reg2]] : i32
div9_test = x0 / x1
END FUNCTION
