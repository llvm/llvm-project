! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test integer intrinsic operation lowering to fir.

! CHECK-LABEL: func.func @_QPeq0_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: %[[reg3:.*]] = arith.cmpi eq, %[[reg1]], %[[reg2]] : i32
! CHECK: fir.convert %[[reg3]] : (i1) -> !fir.logical<4>
LOGICAL FUNCTION eq0_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
eq0_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPne1_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: %[[reg3:.*]] = arith.cmpi ne, %[[reg1]], %[[reg2]] : i32
! CHECK: fir.convert %[[reg3]] : (i1) -> !fir.logical<4>
LOGICAL FUNCTION ne1_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
ne1_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPlt2_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: %[[reg3:.*]] = arith.cmpi slt, %[[reg1]], %[[reg2]] : i32
! CHECK: fir.convert %[[reg3]] : (i1) -> !fir.logical<4>
LOGICAL FUNCTION lt2_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
lt2_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPle3_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: %[[reg3:.*]] = arith.cmpi sle, %[[reg1]], %[[reg2]] : i32
! CHECK: fir.convert %[[reg3]] : (i1) -> !fir.logical<4>
LOGICAL FUNCTION le3_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
le3_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPgt4_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: %[[reg3:.*]] = arith.cmpi sgt, %[[reg1]], %[[reg2]] : i32
! CHECK: fir.convert %[[reg3]] : (i1) -> !fir.logical<4>
LOGICAL FUNCTION gt4_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
gt4_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPge5_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: %[[reg3:.*]] = arith.cmpi sge, %[[reg1]], %[[reg2]] : i32
! CHECK: fir.convert %[[reg3]] : (i1) -> !fir.logical<4>
LOGICAL FUNCTION ge5_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
ge5_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL: func.func @_QPadd6_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: arith.addi %[[reg1]], %[[reg2]] : i32
INTEGER(4) FUNCTION add6_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
add6_test = x0 + x1
END FUNCTION

! CHECK-LABEL: func.func @_QPsub7_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: arith.subi %[[reg1]], %[[reg2]] : i32
INTEGER(4) FUNCTION sub7_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
sub7_test = x0 - x1
END FUNCTION

! CHECK-LABEL: func.func @_QPmult8_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: arith.muli %[[reg1]], %[[reg2]] : i32
INTEGER(4) FUNCTION mult8_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
mult8_test = x0 * x1
END FUNCTION

! CHECK-LABEL: func.func @_QPdiv9_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32> {{.*}}, %[[arg1:.*]]: !fir.ref<i32> {{.*}})
! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG: %[[reg1:.*]] = fir.load %[[VAL_1]]#0
! CHECK-DAG: %[[reg2:.*]] = fir.load %[[VAL_2]]#0
! CHECK: arith.divsi %[[reg1]], %[[reg2]] : i32
INTEGER(4) FUNCTION div9_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
div9_test = x0 / x1
END FUNCTION
