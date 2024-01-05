! RUN: bbc -hlfir=false %s -o "-" | FileCheck %s

! Test real intrinsic operation lowering to FIR.

! CHECK-LABEL:eq0_test
LOGICAL FUNCTION eq0_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpf oeq, [[reg1]], [[reg2]] {{.*}} : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq0_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:ne1_test
LOGICAL FUNCTION ne1_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpf une, [[reg1]], [[reg2]] {{.*}} : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne1_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:lt2_test
LOGICAL FUNCTION lt2_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpf olt, [[reg1]], [[reg2]] {{.*}} : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt2_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:le3_test
LOGICAL FUNCTION le3_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpf ole, [[reg1]], [[reg2]] {{.*}} : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le3_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:gt4_test
LOGICAL FUNCTION gt4_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpf ogt, [[reg1]], [[reg2]] {{.*}} : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt4_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:ge5_test
LOGICAL FUNCTION ge5_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = arith.cmpf oge, [[reg1]], [[reg2]] {{.*}} : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge5_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:add6_test
REAL(4) FUNCTION add6_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:addf [[reg1]], [[reg2]] {{.*}}: f32
add6_test = x0 + x1
END FUNCTION

! CHECK-LABEL:sub7_test
REAL(4) FUNCTION sub7_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:subf [[reg1]], [[reg2]] {{.*}}: f32
sub7_test = x0 - x1
END FUNCTION

! CHECK-LABEL:mult8_test
REAL(4) FUNCTION mult8_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:mulf [[reg1]], [[reg2]] {{.*}}: f32
mult8_test = x0 * x1
END FUNCTION

! CHECK-LABEL:div9_test
REAL(4) FUNCTION div9_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:divf [[reg1]], [[reg2]] {{.*}}: f32
div9_test = x0 / x1
END FUNCTION
