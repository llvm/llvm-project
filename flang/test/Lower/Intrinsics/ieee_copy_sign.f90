! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
use ieee_arithmetic
real(4) :: x = -2.0, y = huge(y)
real(8) :: z = 2.0

! CHECK-DAG: %[[V_0:[0-9]+]] = fir.address_of(@_QFEx) : !fir.ref<f32>
! CHECK-DAG: %[[V_1:[0-9]+]] = fir.address_of(@_QFEy) : !fir.ref<f32>
! CHECK-DAG: %[[V_2:[0-9]+]] = fir.address_of(@_QFEz) : !fir.ref<f64>

! CHECK-DAG: %[[V_6:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<f32>
! CHECK-DAG: %[[V_7:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<f32>
! CHECK:     %[[V_8:[0-9]+]] = math.copysign %[[V_6]], %[[V_7]] fastmath<contract> : f32
! CHECK:     %[[V_9:[0-9]+]] = fir.call @_FortranAioOutputReal32(%{{.*}}, %[[V_8]]) fastmath<contract> : (!fir.ref<i8>, f32) -> i1

! CHECK-DAG: %[[V_10:[0-9]+]] = fir.load %[[V_2]] : !fir.ref<f64>
! CHECK-DAG: %[[V_11:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<f32>
! CHECK:     %[[V_12:[0-9]+]] = arith.bitcast %[[V_10]] : f64 to i64
! CHECK:     %[[V_13:[0-9]+]] = arith.bitcast %[[V_11]] : f32 to i32
! CHECK:     %[[V_14:[0-9]+]] = arith.shrui %[[V_13]], %c31{{.*}} : i32
! CHECK:     %[[V_15:[0-9]+]] = arith.shli %[[V_12]], %c1{{.*}} : i64
! CHECK:     %[[V_16:[0-9]+]] = arith.shrui %[[V_15]], %c1{{.*}} : i64
! CHECK-DAG: %[[V_17:[0-9]+]] = arith.shli %c1{{.*}}, %c63{{.*}} : i64
! CHECK-DAG: %[[V_18:[0-9]+]] = arith.cmpi eq, %[[V_14]], %c0{{.*}} : i32
! CHECK:     %[[V_19:[0-9]+]] = arith.select %[[V_18]], %c0{{.*}}, %[[V_17]] : i64
! CHECK:     %[[V_20:[0-9]+]] = arith.ori %[[V_16]], %[[V_19]] : i64
! CHECK:     %[[V_21:[0-9]+]] = arith.bitcast %[[V_20]] : i64 to f64
! CHECK:     %[[V_22:[0-9]+]] = fir.call @_FortranAioOutputReal64(%{{.*}}, %[[V_21]]) fastmath<contract> : (!fir.ref<i8>, f64) -> i1

print*, ieee_copy_sign(x,y), ieee_copy_sign(z,y)
end
