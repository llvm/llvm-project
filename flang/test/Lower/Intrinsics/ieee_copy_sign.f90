! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QQmain()
use ieee_arithmetic
real(4) :: x = -2.0, y = huge(y)
real(8) :: z = 2.0

! CHECK-DAG: %[[X_ADDR:.*]] = fir.address_of(@_QFEx) : !fir.ref<f32>
! CHECK-DAG: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ADDR]] {uniq_name = "_QFEx"}
! CHECK-DAG: %[[Y_ADDR:.*]] = fir.address_of(@_QFEy) : !fir.ref<f32>
! CHECK-DAG: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y_ADDR]] {uniq_name = "_QFEy"}
! CHECK-DAG: %[[Z_ADDR:.*]] = fir.address_of(@_QFEz) : !fir.ref<f64>
! CHECK-DAG: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z_ADDR]] {uniq_name = "_QFEz"}

! CHECK-DAG: %[[X_VAL:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
! CHECK-DAG: %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f32>
! CHECK:     %[[CS1:.*]] = math.copysign %[[X_VAL]], %[[Y_VAL]] fastmath<contract> : f32
! CHECK:     fir.call @_FortranAioOutputReal32({{.*}}, %[[CS1]])

! CHECK-DAG: %[[Z_VAL:.*]] = fir.load %[[Z_DECL]]#0 : !fir.ref<f64>
! CHECK-DAG: %[[Y_VAL2:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f32>
! CHECK:     %[[BIT_Z:.*]] = arith.bitcast %[[Z_VAL]] : f64 to i64
! CHECK:     %[[BIT_Y:.*]] = arith.bitcast %[[Y_VAL2]] : f32 to i32
! CHECK:     %[[S_Y:.*]] = arith.shrui %[[BIT_Y]], %c31{{.*}} : i32
! CHECK:     %[[SL_Z:.*]] = arith.shli %[[BIT_Z]], %c1{{.*}} : i64
! CHECK:     %[[SR_Z:.*]] = arith.shrui %[[SL_Z]], %c1{{.*}} : i64
! CHECK-DAG: %[[S_BIT:.*]] = arith.shli %c1{{.*}}, %c63{{.*}} : i64
! CHECK-DAG: %[[IS_P:.*]] = arith.cmpi eq, %[[S_Y]], %c0{{.*}} : i32
! CHECK:     %[[S_VAL:.*]] = arith.select %[[IS_P]], %c0{{.*}}, %[[S_BIT]] : i64
! CHECK:     %[[RES_BIT:.*]] = arith.ori %[[SR_Z]], %[[S_VAL]] : i64
! CHECK:     %[[RES:.*]] = arith.bitcast %[[RES_BIT]] : i64 to f64
! CHECK:     fir.call @_FortranAioOutputReal64({{.*}}, %[[RES]])

print*, ieee_copy_sign(x,y), ieee_copy_sign(z,y)
end
