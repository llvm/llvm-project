! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPibits_test(
! CHECK-SAME: %[[I_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[J_ARG:.*]]: !fir.ref<i32>{{.*}}, %[[K_ARG:.*]]: !fir.ref<i32>{{.*}}) -> i32 {
function ibits_test(i, j, k)
! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ARG]]
! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ARG]]
! CHECK-DAG: %[[K:.*]]:2 = hlfir.declare %[[K_ARG]]
! CHECK-DAG: %[[RESULT_VAR:.*]] = fir.alloca i32
! CHECK-DAG: %[[RESULT:.*]]:2 = hlfir.declare %[[RESULT_VAR]]
! CHECK-DAG: %[[I_VAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[J_VAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[K_VAL:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[C32:.*]] = arith.constant 32 : i32
! CHECK-DAG: %[[SUB:.*]] = arith.subi %[[C32]], %[[K_VAL]] : i32
! CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
! CHECK-DAG: %[[CN1:.*]] = arith.constant -1 : i32
! CHECK: %[[SHRUI:.*]] = arith.shrui %[[CN1]], %[[SUB]] : i32
! CHECK: %[[SHRSI:.*]] = arith.shrsi %[[I_VAL]], %[[J_VAL]] : i32
! CHECK: %[[AND:.*]] = arith.andi %[[SHRSI]], %[[SHRUI]] : i32
! CHECK: %[[CMPI:.*]] = arith.cmpi eq, %[[K_VAL]], %[[C0]] : i32
! CHECK: %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[AND]] : i32
! CHECK: hlfir.assign %[[SELECT]] to %[[RESULT]]#0 : i32, !fir.ref<i32>
! CHECK: %[[RET:.*]] = fir.load %[[RESULT]]#0 : !fir.ref<i32>
! CHECK: return %[[RET]] : i32
  ibits_test = ibits(i, j, k)
end
