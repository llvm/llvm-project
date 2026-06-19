! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPishft_test(
! CHECK-SAME: %[[I_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME: %[[J_ARG:.*]]: !fir.ref<i32> {fir.bindc_name = "j"}) -> i32 {
function ishft_test(i, j)
! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[I_ARG]]
! CHECK-DAG: %[[result:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFishft_testEishft_test"}
! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[J_ARG]]
! CHECK-DAG: %[[iVAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[jVAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 32 : i32
! CHECK-DAG: %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK-DAG: %[[VAL_7:.*]] = arith.constant 31 : i32
! CHECK: %[[VAL_8:.*]] = arith.shrsi %[[jVAL]], %[[VAL_7]] : i32
! CHECK: %[[VAL_9:.*]] = arith.xori %[[jVAL]], %[[VAL_8]] : i32
! CHECK: %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_8]] : i32
! CHECK: %[[VAL_11:.*]] = arith.shli %[[iVAL]], %[[VAL_10]] : i32
! CHECK: %[[VAL_12:.*]] = arith.shrui %[[iVAL]], %[[VAL_10]] : i32
! CHECK: %[[VAL_13:.*]] = arith.cmpi sge, %[[VAL_10]], %[[VAL_5]] : i32
! CHECK: %[[VAL_14:.*]] = arith.cmpi slt, %[[jVAL]], %[[VAL_6]] : i32
! CHECK: %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_11]] : i32
! CHECK: %[[VAL_16:.*]] = arith.select %[[VAL_13]], %[[VAL_6]], %[[VAL_15]] : i32
! CHECK: hlfir.assign %[[VAL_16]] to %[[result]]#0 : i32, !fir.ref<i32>
! CHECK: %[[VAL_17:.*]] = fir.load %[[result]]#0 : !fir.ref<i32>
! CHECK: return %[[VAL_17]] : i32
  ishft_test = ishft(i, j)
end
