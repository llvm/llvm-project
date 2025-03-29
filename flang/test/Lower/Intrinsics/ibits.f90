! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: ibits_test
function ibits_test(i, j, k)
  ! CHECK-DAG: %[[result:.*]] = fir.alloca i32 {bindc_name = "ibits_test"
  ! CHECK-DAG: %[[i:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[j:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK-DAG: %[[k:.*]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK-DAG: %[[VAL_7:.*]] = arith.constant 32 : i32
  ! CHECK-DAG: %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[k]] : i32
  ! CHECK-DAG: %[[VAL_9:.*]] = arith.constant 0 : i32
  ! CHECK-DAG: %[[VAL_10:.*]] = arith.constant -1 : i32
  ! CHECK: %[[VAL_11:.*]] = arith.shrui %[[VAL_10]], %[[VAL_8]] : i32
  ! CHECK: %[[VAL_12:.*]] = arith.shrsi %[[i]], %[[j]] : i32
  ! CHECK: %[[VAL_13:.*]] = arith.andi %[[VAL_12]], %[[VAL_11]] : i32
  ! CHECK: %[[VAL_14:.*]] = arith.cmpi eq, %[[k]], %[[VAL_9]] : i32
  ! CHECK: %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_9]], %[[VAL_13]] : i32
  ! CHECK: fir.store %[[VAL_15]] to %[[result]] : !fir.ref<i32>
  ! CHECK: %[[VAL_16:.*]] = fir.load %[[result]] : !fir.ref<i32>
  ! CHECK: return %[[VAL_16]] : i32
  ibits_test = ibits(i, j, k)
end
