! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ibclr_test
function ibclr_test(i, j)
  ! CHECK-DAG: %[[result:.*]] = fir.alloca i32 {bindc_name = "ibclr_test"
  ! CHECK-DAG: %[[i:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[j:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[VAL_6:.*]] = arith.constant -1 : i32
  ! CHECK: %[[VAL_7:.*]] = arith.shli %[[VAL_5]], %[[j]] : i32
  ! CHECK: %[[VAL_8:.*]] = arith.xori %[[VAL_6]], %[[VAL_7]] : i32
  ! CHECK: %[[VAL_9:.*]] = arith.andi %[[i]], %[[VAL_8]] : i32
  ! CHECK: fir.store %[[VAL_9]] to %[[result]] : !fir.ref<i32>
  ! CHECK: %[[VAL_10:.*]] = fir.load %[[result]] : !fir.ref<i32>
  ! CHECK: return %[[VAL_10]] : i32
  ibclr_test = ibclr(i, j)
end

