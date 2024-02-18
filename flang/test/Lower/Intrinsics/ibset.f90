! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: ibset_test
function ibset_test(i, j)
  ! CHECK-DAG: %[[result:.*]] = fir.alloca i32 {bindc_name = "ibset_test"
  ! CHECK-DAG: %[[i:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[j:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_6:.*]] = arith.shli %[[VAL_5]], %[[j]] : i32
  ! CHECK: %[[VAL_7:.*]] = arith.ori %[[i]], %[[VAL_6]] : i32
  ! CHECK: fir.store %[[VAL_7]] to %[[result]] : !fir.ref<i32>
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[result]] : !fir.ref<i32>
  ! CHECK: return %[[VAL_8]] : i32
  ibset_test = ibset(i, j)
end

