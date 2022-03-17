! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: btest_test
function btest_test(i, j)
    logical btest_test
    ! CHECK-DAG: %[[result:[0-9]+]] = fir.alloca !fir.logical<4> {bindc_name = "btest_test"
    ! CHECK-DAG: %[[i:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
    ! CHECK-DAG: %[[j:[0-9]+]] = fir.load %arg1 : !fir.ref<i32>
    ! CHECK-DAG: %[[VAL_5:.*]] = arith.shrui %[[i]], %[[j]] : i32
    ! CHECK-DAG: %[[VAL_6:.*]] = arith.constant 1 : i32
    ! CHECK: %[[VAL_7:.*]] = arith.andi %[[VAL_5]], %[[VAL_6]] : i32
    ! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> !fir.logical<4>
    ! CHECK: fir.store %[[VAL_8]] to %[[result]] : !fir.ref<!fir.logical<4>>
    ! CHECK: %[[VAL_9:.*]] = fir.load %[[result]] : !fir.ref<!fir.logical<4>>
    ! CHECK: return %[[VAL_9]] : !fir.logical<4>
    btest_test = btest(i, j)
  end
  