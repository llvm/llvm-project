! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: ishft_test
function ishft_test(i, j)
    ! CHECK-DAG: %[[result:.*]] = fir.alloca i32 {bindc_name = "ishft_test"
    ! CHECK-DAG:  %[[i:.*]] = fir.load %arg0 : !fir.ref<i32>
    ! CHECK-DAG:  %[[j:.*]] = fir.load %arg1 : !fir.ref<i32>
    ! CHECK-DAG:  %[[VAL_5:.*]] = arith.constant 32 : i32
    ! CHECK-DAG:  %[[VAL_6:.*]] = arith.constant 0 : i32
    ! CHECK-DAG:  %[[VAL_7:.*]] = arith.constant 31 : i32
    ! CHECK:  %[[VAL_8:.*]] = arith.shrsi %[[j]], %[[VAL_7]] : i32
    ! CHECK:  %[[VAL_9:.*]] = arith.xori %[[j]], %[[VAL_8]] : i32
    ! CHECK:  %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_8]] : i32
    ! CHECK:  %[[VAL_11:.*]] = arith.shli %[[i]], %[[VAL_10]] : i32
    ! CHECK:  %[[VAL_12:.*]] = arith.shrui %[[i]], %[[VAL_10]] : i32
    ! CHECK:  %[[VAL_13:.*]] = arith.cmpi sge, %[[VAL_10]], %[[VAL_5]] : i32
    ! CHECK:  %[[VAL_14:.*]] = arith.cmpi slt, %[[j]], %[[VAL_6]] : i32
    ! CHECK:  %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_11]] : i32
    ! CHECK:  %[[VAL_16:.*]] = arith.select %[[VAL_13]], %[[VAL_6]], %[[VAL_15]] : i32
    ! CHECK:  fir.store %[[VAL_16]] to %[[result]] : !fir.ref<i32>
    ! CHECK:  %[[VAL_17:.*]] = fir.load %[[result]] : !fir.ref<i32>
    ! CHECK:  return %[[VAL_17]] : i32
    ishft_test = ishft(i, j)
  end
