! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: _QPbtest_test
function btest_test(i, j)
    logical btest_test
    ! CHECK-DAG: %[[result_alloca:.*]] = fir.alloca !fir.logical<4> {bindc_name = "btest_test"
    ! CHECK-DAG: %[[result:.*]]:2 = hlfir.declare %[[result_alloca]] {uniq_name = "_QFbtest_testEbtest_test"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    ! CHECK-DAG: %[[i_decl:.*]]:2 = hlfir.declare %arg0 {{.*}} {uniq_name = "_QFbtest_testEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    ! CHECK-DAG: %[[j_decl:.*]]:2 = hlfir.declare %arg1 {{.*}} {uniq_name = "_QFbtest_testEj"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    ! CHECK-DAG: %[[i:.*]] = fir.load %[[i_decl]]#0 : !fir.ref<i32>
    ! CHECK-DAG: %[[j:.*]] = fir.load %[[j_decl]]#0 : !fir.ref<i32>
    ! CHECK-DAG: %[[VAL_5:.*]] = arith.shrui %[[i]], %[[j]] : i32
    ! CHECK-DAG: %[[VAL_6:.*]] = arith.constant 1 : i32
    ! CHECK: %[[VAL_7:.*]] = arith.andi %[[VAL_5]], %[[VAL_6]] : i32
    ! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> !fir.logical<4>
    ! CHECK: hlfir.assign %[[VAL_8]] to %[[result]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    ! CHECK: %[[VAL_9:.*]] = fir.load %[[result]]#0 : !fir.ref<!fir.logical<4>>
    ! CHECK: return %[[VAL_9]] : !fir.logical<4>
    btest_test = btest(i, j)
  end
