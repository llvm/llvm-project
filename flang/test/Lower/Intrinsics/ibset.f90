! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPibset_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32> {{.*}}, %[[ARG1:.*]]: !fir.ref<i32> {{.*}})
function ibset_test(i, j)
  ! CHECK-DAG: %[[I:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} arg 1 {uniq_name = "_QFibset_testEi"}
  ! CHECK-DAG: %[[J:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} arg 2 {uniq_name = "_QFibset_testEj"}
  ! CHECK-DAG: %[[RESULT_ALLOC:.*]] = fir.alloca i32 {bindc_name = "ibset_test", uniq_name = "_QFibset_testEibset_test"}
  ! CHECK-DAG: %[[RESULT:.*]]:2 = hlfir.declare %[[RESULT_ALLOC]] {uniq_name = "_QFibset_testEibset_test"}
  ! CHECK-DAG: %[[I_VAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[J_VAL:.*]] = fir.load %[[J]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_6:.*]] = arith.shli %[[C1]], %[[J_VAL]] : i32
  ! CHECK: %[[VAL_7:.*]] = arith.ori %[[I_VAL]], %[[VAL_6]] : i32
  ! CHECK: hlfir.assign %[[VAL_7]] to %[[RESULT]]#0 : i32, !fir.ref<i32>
  ! CHECK: %[[RET_VAL:.*]] = fir.load %[[RESULT]]#0 : !fir.ref<i32>
  ! CHECK: return %[[RET_VAL]] : i32
  ibset_test = ibset(i, j)
end
