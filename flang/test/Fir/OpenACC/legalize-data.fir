// RUN: fir-opt -split-input-file --openacc-legalize-data %s | FileCheck %s

func.func @_QPsub1(%arg0: !fir.ref<i32> {fir.bindc_name = "i"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1 = acc.copyin varPtr(%0#0 : !fir.ref<i32>) -> !fir.ref<i32> {dataClause = #acc<data_clause acc_copy>, name = "i"}
  acc.parallel dataOperands(%1 : !fir.ref<i32>) {
    %c0_i32 = arith.constant 0 : i32
    hlfir.assign %c0_i32 to %0#0 : i32, !fir.ref<i32>
    acc.yield
  }
  acc.copyout accPtr(%1 : !fir.ref<i32>) to varPtr(%0#0 : !fir.ref<i32>) {dataClause = #acc<data_clause acc_copy>, name = "i"}
  return
}

// CHECK-LABEL: func.func @_QPsub1
// CHECK-SAME: (%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"})
// CHECK: %[[I:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[I]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {dataClause = #acc<data_clause acc_copy>, name = "i"}
// CHECK: acc.parallel dataOperands(%[[COPYIN]] : !fir.ref<i32>) {
// CHECK:   %c0_i32 = arith.constant 0 : i32
// CHECK:   hlfir.assign %c0{{.*}} to %[[COPYIN]] : i32, !fir.ref<i32>
// CHECK:   acc.yield
// CHECK: }
// CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<i32>) to varPtr(%[[I]]#0 : !fir.ref<i32>) {dataClause = #acc<data_clause acc_copy>, name = "i"}
