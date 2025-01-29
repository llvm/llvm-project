// RUN: fir-opt %s | fir-opt | FileCheck %s
// RUN: fir-opt %s | fir-opt -cse | FileCheck %s

// CHECK-LABEL:   func.func @dummy_scope(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<f32>) {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           return
// CHECK:         }
func.func @dummy_scope(%arg0: !fir.ref<f32>) {
  %scope = fir.dummy_scope : !fir.dscope
  %0 = fir.declare %arg0 dummy_scope %scope {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  return
}

// CHECK-LABEL:   func.func @dummy_scopes(
// CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<f32>) {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_3]] {uniq_name = "innerEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_5]] {uniq_name = "innerEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           return
// CHECK:         }
func.func @dummy_scopes(%arg0: !fir.ref<f32>) {
  %scope_out = fir.dummy_scope : !fir.dscope
  %0 = fir.declare %arg0 dummy_scope %scope_out {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %scope_in1 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %scope_in1 {uniq_name = "innerEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %scope_in2 = fir.dummy_scope : !fir.dscope
  %2 = fir.declare %arg0 dummy_scope %scope_in2 {uniq_name = "innerEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  return
}
