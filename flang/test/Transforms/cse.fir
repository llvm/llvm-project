// RUN: fir-opt -cse %s | FileCheck %s

// Verify that fir.dummy_scope writing to the DebuggingResource
// does not prevent CSE for the two fir.load's:
// CHECK-LABEL:   func.func @dummy_scope(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<f32>) -> (f32, f32) {
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_0]] : !fir.ref<f32>
// CHECK:           %[[DUMMY_SCOPE_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ARG1]] dummy_scope %[[DUMMY_SCOPE_1]] {uniq_name = "y"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           return %[[LOAD_0]], %[[LOAD_0]] : f32, f32
// CHECK:         }
func.func @dummy_scope(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>) -> (f32, f32) {
  %scope1 = fir.dummy_scope : !fir.dscope
  %0 = fir.declare %arg0 dummy_scope %scope1 {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %1 = fir.load %0 : !fir.ref<f32>
  %scope2 = fir.dummy_scope : !fir.dscope
  %2 = fir.declare %arg1 dummy_scope %scope2 {uniq_name = "y"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %1, %3 : f32, f32
}
