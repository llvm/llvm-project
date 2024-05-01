// RUN: fir-opt --cg-rewrite %s -o - | FileCheck %s

func.func @dummy_scope(%arg0: !fir.ref<f32>) {
  %scope = fir.dummy_scope : !fir.dscope
  %0 = fir.declare %arg0 dummy_scope %scope {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: func.func @dummy_scope(
// CHECK-NEXT: return
