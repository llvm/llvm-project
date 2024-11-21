// RUN: fir-opt --cg-rewrite="preserve-declare=true" %s -o - | FileCheck %s --check-prefixes DECL
// RUN: fir-opt --cg-rewrite="preserve-declare=false" %s -o - | FileCheck %s --check-prefixes NODECL
// RUN: fir-opt --cg-rewrite %s -o - | FileCheck %s  --check-prefixes NODECL

func.func @dummy_scope(%arg0: !fir.ref<f32>) {
  %scope = fir.dummy_scope : !fir.dscope
  %0 = fir.declare %arg0 dummy_scope %scope {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  return
}
// DECL-LABEL: func.func @dummy_scope(
// DECL: fircg.ext_declare

// NODECL-LABEL: func.func @dummy_scope(
// NODECL-NEXT: return