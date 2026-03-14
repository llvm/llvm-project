// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// `!fir.coordinate_of` - derived type with `!fir.len_param_index`. As
// `!fir.len_param_index` is not implemented yet, the error that we hit is
// related to `!fir.len_param_index` rather than `!fir.coordinate_of`.
func.func @coordinate_box_derived_with_fir_len(%arg0: !fir.box<!fir.type<derived_2{len1:i32}>>) {
// CHECK: not yet implemented: fir.len_param_index codegen
  %e = fir.len_param_index len1, !fir.type<derived_2{len1:i32}>
  %q = fir.coordinate_of %arg0, %e : (!fir.box<!fir.type<derived_2{len1:i32}>>, !fir.len) -> !fir.ref<i32>
  return
}
