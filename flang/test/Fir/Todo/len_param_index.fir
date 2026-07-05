// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.len_param_index` conversion to llvm.
// Not implemented yet.

func.func @lenparamindex() {
  // CHECK: not yet implemented: fir.len_param_index codegen
  %0 = fir.len_param_index l1, !fir.type<twolens(l1:i32, l2:i32){i:i32, f:f32, l:i64}>
  return
}

