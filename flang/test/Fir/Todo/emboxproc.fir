// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.emboxproc` conversion to llvm.
// Not implemented yet.

func @emboxproc_test() {
  %host_vars = fir.alloca tuple<i32,f64>
// CHECK: not yet implemented fir.emboxproc codegen
  %bproc = fir.emboxproc @method_impl, %host_vars : ((i32) -> (), !fir.ref<tuple<i32,f64>>) -> !fir.boxproc<(i32) -> ()>
  return
}
