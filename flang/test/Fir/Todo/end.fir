// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.end` conversion to llvm.
// Not implemented yet.

func.func @end_test() {
// CHECK: not yet implemented fir.end codegen
  "fir.end"() : () -> ()
}
