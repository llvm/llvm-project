// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.dispatch` conversion to llvm.
// Not implemented yet.

func.func @dispatch(%arg0: !fir.class<!fir.type<derived3{f:f32}>>) {
// CHECK: not yet implemented: fir.class type conversion
  %0 = fir.dispatch "method"(%arg0 : !fir.class<!fir.type<derived3{f:f32}>>) -> i32
  return
}
