// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.gentypedesc` conversion to llvm.
// Not implemented yet.

func.func @gentypedesc() {
// CHECK: not yet implemented: fir.gentypedesc codegen
  %0 = fir.gentypedesc !fir.type<derived3>
  return
}
