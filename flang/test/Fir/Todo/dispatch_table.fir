// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test fir.dispatch_table conversion to llvm.
// Not implemented yet.

// CHECK: not yet implemented: fir.dispatch_table codegen
fir.dispatch_table @dispatch_tbl {
  fir.dt_entry "method", @method_impl
}
