// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.global_len` conversion to llvm.
// Not implemented yet.

fir.global @global_derived : !fir.type<minez(f:i32)> {
// CHECK: not yet implemented: fir.global_len codegen
  fir.global_len f, 1 : i32
  %0 = fir.undefined !fir.type<minez>
  fir.has_value %0 : !fir.type<minez>
}
