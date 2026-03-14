// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.allocmem` of derived type with LEN parameters conversion to llvm.
// Not implemented yet.

func.func @allocmem_test(%arg0 : i32, %arg1 : i16) {
// CHECK: not yet implemented: fir.allocmem codegen of derived type with length parameters
  %0 = fir.allocmem !fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>(%arg0, %arg1 : i32, i16) {name = "_QEvar"}
  return 
}
