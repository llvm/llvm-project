// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s

// Tests that `fir.local` ops are dropped from the module before LLVM lowering.

fir.local {type = local} @local_privatizer : i32
func.func @foo() {
  return
}

// CHECK-NOT: fir.local
