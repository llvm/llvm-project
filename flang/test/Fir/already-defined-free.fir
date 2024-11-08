// Test that FIR codegen handles cases when free and malloc have
// already been defined in FIR (either by the user in Fortran via
// BIND(C) or by some FIR pass in between).
// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s


func.func @already_declared_free_malloc() {
  %c4 = arith.constant 4 : index
  %0 = fir.call @malloc(%c4) : (index) -> !fir.heap<i32>
  fir.call @free(%0) : (!fir.heap<i32>) -> ()
  %1 = fir.allocmem i32
  fir.freemem %1 : !fir.heap<i32>
  return
}

// CHECK: llvm.call @malloc(%{{.*}})
// CHECK: llvm.call @free(%{{.*}})
// CHECK: llvm.call @malloc(%{{.*}})
// CHECK: llvm.call @free(%{{.*}})

func.func private @free(!fir.heap<i32>)
func.func private @malloc(index) -> !fir.heap<i32>
