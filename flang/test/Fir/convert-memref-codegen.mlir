// RUN: not fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s -o - 2>&1 | FileCheck %s

// This test ensures that the FIR CodeGen ConvertOpConversion
// rejects fir.convert when either the source or the destination
// type is a memref (i.e. it fails to legalize those ops).

module {
  // CHECK: error: failed to legalize operation 'fir.convert'
  func.func @memref_to_ref_convert(%arg0: memref<f32>) {
    %0 = fir.convert %arg0 : (memref<f32>) -> !fir.ref<f32>
    return
  }
}


