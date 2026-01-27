// RUN: fir-opt %s -affine-parallelize | FileCheck %s

// Check that a read/write to a unit DebuggingResource
// by fir.dummy_scope does not prevent parallelization:
func.func @write_to_debugging_resource() {
  %cst = arith.constant 0.0 : f32
  affine.for %i = 0 to 100 {
    %m = memref.alloc() : memref<1xf32>
    affine.store %cst, %m[0] : memref<1xf32>
    %scope = fir.dummy_scope : !fir.dscope
  }
  // CHECK: affine.parallel
  return
}
