// RUN: fir-opt -flang-licm='hoist-from-nested-regions=none' %s | FileCheck %s

// Verify that with hoist-from-nested-regions=none, fir.convert inside
// scf.if is NOT hoisted out of the loop.
// CHECK-LABEL: func.func @test_nested_no_hoist(
// CHECK:         scf.for
// CHECK:           scf.if
// CHECK:             fir.convert
// CHECK:             memref.store
// CHECK:           scf.if
// CHECK:             fir.convert
// CHECK:             memref.store
func.func @test_nested_no_hoist(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %cst1 = arith.constant 1.000000e+00 : f32
  %cst2 = arith.constant 2.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    %cmp1 = arith.cmpi slt, %i, %c5 : index
    scf.if %cmp1 {
      %mem = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
      memref.store %cst1, %mem[%i] : memref<10xf32>
    }
    %cmp2 = arith.cmpi sge, %i, %c5 : index
    scf.if %cmp2 {
      %mem = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
      memref.store %cst2, %mem[%i] : memref<10xf32>
    }
  }
  return
}
