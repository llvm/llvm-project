// Merging two blocks that differ only in the complex part index passes that
// index as a block argument, so it is no longer constant. A GEP index into a
// struct has to be constant, so index the element type instead.

// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}),fir-to-llvm-ir)' | FileCheck %s

func.func private @use(f32)

// CHECK-LABEL: llvm.func @complex_part_block_argument
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:         %[[IDX:.*]] = llvm.select
// CHECK:         llvm.getelementptr %[[ARG0]][%[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
func.func @complex_part_block_argument(%z: !fir.ref<complex<f32>>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %re = fir.coordinate_of %z, %c0 : (!fir.ref<complex<f32>>, index) -> !fir.ref<f32>
  %rev = fir.load %re : !fir.ref<f32>
  func.call @use(%rev) : (f32) -> ()
  cf.br ^exit
^bb2:
  %im = fir.coordinate_of %z, %c1 : (!fir.ref<complex<f32>>, index) -> !fir.ref<f32>
  %imv = fir.load %im : !fir.ref<f32>
  func.call @use(%imv) : (f32) -> ()
  cf.br ^exit
^exit:
  return
}
