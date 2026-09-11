// Test fir.create_box code generation.
// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// The byte stride operand is written into the descriptor verbatim: unlike
// fir.embox, there is no scaling by the element size (no llvm.mul), so a
// non-element-multiple stride is representable. elem_len/type still come from
// the result box element type.

func.func @create_box_strided(%base: !fir.heap<!fir.array<?xcomplex<f32>>>, %lb: index, %ext: index, %sm: index) {
  %0 = fir.create_box %base lbs(%lb) extents(%ext) strides(%sm) : (!fir.heap<!fir.array<?xcomplex<f32>>>, index, index, index) -> !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
  return
}
// The stride operand (%[[SM]]) is inserted at dims[0].sm ([7, 0, 2]) verbatim,
// i.e. the value bound to the block argument, not a product of it and the
// element size. Combined with CHECK-NOT: llvm.mul this proves no scaling.
// CHECK-LABEL: llvm.func @create_box_strided(
// CHECK-SAME:      %[[BASE:[^:]+]]: !llvm.ptr, %[[LB:[^:]+]]: i64, %[[EXT:[^:]+]]: i64, %[[SM:[^:]+]]: i64
// CHECK-NOT:       llvm.mul
// CHECK:           llvm.insertvalue %[[LB]], %{{.*}}[7, 0, 0] :
// CHECK:           llvm.insertvalue %[[EXT]], %{{.*}}[7, 0, 1] :
// CHECK:           llvm.insertvalue %[[SM]], %{{.*}}[7, 0, 2] :
// CHECK:           llvm.insertvalue %[[BASE]], %{{.*}}[0] :

// -----

func.func @create_box_strided_2d(%base: !fir.ref<!fir.array<?x?xf32>>, %lb0: index, %ext0: index, %sm0: index, %lb1: index, %ext1: index, %sm1: index) {
  %0 = fir.create_box %base lbs(%lb0, %lb1) extents(%ext0, %ext1) strides(%sm0, %sm1) : (!fir.ref<!fir.array<?x?xf32>>, index, index, index, index, index, index) -> !fir.box<!fir.array<?x?xf32>>
  return
}
// CHECK-LABEL: llvm.func @create_box_strided_2d(
// CHECK-SAME:      %[[BASE:[^:]+]]: !llvm.ptr, %[[LB0:[^:]+]]: i64, %[[EXT0:[^:]+]]: i64, %[[SM0:[^:]+]]: i64, %[[LB1:[^:]+]]: i64, %[[EXT1:[^:]+]]: i64, %[[SM1:[^:]+]]: i64
// CHECK-NOT:       llvm.mul
// CHECK:           llvm.insertvalue %[[LB0]], %{{.*}}[7, 0, 0] :
// CHECK:           llvm.insertvalue %[[EXT0]], %{{.*}}[7, 0, 1] :
// CHECK:           llvm.insertvalue %[[SM0]], %{{.*}}[7, 0, 2] :
// CHECK:           llvm.insertvalue %[[LB1]], %{{.*}}[7, 1, 0] :
// CHECK:           llvm.insertvalue %[[EXT1]], %{{.*}}[7, 1, 1] :
// CHECK:           llvm.insertvalue %[[SM1]], %{{.*}}[7, 1, 2] :
// CHECK:           llvm.insertvalue %[[BASE]], %{{.*}}[0] :
