// RUN: fir-opt %s --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %s

// ShapeOpConversion: live fir.shape lowers to an n-field LLVM struct.
// CHECK-LABEL: llvm.func @live_shape(
// CHECK-SAME: %[[N:.+]]: i64
// CHECK: %[[UNDEF:.+]] = llvm.mlir.undef
// CHECK: %[[STRUCT:.+]] = llvm.insertvalue %[[N]], %[[UNDEF]][0]
// CHECK: llvm.extractvalue %[[STRUCT]][0]
// CHECK-NOT: fir.shape
// CHECK-NOT: fir.shape_extents
// CHECK: llvm.return
func.func @live_shape(%n : index) {
  %sh = fir.shape %n : (index) -> !fir.shape<1>
  %e = fir.shape_extents %sh : (!fir.shape<1>) -> index
  %c0 = arith.constant 0 : index
  %sink = arith.addi %e, %c0 : index
  return
}

// -----

// ShapeShiftOpConversion: packs extent operands only (n-field struct, not 2n).
// ShapeExtentsOpConversion: extractvalue field 0 for rank-1 shapeshift.
// CHECK-LABEL: llvm.func @live_shape_shift(
// CHECK-SAME: %[[LB:.+]]: i64, %[[EXT:.+]]: i64
// CHECK: %[[UNDEF:.+]] = llvm.mlir.undef
// CHECK: %[[STRUCT:.+]] = llvm.insertvalue %[[EXT]], %[[UNDEF]][0]
// CHECK-NOT: llvm.insertvalue %[[LB]]
// CHECK: llvm.extractvalue %[[STRUCT]][0]
// CHECK-NOT: fir.shape_shift
// CHECK-NOT: fir.shape_extents
// CHECK: llvm.return
func.func @live_shape_shift(%lb : index, %ext : index) {
  %ss = fir.shape_shift %lb, %ext : (index, index) -> !fir.shapeshift<1>
  %e = fir.shape_extents %ss : (!fir.shapeshift<1>) -> index
  %c0 = arith.constant 0 : index
  %sink = arith.addi %e, %c0 : index
  return
}

// -----

// ShapeExtentsOpConversion on a forwarded !fir.shape block argument.
// CHECK-LABEL: llvm.func @live_shape_extents_forwarded(
// CHECK-SAME: %[[PRED:.+]]: i1, %[[N1:.+]]: i64, %[[N2:.+]]: i64
// CHECK: llvm.cond_br %[[PRED]]
// CHECK: %[[UNDEF1:.+]] = llvm.mlir.undef
// CHECK: %[[S1:.+]] = llvm.insertvalue %[[N1]], %[[UNDEF1]][0]
// CHECK: llvm.br {{.*}}(%[[S1]]
// CHECK: %[[UNDEF2:.+]] = llvm.mlir.undef
// CHECK: %[[S2:.+]] = llvm.insertvalue %[[N2]], %[[UNDEF2]][0]
// CHECK: llvm.br {{.*}}(%[[S2]]
// CHECK: llvm.extractvalue %{{.*}}[0]
// CHECK-NOT: fir.shape
// CHECK-NOT: fir.shape_extents
// CHECK: llvm.return
func.func @live_shape_extents_forwarded(%pred : i1, %n1 : index, %n2 : index) {
  cf.cond_br %pred, ^bb1, ^bb2
^bb1:
  %sh1 = fir.shape %n1 : (index) -> !fir.shape<1>
  cf.br ^bb3(%sh1 : !fir.shape<1>)
^bb2:
  %sh2 = fir.shape %n2 : (index) -> !fir.shape<1>
  cf.br ^bb3(%sh2 : !fir.shape<1>)
^bb3(%phi : !fir.shape<1>):
  %e = fir.shape_extents %phi : (!fir.shape<1>) -> index
  %c0 = arith.constant 0 : index
  %sink = arith.addi %e, %c0 : index
  return
}

// -----

// 2-D shape: struct has two extent fields; shape_extents extracts [0] and [1].
// CHECK-LABEL: llvm.func @live_shape_extents_2d(
// CHECK: llvm.insertvalue {{.+}}[0]
// CHECK: llvm.insertvalue {{.+}}[1]
// CHECK: llvm.extractvalue {{.+}}[0]
// CHECK: llvm.extractvalue {{.+}}[1]
// CHECK-NOT: fir.shape
// CHECK-NOT: fir.shape_extents
func.func @live_shape_extents_2d(%n1 : index, %n2 : index) {
  %sh = fir.shape %n1, %n2 : (index, index) -> !fir.shape<2>
  %e0, %e1 = fir.shape_extents %sh : (!fir.shape<2>) -> (index, index)
  %c0 = arith.constant 0 : index
  %s0 = arith.addi %e0, %c0 : index
  %s1 = arith.addi %e1, %c0 : index
  return
}
