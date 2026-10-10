// Test the single-pass folding of constant array `fir.global` initializer
// bodies into dense constants. A partially initialized large array must be
// lowered to a compact constant instead of a long `llvm.insertvalue` chain
// (which is pathologically slow to materialize and fold back, see
// https://github.com/llvm/llvm-project/issues/209393).

// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck %s

// A 1D array initialized with a couple of leading elements and a trailing
// range folds to a bodyless dense global.
fir.global @oned : !fir.array<8xi32> {
  %c7 = arith.constant 7 : i32
  %c77 = arith.constant 77 : i32
  %c0 = arith.constant 0 : i32
  %0 = fir.undefined !fir.array<8xi32>
  %1 = fir.insert_value %0, %c7, [0 : index] : (!fir.array<8xi32>, i32) -> !fir.array<8xi32>
  %2 = fir.insert_value %1, %c77, [1 : index] : (!fir.array<8xi32>, i32) -> !fir.array<8xi32>
  %3 = fir.insert_on_range %2, %c0 from (2) to (7) : (!fir.array<8xi32>, i32) -> !fir.array<8xi32>
  fir.has_value %3 : !fir.array<8xi32>
}

// CHECK: llvm.mlir.global external @oned(dense<[7, 77, 0, 0, 0, 0, 0, 0]> : tensor<8xi32>) {{.*}} : !llvm.array<8 x i32>
// CHECK-NOT: llvm.insertvalue

// -----

// A 2D array with a mix of a single-element insert and a range insert folds to
// a dense global. The tensor shape is the reverse of the Fortran dimensions,
// and the element order is column-major (first Fortran dimension fastest).
fir.global @twod : !fir.array<3x2xi32> {
  %c9 = arith.constant 9 : i32
  %c0 = arith.constant 0 : i32
  %0 = fir.undefined !fir.array<3x2xi32>
  %1 = fir.insert_value %0, %c9, [0 : index, 0 : index] : (!fir.array<3x2xi32>, i32) -> !fir.array<3x2xi32>
  %2 = fir.insert_on_range %1, %c0 from (1,0) to (2,1) : (!fir.array<3x2xi32>, i32) -> !fir.array<3x2xi32>
  fir.has_value %2 : !fir.array<3x2xi32>
}

// CHECK: llvm.mlir.global external @twod(dense<{{\[}}[9, 0, 0], [0, 0, 0]]> : tensor<2x3xi32>) {{.*}} : !llvm.array<2 x array<3 x i32>>
// CHECK-NOT: llvm.insertvalue

// -----

// A 3D array fully initialized by a single range folds to a dense splat.
fir.global @threed : !fir.array<2x2x2xi32> {
  %c5 = arith.constant 5 : i32
  %0 = fir.undefined !fir.array<2x2x2xi32>
  %1 = fir.insert_on_range %0, %c5 from (0,0,0) to (1,1,1) : (!fir.array<2x2x2xi32>, i32) -> !fir.array<2x2x2xi32>
  fir.has_value %1 : !fir.array<2x2x2xi32>
}

// CHECK: llvm.mlir.global external @threed(dense<5> : tensor<2x2x2xi32>) {{.*}} : !llvm.array<2 x array<2 x array<2 x i32>>>
// CHECK-NOT: llvm.insertvalue

// -----

// A real (floating point) array folds as well.
fir.global @reals : !fir.array<4xf32> {
  %c1 = arith.constant 1.5 : f32
  %c0 = arith.constant 0.0 : f32
  %0 = fir.undefined !fir.array<4xf32>
  %1 = fir.insert_value %0, %c1, [0 : index] : (!fir.array<4xf32>, f32) -> !fir.array<4xf32>
  %2 = fir.insert_on_range %1, %c0 from (1) to (3) : (!fir.array<4xf32>, f32) -> !fir.array<4xf32>
  fir.has_value %2 : !fir.array<4xf32>
}

// CHECK: llvm.mlir.global external @reals(dense<[1.500000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<4xf32>) {{.*}} : !llvm.array<4 x f32>
// CHECK-NOT: llvm.insertvalue

// -----

// A logical array reaches the insertion through a fir.convert; the fold
// normalizes the value to a canonical 0/1 of the lowered element type.
fir.global @logicals : !fir.array<4x!fir.logical<4>> {
  %true = arith.constant true
  %0 = fir.undefined !fir.array<4x!fir.logical<4>>
  %1 = fir.convert %true : (i1) -> !fir.logical<4>
  %2 = fir.insert_on_range %0, %1 from (0) to (3) : (!fir.array<4x!fir.logical<4>>, !fir.logical<4>) -> !fir.array<4x!fir.logical<4>>
  fir.has_value %2 : !fir.array<4x!fir.logical<4>>
}

// CHECK: llvm.mlir.global external @logicals(dense<1> : tensor<4xi32>) {{.*}} : !llvm.array<4 x i32>
// CHECK-NOT: llvm.insertvalue

// -----

// A COMMON block lowers to a single-element tuple wrapping the array. A single
// struct-level llvm.mlir.constant cannot represent a struct containing an
// array, so the array member is materialized with one llvm.mlir.constant and
// inserted into the aggregate.
fir.global @common1 {alignment = 4 : i64} : tuple<!fir.array<8xi32>> {
  %c7 = arith.constant 7 : i32
  %c77 = arith.constant 77 : i32
  %c0 = arith.constant 0 : i32
  %0 = fir.zero_bits tuple<!fir.array<8xi32>>
  %1 = fir.undefined !fir.array<8xi32>
  %2 = fir.insert_value %1, %c7, [0 : index] : (!fir.array<8xi32>, i32) -> !fir.array<8xi32>
  %3 = fir.insert_value %2, %c77, [1 : index] : (!fir.array<8xi32>, i32) -> !fir.array<8xi32>
  %4 = fir.insert_on_range %3, %c0 from (2) to (7) : (!fir.array<8xi32>, i32) -> !fir.array<8xi32>
  %5 = fir.insert_value %0, %4, [0 : index] : (tuple<!fir.array<8xi32>>, !fir.array<8xi32>) -> tuple<!fir.array<8xi32>>
  fir.has_value %5 : tuple<!fir.array<8xi32>>
}

// CHECK: llvm.mlir.global external @common1() {{.*}} : !llvm.struct<(array<8 x i32>)> {
// CHECK:   %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(array<8 x i32>)>
// CHECK:   %[[CST:.*]] = llvm.mlir.constant(dense<[7, 77, 0, 0, 0, 0, 0, 0]> : tensor<8xi32>) : !llvm.array<8 x i32>
// CHECK:   %[[INS:.*]] = llvm.insertvalue %[[CST]], %[[UNDEF]][0] : !llvm.struct<(array<8 x i32>)>
// CHECK:   llvm.return %[[INS]] : !llvm.struct<(array<8 x i32>)>
// CHECK: }

// -----

// A COMMON block with several members: each member folds to one constant and
// is inserted at its field index.
fir.global @common2 : tuple<!fir.array<4xi32>, !fir.array<2xi32>> {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %0 = fir.zero_bits tuple<!fir.array<4xi32>, !fir.array<2xi32>>
  %1 = fir.undefined !fir.array<4xi32>
  %2 = fir.insert_on_range %1, %c1 from (0) to (3) : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
  %3 = fir.insert_value %0, %2, [0 : index] : (tuple<!fir.array<4xi32>, !fir.array<2xi32>>, !fir.array<4xi32>) -> tuple<!fir.array<4xi32>, !fir.array<2xi32>>
  %4 = fir.undefined !fir.array<2xi32>
  %5 = fir.insert_value %4, %c3, [0 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  %6 = fir.insert_value %5, %c0, [1 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  %7 = fir.insert_value %3, %6, [1 : index] : (tuple<!fir.array<4xi32>, !fir.array<2xi32>>, !fir.array<2xi32>) -> tuple<!fir.array<4xi32>, !fir.array<2xi32>>
  fir.has_value %7 : tuple<!fir.array<4xi32>, !fir.array<2xi32>>
}

// CHECK: llvm.mlir.global external @common2() {{.*}} : !llvm.struct<(array<4 x i32>, array<2 x i32>)> {
// CHECK:   %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(array<4 x i32>, array<2 x i32>)>
// CHECK:   %[[CST0:.*]] = llvm.mlir.constant(dense<1> : tensor<4xi32>) : !llvm.array<4 x i32>
// CHECK:   %[[INS0:.*]] = llvm.insertvalue %[[CST0]], %[[UNDEF]][0] : !llvm.struct<(array<4 x i32>, array<2 x i32>)>
// CHECK:   %[[CST1:.*]] = llvm.mlir.constant(dense<[3, 0]> : tensor<2xi32>) : !llvm.array<2 x i32>
// CHECK:   %[[INS1:.*]] = llvm.insertvalue %[[CST1]], %[[INS0]][1] : !llvm.struct<(array<4 x i32>, array<2 x i32>)>
// CHECK:   llvm.return %[[INS1]] : !llvm.struct<(array<4 x i32>, array<2 x i32>)>
// CHECK: }

// -----

// An array whose leading elements are left undefined (no zero base) has no
// constant to fold to for those elements: it must stay on the regular
// per-operation lowering rather than being force-folded.
fir.global @partialundef : !fir.array<8xi32> {
  %c1 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<8xi32>
  %1 = fir.insert_on_range %0, %c1 from (5) to (7) : (!fir.array<8xi32>, i32) -> !fir.array<8xi32>
  fir.has_value %1 : !fir.array<8xi32>
}

// CHECK: llvm.mlir.global external @partialundef() {{.*}} : !llvm.array<8 x i32> {
// CHECK:   llvm.insertvalue
// CHECK: }
