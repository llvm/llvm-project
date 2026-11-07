// RUN: fir-opt %s --fir-to-memref | FileCheck %s

// When both the embox and its array_coor consumer carry a slice, FIRToMemRef
// must synthesize reinterpret_cast strides from the *parent's* extents, not
// the box's (= slice's). For a slice that keeps > 1 element per dim, using
// the slice extents makes the outer stride walk by the slice's size instead
// of the parent's leading dim, so memref.load reads across the wrong columns.

func.func @emboxed_slice_stride(%arg0: !fir.ref<!fir.array<4x2xi32>>) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %ci = arith.constant 1 : i64
  %shape = fir.shape %c4, %c2 : (index, index) -> !fir.shape<2>
  // Embox slice: a(1:2, :) -- keeps 2 elements per dim (all steps 1, all lb 1).
  %eslice = fir.slice %c1, %c2, %c1, %c1, %c2, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
  %box = fir.embox %arg0(%shape) [%eslice]
      : (!fir.ref<!fir.array<4x2xi32>>, !fir.shape<2>, !fir.slice<2>)
      -> !fir.box<!fir.array<2x2xi32>>
  // Inner array_coor slice (identity 1:1:1 in both dims).
  %ashape = fir.shape %c2, %c2 : (index, index) -> !fir.shape<2>
  %islice = fir.slice %c1, %c1, %c1, %c1, %c1, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
  %addr = fir.array_coor %box(%ashape) [%islice] %ci, %c1
      : (!fir.box<!fir.array<2x2xi32>>, !fir.shape<2>, !fir.slice<2>, i64, index)
      -> !fir.ref<i32>
  %v = fir.load %addr : !fir.ref<i32>
  return %v : i32
}

// CHECK-LABEL: func.func @emboxed_slice_stride(
// CHECK-SAME:      %[[ARG0:.+]]: !fir.ref<!fir.array<4x2xi32>>) -> i32
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index

// Parent fir.shape and embox fir.slice survive.
// CHECK:       %[[SHAPE:.+]] = fir.shape %[[C4]], %[[C2]] : (index, index) -> !fir.shape<2>
// CHECK:       %[[ESLICE:.+]] = fir.slice %[[C1]], %[[C2]], %[[C1]], %[[C1]], %[[C2]], %[[C1]] : ({{.+}}) -> !fir.slice<2>
// CHECK:       fir.embox %[[ARG0]](%[[SHAPE]]) [%[[ESLICE]]] : ({{.+}}) -> !fir.box<!fir.array<2x2xi32>>

// Row-major memref view of the parent (col-major !fir.array<4x2>).
// CHECK:       %[[MEMREF:.+]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<4x2xi32>>) -> memref<2x4xi32>

// Anchor via the two `arith.constant 0 : index` uses (getMemrefIndices' zero
// placeholder, then reinterpret_cast's offset) -- the fold's `cOne` is
// emitted immediately after the second.
// CHECK:       arith.constant 0 : index
// CHECK:       arith.constant 0 : index
// CHECK-NEXT:  %[[CONE:.+]] = arith.constant 1 : index

// Fortran dim 0 (embox lb = 1, stride = 1) -> delta 0 added onto memref
// position 1.
// CHECK-NEXT:  %[[LB0_DELTA:.+]] = arith.subi %[[C1]], %[[CONE]] : index
// CHECK-NEXT:  %[[SCALED0:.+]] = arith.muli %{{.+}}, %[[C1]] : index
// CHECK-NEXT:  %[[IDX_DIM0:.+]] = arith.addi %[[SCALED0]], %[[LB0_DELTA]] : index

// Fortran dim 1 (embox lb = 1, stride = 1) -> delta 0 added onto memref
// position 0.
// CHECK-NEXT:  %[[LB1_DELTA:.+]] = arith.subi %[[C1]], %[[CONE]] : index
// CHECK-NEXT:  %[[SCALED1:.+]] = arith.muli %{{.+}}, %[[C1]] : index
// CHECK-NEXT:  %[[IDX_DIM1:.+]] = arith.addi %[[SCALED1]], %[[LB1_DELTA]] : index

// The stride check is the point of the test: sizes = slice extents (2, 2),
// strides = [parent's dim-0 element stride = 4, 1] in memref order. The
// outer stride MUST be %c4 (parent dim 0), not %c2 (slice's own extent).
// CHECK-NEXT:  memref.reinterpret_cast %[[MEMREF]] to offset: [%{{.+}}], sizes: [%[[C2]], %[[C2]]], strides: [%[[C4]], %{{.+}}] : memref<2x4xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>

// memref.load consumes the shifted indices in memref order.
// CHECK-NEXT:  memref.load %{{.+}}[%[[IDX_DIM1]], %[[IDX_DIM0]]] : memref<?x?xi32, strided<[?, ?], offset: ?>>
