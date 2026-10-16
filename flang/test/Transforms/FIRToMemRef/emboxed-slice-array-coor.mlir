// RUN: fir-opt %s --fir-to-memref | FileCheck %s

// Verify that when both the embox and the array_coor carry a slice, the
// embox's per-dim (lb - 1) shift is folded into the memref indices.

func.func @emboxed_slice_array_coor(%arg0: !fir.ref<!fir.array<4x2xi32>>) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %ci = arith.constant 1 : i64
  %shape = fir.shape %c4, %c2 : (index, index) -> !fir.shape<2>
  // Embox slice: a(2:2, 1:1) -- Fortran dim-0 lb = 2, dim-1 lb = 1.
  %eslice = fir.slice %c2, %c2, %c1, %c1, %c1, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
  %box = fir.embox %arg0(%shape) [%eslice]
      : (!fir.ref<!fir.array<4x2xi32>>, !fir.shape<2>, !fir.slice<2>)
      -> !fir.box<!fir.array<1x1xi32>>
  // Inner array_coor slice (identity 1:1:1 in both dims).
  %ashape = fir.shape %c1, %c1 : (index, index) -> !fir.shape<2>
  %islice = fir.slice %c1, %c1, %c1, %c1, %c1, %c1
      : (index, index, index, index, index, index) -> !fir.slice<2>
  %addr = fir.array_coor %box(%ashape) [%islice] %ci, %c1
      : (!fir.box<!fir.array<1x1xi32>>, !fir.shape<2>, !fir.slice<2>, i64, index)
      -> !fir.ref<i32>
  %v = fir.load %addr : !fir.ref<i32>
  return %v : i32
}

// CHECK-LABEL: func.func @emboxed_slice_array_coor(
// CHECK-SAME:      %[[ARG0:.+]]: !fir.ref<!fir.array<4x2xi32>>) -> i32
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index

// Parent fir.shape and embox fir.slice survive to the emitted module.
// CHECK:       %[[SHAPE:.+]] = fir.shape %[[C4]], %[[C2]] : (index, index) -> !fir.shape<2>
// CHECK:       %[[ESLICE:.+]] = fir.slice %[[C2]], %[[C2]], %[[C1]], %[[C1]], %[[C1]], %[[C1]] : ({{.+}}) -> !fir.slice<2>
// CHECK:       fir.embox %[[ARG0]](%[[SHAPE]]) [%[[ESLICE]]] : ({{.+}}) -> !fir.box<!fir.array<1x1xi32>>

// Row-major memref view of the parent (col-major !fir.array<4x2>).
// CHECK:       %[[MEMREF:.+]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<4x2xi32>>) -> memref<2x4xi32>

// Anchor: `arith.constant 0 : index` appears exactly twice between the
// fir.convert and the fold -- once for getMemrefIndices' zero placeholder,
// then again as the reinterpret_cast offset. The fold's `cOne` is emitted
// immediately after the second.
// CHECK:       arith.constant 0 : index
// CHECK:       arith.constant 0 : index
// CHECK-NEXT:  %[[CONE:.+]] = arith.constant 1 : index

// Fortran dim 0 (embox lb = 2, stride = 1) -> delta 1, added onto memref
// position 1.
// CHECK-NEXT:  %[[LB0_DELTA:.+]] = arith.subi %[[C2]], %[[CONE]] : index
// CHECK-NEXT:  %[[SCALED0:.+]] = arith.muli %{{.+}}, %[[C1]] : index
// CHECK-NEXT:  %[[IDX_DIM0:.+]] = arith.addi %[[SCALED0]], %[[LB0_DELTA]] : index

// Fortran dim 1 (embox lb = 1, stride = 1) -> delta 0, added onto memref
// position 0.
// CHECK-NEXT:  %[[LB1_DELTA:.+]] = arith.subi %[[C1]], %[[CONE]] : index
// CHECK-NEXT:  %[[SCALED1:.+]] = arith.muli %{{.+}}, %[[C1]] : index
// CHECK-NEXT:  %[[IDX_DIM1:.+]] = arith.addi %[[SCALED1]], %[[LB1_DELTA]] : index

// reinterpret_cast: offset stays a plain 0 constant, sizes = slice extents
// (1, 1), strides = [parent's dim-0 element stride, 1] in memref order.
// CHECK-NEXT:  memref.reinterpret_cast %[[MEMREF]] to offset: [%{{.+}}], sizes: [%[[C1]], %[[C1]]], strides: [%[[C4]], %{{.+}}] : memref<2x4xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>

// memref.load consumes the shifted indices in memref order
// (dim 1 outer -> IDX_DIM1, dim 0 inner -> IDX_DIM0).
// CHECK-NEXT:  memref.load %{{.+}}[%[[IDX_DIM1]], %[[IDX_DIM0]]] : memref<?x?xi32, strided<[?, ?], offset: ?>>
