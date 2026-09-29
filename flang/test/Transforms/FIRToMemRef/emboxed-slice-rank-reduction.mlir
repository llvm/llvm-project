// RUN: fir-opt %s --fir-to-memref | FileCheck %s

// When the fir.embox slice has a rank-reducing scalar-subscript triple
// (undef ub/step), the corresponding parent Fortran dim is collapsed in the
// box's rank but still has a memref index position under the reinterpret_cast
// (getRankFromEmbox returns the parent rank). The fold OVERWRITES that index
// with (lb - 1) instead of adding to it; the memref stride carried by that
// slot times (lb - 1) reaches the correct parent element, without needing to
// touch the reinterpret_cast's flat offset.

func.func @rank_reduced_embox_slice(
    %arg0: !fir.ref<!fir.array<3x2x2xi32>>, %i: index, %j: index) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %undef = fir.undefined index
  %shape = fir.shape %c3, %c2, %c2 : (index, index, index) -> !fir.shape<3>
  // Embox slice: a(:, 2, :) -- range/scalar/range. Collapses parent dim 1
  // (Fortran dim 1) at lb = 2. Result is a rank-2 fir.box.
  %eslice = fir.slice %c1, %c3, %c1, %c2, %undef, %undef, %c1, %c2, %c1
      : (index, index, index, index, index, index, index, index, index)
      -> !fir.slice<3>
  %box = fir.embox %arg0(%shape) [%eslice]
      : (!fir.ref<!fir.array<3x2x2xi32>>, !fir.shape<3>, !fir.slice<3>)
      -> !fir.box<!fir.array<3x2xi32>>
  // Array_coor with its own rank-2 slice on the box + 2 indices.
  %ashape = fir.shape %c3, %c2 : (index, index) -> !fir.shape<2>
  %ai = arith.index_cast %i : index to i64
  %islice = fir.slice %c1, %c3, %c1, %j, %undef, %undef
      : (index, index, index, index, index, index) -> !fir.slice<2>
  %addr = fir.array_coor %box(%ashape) [%islice] %ai, %j
      : (!fir.box<!fir.array<3x2xi32>>, !fir.shape<2>, !fir.slice<2>, i64, index)
      -> !fir.ref<i32>
  %v = fir.load %addr : !fir.ref<i32>
  return %v : i32
}

// CHECK-LABEL: func.func @rank_reduced_embox_slice(
// CHECK-SAME:      %[[ARG0:.+]]: !fir.ref<!fir.array<3x2x2xi32>>,
// CHECK-SAME:      %[[I:.+]]: index, %[[J:.+]]: index) -> i32
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK:       %[[UNDEF:.+]] = fir.undefined index

// Parent (rank-3) fir.shape and rank-reducing fir.slice survive.
// CHECK:       %[[SHAPE:.+]] = fir.shape %[[C3]], %[[C2]], %[[C2]] : (index, index, index) -> !fir.shape<3>
// CHECK:       %[[ESLICE:.+]] = fir.slice %[[C1]], %[[C3]], %[[C1]], %[[C2]], %[[UNDEF]], %[[UNDEF]], %[[C1]], %[[C2]], %[[C1]] : ({{.+}}) -> !fir.slice<3>
// CHECK:       fir.embox %[[ARG0]](%[[SHAPE]]) [%[[ESLICE]]] : ({{.+}}) -> !fir.box<!fir.array<3x2xi32>>

// Row-major memref view of the parent (col-major !fir.array<3x2x2>).
// CHECK:       %[[MEMREF:.+]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<3x2x2xi32>>) -> memref<2x2x3xi32>

// Anchor via the two `arith.constant 0 : index` uses (getMemrefIndices' zero
// placeholder, then reinterpret_cast's offset) -- the fold's `cOne` is
// emitted immediately after the second.
// CHECK:       arith.constant 0 : index
// CHECK:       arith.constant 0 : index
// CHECK-NEXT:  %[[CONE:.+]] = arith.constant 1 : index

// Fortran dim 0 (embox slice range, lb = 1, stride = 1) -> delta 0 added onto
// memref position 2.
// CHECK-NEXT:  %[[LB0_DELTA:.+]] = arith.subi %[[C1]], %[[CONE]] : index
// CHECK-NEXT:  %[[SCALED0:.+]] = arith.muli %{{.+}}, %[[C1]] : index
// CHECK-NEXT:  %[[IDX_DIM0:.+]] = arith.addi %[[SCALED0]], %[[LB0_DELTA]] : index

// Fortran dim 1 (embox slice SCALAR, lb = 2) -> delta 1 OVERWRITES memref
// position 1 (no arith.muli or arith.addi -- the delta itself becomes the
// index).
// CHECK-NEXT:  %[[LB1_DELTA:.+]] = arith.subi %[[C2]], %[[CONE]] : index

// Fortran dim 2 (embox slice range, lb = 1, stride = 1) -> delta 0 added onto
// memref position 0.
// CHECK-NEXT:  %[[LB2_DELTA:.+]] = arith.subi %[[C1]], %[[CONE]] : index
// CHECK-NEXT:  %[[SCALED2:.+]] = arith.muli %{{.+}}, %[[C1]] : index
// CHECK-NEXT:  %[[IDX_DIM2:.+]] = arith.addi %[[SCALED2]], %[[LB2_DELTA]] : index

// reinterpret_cast: offset stays 0 (never touched), sizes = parent extents in
// memref order (3, 2, 3) via getMemrefIndices' rank override, strides =
// parent's col-major element strides in memref order (2*3, 3, 1).
// CHECK-NEXT:  memref.reinterpret_cast %[[MEMREF]] to offset: [%{{.+}}], sizes: [%[[C3]], %[[C2]], %[[C3]]], strides: [%{{.+}}, %[[C3]], %{{.+}}] : memref<2x2x3xi32> to memref<?x?x?xi32, strided<[?, ?, ?], offset: ?>>

// memref.load: memref-order indices. Position 1 (collapsed dim) is
// LB1_DELTA directly (not through arith.addi).
// CHECK-NEXT:  memref.load %{{.+}}[%[[IDX_DIM2]], %[[LB1_DELTA]], %[[IDX_DIM0]]] : memref<?x?x?xi32, strided<[?, ?, ?], offset: ?>>
