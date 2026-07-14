// RUN: fir-opt %s --fir-to-memref | FileCheck %s

// Rank-reducing embox slice where the box's visible extents do NOT share a
// prefix with the parent's extents.  Parent is 4x3x2 (Fortran col-major),
// dim 1 is collapsed at lb=2, leaving box rank 2 with extents [4, 2].
//
// The parent's dim-1 extent is 3 while the box's dim-1 extent is 2, so if
// the reinterpret_cast outer stride is accidentally built from box extents
// instead of parent extents it becomes 4*2=8 instead of the correct 4*3=12.
//
// The hasParentShape/parentShapeBase logic must use acRank (= the array_coor
// index count = box's visible rank = 2) as the base into shapeVec, NOT the
// parent rank (= 3).  With rank reduction acRank < rank, so the old
// `shapeVec.size() >= 2*rank` guard would be false and parentShapeBase would
// fall back to 0, mixing box and parent extents in the stride product.

func.func @rank_reduced_nonprefix(
    %arg0: !fir.ref<!fir.array<4x3x2xi32>>, %i: index, %j: index) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %undef = fir.undefined index
  %shape = fir.shape %c4, %c3, %c2 : (index, index, index) -> !fir.shape<3>
  // Embox slice: a(:, 2, :) -- collapse parent dim 1 at lb=2.
  %eslice = fir.slice %c1, %c4, %c1, %c2, %undef, %undef, %c1, %c2, %c1
      : (index, index, index, index, index, index, index, index, index)
      -> !fir.slice<3>
  %box = fir.embox %arg0(%shape) [%eslice]
      : (!fir.ref<!fir.array<4x3x2xi32>>, !fir.shape<3>, !fir.slice<3>)
      -> !fir.box<!fir.array<4x2xi32>>
  %ashape = fir.shape %c4, %c2 : (index, index) -> !fir.shape<2>
  %ai = arith.index_cast %i : index to i64
  %islice = fir.slice %c1, %c4, %c1, %j, %undef, %undef
      : (index, index, index, index, index, index) -> !fir.slice<2>
  %addr = fir.array_coor %box(%ashape) [%islice] %ai, %j
      : (!fir.box<!fir.array<4x2xi32>>, !fir.shape<2>, !fir.slice<2>, i64, index)
      -> !fir.ref<i32>
  %v = fir.load %addr : !fir.ref<i32>
  return %v : i32
}

// CHECK-LABEL: func.func @rank_reduced_nonprefix(
// CHECK-SAME:      %[[ARG0:.+]]: !fir.ref<!fir.array<4x3x2xi32>>,
// CHECK-SAME:      %[[I:.+]]: index, %[[J:.+]]: index) -> i32
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index

// Row-major memref view of the parent (col-major !fir.array<4x3x2>).
// CHECK:       %[[MEMREF:.+]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<4x3x2xi32>>) -> memref<2x3x4xi32>

// The outer stride must use the parent's dim-1 extent (%c3), not the box's
// dim-1 extent (%c2): correct = %c3*%c4 = 12, wrong = %c2*%c4 = 8.
// CHECK:       %[[OUTER_STRIDE:.+]] = arith.muli %[[C3]], %[[C4]] : index

// Middle stride is parent's dim-0 extent (%c4); inner is 1.
// CHECK:       memref.reinterpret_cast %[[MEMREF]] to offset: [%{{.+}}], sizes: [%{{.+}}, %{{.+}}, %{{.+}}], strides: [%[[OUTER_STRIDE]], %[[C4]], %{{.+}}] : memref<2x3x4xi32> to memref<?x?x?xi32, strided<[?, ?, ?], offset: ?>>
