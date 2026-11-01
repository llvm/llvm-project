// RUN: fir-opt %s --fir-to-memref | FileCheck %s

// Verify that when BOTH the embox slice and the inner array_coor slice have
// non-unit lower bounds and non-unit strides, the correct physical index is
// produced.  All four values (embox lb, embox stride, ac lb, ac stride) are
// intentionally distinct to catch any mix-up between them.
//
// Embox slice: a(3:20:2)  -- lb=3, stride=2, 9 elements: 3,5,7,...,19
// AC    slice: 4:9:5       -- lb=4, stride=5, 2 positions in box: 4,9
// Index %i (1-based into ac slice):
//   getMemrefIndices:         inner = (i-1)*5 + (4-1) = (i-1)*5 + 3
//   foldSliceLbIntoIndices:   physical = inner*2 + (3-1) = inner*2 + 2
//
// For i=1: inner=3, physical=8  (0-based) -> a(9)  = a(3 + 3*2). ok
// For i=2: inner=8, physical=18 (0-based) -> a(19) = a(3 + 8*2). ok

func.func @emboxed_slice_nonstride_nonlb(
    %arg0: !fir.ref<!fir.array<20xi32>>, %i: index) -> i32 {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c9 = arith.constant 9 : index
  %c20 = arith.constant 20 : index
  %shape = fir.shape %c20 : (index) -> !fir.shape<1>
  // Embox slice: a(3:20:2) -- lb=3, stride=2, 9 elements.
  %eslice = fir.slice %c3, %c20, %c2 : (index, index, index) -> !fir.slice<1>
  %box = fir.embox %arg0(%shape) [%eslice]
      : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>)
      -> !fir.box<!fir.array<9xi32>>
  // Inner array_coor slice: 4:9:5 -- lb=4, stride=5.
  %ashape = fir.shape %c9 : (index) -> !fir.shape<1>
  %islice = fir.slice %c4, %c9, %c5 : (index, index, index) -> !fir.slice<1>
  %addr = fir.array_coor %box(%ashape) [%islice] %i
      : (!fir.box<!fir.array<9xi32>>, !fir.shape<1>, !fir.slice<1>, index)
      -> !fir.ref<i32>
  %v = fir.load %addr : !fir.ref<i32>
  return %v : i32
}

// CHECK-LABEL: func.func @emboxed_slice_nonstride_nonlb(
// CHECK-SAME:      %[[ARG0:.+]]: !fir.ref<!fir.array<20xi32>>,
// CHECK-SAME:      %[[I:.+]]: index) -> i32
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:   %[[C9:.+]] = arith.constant 9 : index

// Row-major memref view of the parent.
// CHECK:       %[[MEMREF:.+]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<20xi32>>) -> memref<20xi32>

// Anchor: two `arith.constant 0 : index` -- getMemrefIndices' zero placeholder
// and the reinterpret_cast offset. The fold's cOne is emitted right after.
// CHECK:       arith.constant 0 : index
// CHECK:       arith.constant 0 : index
// CHECK-NEXT:  %[[CONE:.+]] = arith.constant 1 : index

// foldSliceLbIntoIndices for embox slice (lb=3, stride=2):
//   delta = 3 - 1 = 2
//   physical = inner * 2 + 2
// CHECK-NEXT:  %[[LB_DELTA:.+]] = arith.subi %[[C3]], %[[CONE]] : index
// CHECK-NEXT:  %[[SCALED:.+]] = arith.muli %{{.+}}, %[[C2]] : index
// CHECK-NEXT:  %[[IDX:.+]] = arith.addi %[[SCALED]], %[[LB_DELTA]] : index

// reinterpret_cast: contiguous parent layout, box extent as size.
// CHECK-NEXT:  memref.reinterpret_cast %[[MEMREF]] to offset: [%{{.+}}], sizes: [%[[C9]]], strides: [%{{.+}}] : memref<20xi32> to memref<?xi32, strided<[?], offset: ?>>

// memref.load uses the fully-adjusted physical index.
// CHECK-NEXT:  memref.load %{{.+}}[%[[IDX]]] : memref<?xi32, strided<[?], offset: ?>>
