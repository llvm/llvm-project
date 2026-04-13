// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// Tests for fir.slice with a path component (projected component slice).
// A projected slice changes the element type of the boxed view, e.g.
// z%re projects complex<f32> -> f32.  The layout (strides / base address)
// must come from the projected box descriptor, NOT from reconstructing the
// triplets, because memref.reinterpret_cast requires the same element type
// on both sides and the triplet strides are in storage-element units
// (complex<f32>) while the MemRef strides must be in projected-element units
// (f32).
//
// Derived from:
//   complex, target :: z(4) = 0.
//   real, pointer  :: r(:)
//   r => z%re
//   r = r + z(4:1:-1)%re

// ----------------------------------------------------------------------------
// Forward projected slice: z(1:4:1)%re
// The slice path %c0 projects complex<f32> -> f32 (real part).
// Expected lowering:
//   - fir.box_addr on the projected box (!fir.box<!fir.array<4xf32>>)
//   - fir.convert to memref<4xf32>   (NOT to memref<4xcomplex<f32>>)
//   - index = i - 1  (1-based, no triplet arithmetic)
//   - strides from fir.box_dims / fir.box_elesize on the projected box
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_fwd
// CHECK:       [[C1:%.*]] = arith.constant 1 : index
// CHECK:       [[C4:%.*]] = arith.constant 4 : index
// CHECK:       [[C0:%.*]] = arith.constant 0 : index
// CHECK:       [[SHAPE:%.*]] = fir.shape [[C4]] : (index) -> !fir.shape<1>
// CHECK:       [[SLICE:%.*]] = fir.slice [[C1]], [[C4]], [[C1]] path [[C0]] : (index, index, index, index) -> !fir.slice<1>
// CHECK:       [[EMBOX:%.*]] = fir.embox %arg0([[SHAPE]]) {{\[}}[[SLICE]]{{\]}} : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
// CHECK:       fir.do_loop [[I:%.*]] = [[C1]] to [[C4]] step [[C1]] unordered {
// Projected box_addr gives f32 pointer, not complex<f32>.
// CHECK:         [[BOXADDR:%.*]] = fir.box_addr [[EMBOX]] : (!fir.box<!fir.array<4xf32>>) -> !fir.ref<!fir.array<4xf32>>
// CHECK:         [[CONVERT:%.*]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<4xf32>>) -> memref<4xf32>
// Index: i-1 (1-based). The lowering emits: delta=i-1, scaled=delta*1,
// offset=1-1=0, finalIdx=scaled+offset.  The addi result is what feeds the load.
// CHECK:         [[C1_0:%.*]] = arith.constant 1 : index
// CHECK:         [[DELTA:%.*]] = arith.subi [[I]], [[C1_0]] : index
// CHECK:         [[SCALED:%.*]] = arith.muli [[DELTA]], [[C1_0]] : index
// CHECK:         [[OFFSET:%.*]] = arith.subi [[C1_0]], [[C1_0]] : index
// CHECK:         [[IDX:%.*]] = arith.addi [[SCALED]], [[OFFSET]] : index
// Layout: extent and stride come from the projected box descriptor.
// CHECK:         [[ELE:%.*]] = fir.box_elesize [[EMBOX]] : (!fir.box<!fir.array<4xf32>>) -> index
// CHECK:         [[C0_0:%.*]] = arith.constant 0 : index
// CHECK:         [[DIMS:%.*]]:3 = fir.box_dims [[EMBOX]], [[C0_0]] : (!fir.box<!fir.array<4xf32>>, index) -> (index, index, index)
// CHECK:         [[STRIDE:%.*]] = arith.divsi [[DIMS]]#2, [[ELE]] : index
// CHECK:         [[C0_1:%.*]] = arith.constant 0 : index
// CHECK:         [[VIEW:%.*]] = memref.reinterpret_cast [[CONVERT]] to offset: {{\[}}[[C0_1]]{{\]}}, sizes: {{\[}}[[DIMS]]#1{{\]}}, strides: {{\[}}[[STRIDE]]{{\]}} : memref<4xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:         memref.load [[VIEW]]{{\[}}[[IDX]]{{\]}} : memref<?xf32, strided<[?], offset: ?>>
func.func @projected_slice_fwd(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %c0 : (index, index, index, index) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    %val = fir.load %coor : !fir.ref<f32>
  }
  return
}

// ----------------------------------------------------------------------------
// Reverse projected slice: z(4:1:-1)%re  (the original bug reproducer)
// The descriptor will encode a negative byte-stride for the reversed view.
// The MemRef view must preserve that sign through box_dims / box_elesize.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_rev
// CHECK:       [[C_NEG1:%.*]] = arith.constant -1 : index
// CHECK:       [[C1:%.*]] = arith.constant 1 : index
// CHECK:       [[C4:%.*]] = arith.constant 4 : index
// CHECK:       [[C0:%.*]] = arith.constant 0 : index
// CHECK:       [[SHAPE:%.*]] = fir.shape [[C4]] : (index) -> !fir.shape<1>
// CHECK:       [[SLICE:%.*]] = fir.slice [[C4]], [[C1]], [[C_NEG1]] path [[C0]] : (index, index, index, index) -> !fir.slice<1>
// CHECK:       [[EMBOX:%.*]] = fir.embox %arg0([[SHAPE]]) {{\[}}[[SLICE]]{{\]}} : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
// CHECK:       fir.do_loop [[I:%.*]] = [[C1]] to [[C4]] step [[C1]] unordered {
// CHECK:         [[BOXADDR:%.*]] = fir.box_addr [[EMBOX]] : (!fir.box<!fir.array<4xf32>>) -> !fir.ref<!fir.array<4xf32>>
// CHECK:         [[CONVERT:%.*]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<4xf32>>) -> memref<4xf32>
// CHECK:         [[C1_0:%.*]] = arith.constant 1 : index
// CHECK:         [[DELTA:%.*]] = arith.subi [[I]], [[C1_0]] : index
// CHECK:         [[SCALED:%.*]] = arith.muli [[DELTA]], [[C1_0]] : index
// CHECK:         [[OFFSET:%.*]] = arith.subi [[C1_0]], [[C1_0]] : index
// CHECK:         [[IDX:%.*]] = arith.addi [[SCALED]], [[OFFSET]] : index
// CHECK:         [[ELE:%.*]] = fir.box_elesize [[EMBOX]] : (!fir.box<!fir.array<4xf32>>) -> index
// CHECK:         [[C0_0:%.*]] = arith.constant 0 : index
// CHECK:         [[DIMS:%.*]]:3 = fir.box_dims [[EMBOX]], [[C0_0]] : (!fir.box<!fir.array<4xf32>>, index) -> (index, index, index)
// CHECK:         [[STRIDE:%.*]] = arith.divsi [[DIMS]]#2, [[ELE]] : index
// CHECK:         [[C0_1:%.*]] = arith.constant 0 : index
// CHECK:         [[VIEW:%.*]] = memref.reinterpret_cast [[CONVERT]] to offset: {{\[}}[[C0_1]]{{\]}}, sizes: {{\[}}[[DIMS]]#1{{\]}}, strides: {{\[}}[[STRIDE]]{{\]}} : memref<4xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:         memref.load [[VIEW]]{{\[}}[[IDX]]{{\]}} : memref<?xf32, strided<[?], offset: ?>>
func.func @projected_slice_rev(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c4, %c1, %c-1 path %c0 : (index, index, index, index) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    %val = fir.load %coor : !fir.ref<f32>
  }
  return
}
