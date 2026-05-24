// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// Tests for fir.slice with a path component (projected component slice).
// A projected slice changes the element type of the boxed view, e.g.
// z%re projects complex<f32> -> f32.  The pass bypasses the box descriptor
// and reinterprets the underlying complex array as memref<...x2xf32>, then
// appends the component index (0=re, 1=im) as the final memref index.
//
// Derived from:
//   complex, target :: z(4) = 0.
//   real, pointer  :: r(:)
//   r => z%re
//   r = r + z(4:1:-1)%re

// ----------------------------------------------------------------------------
// Forward projected slice load: z(1:4:1)%re
// The fir.convert appears inside the loop body (insertion point tracks the
// array_coor inside the loop).  elemIdx = (i - 1) * step + (lb - 1) = i - 1
// for step=1, lb=1.  Indices are reversed (col-major → row-major) but for 1D
// that is a no-op.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_fwd
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         [[MEMREF:%.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xcomplex<f32>>
// CHECK:         [[IDX:%.*]] = arith.addi
// CHECK:         [[COMP:%.*]] = fir.convert [[MEMREF]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         arith.constant 0
// CHECK:         memref.load [[COMP]][[[IDX]], {{%.*}}] : memref<4x2xf32>
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
// Backward projected slice load: z(4:1:-1)%re
// step = -1, lb = 4  →  elemIdx = (i - 1) * (-1) + (4 - 1) = 3 - (i-1)
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_bwd
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         [[MEMREF:%.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xcomplex<f32>>
// CHECK:         [[IDX:%.*]] = arith.addi
// CHECK:         [[COMP:%.*]] = fir.convert [[MEMREF]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         arith.constant 0
// CHECK:         memref.load [[COMP]][[[IDX]], {{%.*}}] : memref<4x2xf32>
func.func @projected_slice_bwd(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cm1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c4, %c1, %cm1 path %c0 : (index, index, index, index) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    %val = fir.load %coor : !fir.ref<f32>
  }
  return
}

// ----------------------------------------------------------------------------
// Imaginary component store: z(1:4:1)%im = val
// Direct scalar store — no read-modify-write, no complex.create.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_store_im
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         [[MEMREF:%.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xcomplex<f32>>
// CHECK:         [[IDX:%.*]] = arith.addi
// CHECK:         [[COMP:%.*]] = fir.convert [[MEMREF]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         arith.constant 1
// CHECK:         memref.store %arg1, [[COMP]][[[IDX]], {{%.*}}] : memref<4x2xf32>
func.func @projected_slice_store_im(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>,
                                    %arg1: f32) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_im = arith.constant 1 : index  // imaginary component index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %c1_im : (index, index, index, index) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    fir.store %arg1 to %coor : !fir.ref<f32>
  }
  return
}

// ----------------------------------------------------------------------------
// 2-D boxed projected slice load: z(1:2:1, 1:3:1)%re
// Storage: !fir.array<2x3xcomplex<f32>>
//
// convertMemrefType reverses Fortran column-major extents to MLIR row-major:
//   !fir.ref<!fir.array<2x3xcomplex<f32>>> → memref<3x2xcomplex<f32>>
// Reinterpret adds the component dimension:
//   memref<3x2xcomplex<f32>> → memref<3x2x2xf32>
//
// Per-dimension element index (0-based, column-major):
//   elemIdx_i = (i-1)*1 + (1-1) = i-1   (Fortran dim 1, size 2)
//   elemIdx_j = (j-1)*1 + (1-1) = j-1   (Fortran dim 2, size 3)
//
// After reversing for MLIR row-major access:
//   memref.load [elemIdx_j, elemIdx_i, 0]
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_2d
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         fir.do_loop [[J:%.*]] =
// CHECK:           [[MEMREF:%.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<2x3xcomplex<f32>>>) -> memref<3x2xcomplex<f32>>
// CHECK:           [[IDX_I:%.*]] = arith.addi
// CHECK:           [[IDX_J:%.*]] = arith.addi
// CHECK:           [[COMP:%.*]] = fir.convert [[MEMREF]] : (memref<3x2xcomplex<f32>>) -> memref<3x2x2xf32>
// CHECK:           arith.constant 0
// CHECK:           memref.load [[COMP]][[[IDX_J]], [[IDX_I]], {{%.*}}] : memref<3x2x2xf32>
func.func @projected_slice_2d(%arg0: !fir.ref<!fir.array<2x3xcomplex<f32>>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c2, %c3 : (index, index) -> !fir.shape<2>
  %slice = fir.slice %c1, %c2, %c1, %c1, %c3, %c1 path %c0 : (index, index, index, index, index, index, index) -> !fir.slice<2>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<2x3xcomplex<f32>>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x3xf32>>
  fir.do_loop %i = %c1 to %c2 step %c1 unordered {
    fir.do_loop %j = %c1 to %c3 step %c1 unordered {
      %coor = fir.array_coor %embox %i, %j : (!fir.box<!fir.array<2x3xf32>>, index, index) -> !fir.ref<f32>
      %val = fir.load %coor : !fir.ref<f32>
    }
  }
  return
}

// ----------------------------------------------------------------------------
// Derived-type component projection: a%x where a : TYPE{x:f64, y:complex<f64>}
//
// This is NOT a complex projection — the storage element is the derived type T,
// not complex<T>.  FIRToMemRef cannot safely handle this; downstream
// FIR-to-LLVM lowering handles it correctly via the descriptor.
//
// CHECK-LABEL: func.func @derived_component_not_projected
// The fir.array_coor must survive (not be erased).
// CHECK:       fir.array_coor
// The store must remain as fir.store, not memref.store.
// CHECK:       fir.store
// CHECK-NOT:   memref.store
// ----------------------------------------------------------------------------
func.func @derived_component_not_projected(
    %arg0: !fir.ref<!fir.array<4x!fir.type<T{x:f64,y:complex<f64>}>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 9.9e+01 : f64
  %field = fir.field_index x, !fir.type<T{x:f64,y:complex<f64>}>
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %field : (index, index, index, !fir.field) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4x!fir.type<T{x:f64,y:complex<f64>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf64>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf64>>, index) -> !fir.ref<f64>
    fir.store %cst to %coor : !fir.ref<f64>
  }
  return
}
