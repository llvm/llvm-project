// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// Tests for fir.slice with a path component (projected component slice).
// A projected slice changes the logical element type of the boxed view, e.g.
// z%re maps complex<f32> -> f32 in the box type.
//
// FIRToMemRef only lowers array_coor through this path when storage is
// complex<T> and the slice path is constant 0 (%re) or 1 (%im).  It then
// fir.convert's storage to memref<...x2xT>, appends that component index,
// and memref.reinterpret_cast's a strided memref<...xT> using fir.box_dims
// byte strides divided by sizeof(T) (not sizeof(complex<T>) from box_elesize).
// Derived-type component projections (e.g. a%x) are left for FIR codegen.
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
// CHECK:         [[COMP:%[0-9]+]] = fir.convert [[MEMREF]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         %[[FWD_C_RE:.*]] = arith.constant 0 : index
// CHECK:         %[[FWD_C_SZF32:.*]] = arith.constant 4 : index
// CHECK:         %[[FWD_C_DIM0:.*]] = arith.constant 0 : index
// CHECK:         [[BD:%[0-9]+]]:3 = fir.box_dims %2, %[[FWD_C_DIM0]] : (!fir.box<!fir.array<4xf32>>, index) -> (index, index, index)
// CHECK:         [[STRIDE:%[0-9]+]] = arith.divsi [[BD]]#2, %[[FWD_C_SZF32]] : index
// Reinterpret applies the embox descriptor layout onto the scalar view:
//   sizes[0]   = box extent (section length in f32 slots)
//   sizes[1]   = 2 for the (re, im) pair exposed by memref<4x2xf32>
//   strides[0] = box_dims byte_stride / sizeof(f32) (not box_elesize)
//   strides[1] = 1 between adjacent real/imag scalars
// Without this, memref.load would use dense strides from fir.convert only.
// CHECK:         %[[FWD_C_PAIR:.*]] = arith.constant 2 : index
// CHECK:         %[[FWD_C_COMP_STRIDE:.*]] = arith.constant 1 : index
// CHECK:         %[[FWD_C_OFF:.*]] = arith.constant 0 : index
// CHECK:         [[VIEW:%.*]] = memref.reinterpret_cast [[COMP]] to offset: [%[[FWD_C_OFF]]], sizes: [[[BD]]#1, %[[FWD_C_PAIR]]], strides: [[[STRIDE]], %[[FWD_C_COMP_STRIDE]]] : memref<4x2xf32> to memref<?x?xf32, strided<
// CHECK:         [[LOAD:%[0-9]+]] = memref.load [[VIEW]][[[IDX]], %[[FWD_C_RE]]] : memref<?x?xf32, strided<
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
// CHECK:         [[COMP:%[0-9]+]] = fir.convert [[MEMREF]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         %[[BWD_C_RE:.*]] = arith.constant 0 : index
// CHECK:         %[[BWD_C_SZF32:.*]] = arith.constant 4 : index
// CHECK:         %[[BWD_C_DIM0:.*]] = arith.constant 0 : index
// CHECK:         [[BD:%[0-9]+]]:3 = fir.box_dims %2, %[[BWD_C_DIM0]] : (!fir.box<!fir.array<4xf32>>, index) -> (index, index, index)
// CHECK:         [[STRIDE:%[0-9]+]] = arith.divsi [[BD]]#2, %[[BWD_C_SZF32]] : index
// Same reinterpret as forward; slice triple only changes [[IDX]], not strides.
// CHECK:         %[[BWD_C_PAIR:.*]] = arith.constant 2 : index
// CHECK:         %[[BWD_C_COMP_STRIDE:.*]] = arith.constant 1 : index
// CHECK:         %[[BWD_C_OFF:.*]] = arith.constant 0 : index
// CHECK:         [[VIEW:%.*]] = memref.reinterpret_cast [[COMP]] to offset: [%[[BWD_C_OFF]]], sizes: [[[BD]]#1, %[[BWD_C_PAIR]]], strides: [[[STRIDE]], %[[BWD_C_COMP_STRIDE]]] : memref<4x2xf32> to memref<?x?xf32, strided<
// CHECK:         [[LOAD:%[0-9]+]] = memref.load [[VIEW]][[[IDX]], %[[BWD_C_RE]]] : memref<?x?xf32, strided<
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
// CHECK:         [[COMP:%[0-9]+]] = fir.convert [[MEMREF]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         %[[IM_C_IM:.*]] = arith.constant 1 : index
// CHECK:         %[[IM_C_SZF32:.*]] = arith.constant 4 : index
// CHECK:         %[[IM_C_DIM0:.*]] = arith.constant 0 : index
// CHECK:         [[BD:%[0-9]+]]:3 = fir.box_dims %2, %[[IM_C_DIM0]] : (!fir.box<!fir.array<4xf32>>, index) -> (index, index, index)
// CHECK:         [[STRIDE:%[0-9]+]] = arith.divsi [[BD]]#2, %[[IM_C_SZF32]] : index
// Same layout as %re; store uses component index 1 for imaginary.
// CHECK:         %[[IM_C_PAIR:.*]] = arith.constant 2 : index
// CHECK:         %[[IM_C_COMP_STRIDE:.*]] = arith.constant 1 : index
// CHECK:         %[[IM_C_OFF:.*]] = arith.constant 0 : index
// CHECK:         [[VIEW:%.*]] = memref.reinterpret_cast [[COMP]] to offset: [%[[IM_C_OFF]]], sizes: [[[BD]]#1, %[[IM_C_PAIR]]], strides: [[[STRIDE]], %[[IM_C_COMP_STRIDE]]] : memref<4x2xf32> to memref<?x?xf32, strided<
// CHECK:         memref.store %arg1, [[VIEW]][[[IDX]], %[[IM_C_IM]]] : memref<?x?xf32, strided<
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
// CHECK:           [[COMP:%[0-9]+]] = fir.convert [[MEMREF]] : (memref<3x2xcomplex<f32>>) -> memref<3x2x2xf32>
// CHECK:           %[[D2_C_RE:.*]] = arith.constant 0 : index
// CHECK:           %[[D2_C_SZF32:.*]] = arith.constant 4 : index
// CHECK:           %[[D2_C_DIM1:.*]] = arith.constant 1 : index
// CHECK:           [[BD0:%[0-9]+]]:3 = fir.box_dims %2, %[[D2_C_DIM1]] : (!fir.box<!fir.array<2x3xf32>>, index) -> (index, index, index)
// CHECK:           [[STR0:%[0-9]+]] = arith.divsi [[BD0]]#2, %[[D2_C_SZF32]] : index
// CHECK:           %[[D2_C_DIM0:.*]] = arith.constant 0 : index
// CHECK:           [[BD1:%[0-9]+]]:3 = fir.box_dims %2, %[[D2_C_DIM0]] : (!fir.box<!fir.array<2x3xf32>>, index) -> (index, index, index)
// CHECK:           [[STR1:%[0-9]+]] = arith.divsi [[BD1]]#2, %[[D2_C_SZF32]] : index
// 2-D embox: two box_dims strides (both / sizeof(f32)), plus pair dim (2, 1).
// Row-major memref indices are [j, i, 0] after Fortran dim reversal.
// CHECK:           %[[D2_C_PAIR:.*]] = arith.constant 2 : index
// CHECK:           %[[D2_C_COMP_STRIDE:.*]] = arith.constant 1 : index
// CHECK:           %[[D2_C_OFF:.*]] = arith.constant 0 : index
// CHECK:           [[VIEW:%.*]] = memref.reinterpret_cast [[COMP]] to offset: [%[[D2_C_OFF]]], sizes: [[[BD0]]#1, [[BD1]]#1, %[[D2_C_PAIR]]], strides: [[[STR0]], [[STR1]], %[[D2_C_COMP_STRIDE]]] : memref<3x2x2xf32> to memref<?x?x?xf32, strided<
// CHECK:           [[LOAD:%[0-9]+]] = memref.load [[VIEW]][[[IDX_J]], [[IDX_I]], %[[D2_C_RE]]] : memref<?x?x?xf32, strided<
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
// Projected slice on a complex descriptor (no embox): array_coor %box[%slice]
// path %c0 for %re.  Strides must be scaled by sizeof(f32), not
// sizeof(complex<f32>) from fir.box_elesize on the complex box.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_complex_box
// CHECK:         fir.do_loop
// CHECK:           fir.do_loop
// CHECK:             [[CB_BOX:%[0-9]+]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?x?xcomplex<f32>>>) -> !fir.ref<!fir.array<?x?xcomplex<f32>>>
// CHECK:             [[CB_REF:%[0-9]+]] = fir.convert [[CB_BOX]] : (!fir.ref<!fir.array<?x?xcomplex<f32>>>) -> memref<?x?xcomplex<f32>>
// CHECK:             [[CB_I:%[0-9]+]] = arith.addi
// CHECK:             [[CB_J:%[0-9]+]] = arith.addi
// CHECK:             [[CB_COMP:%[0-9]+]] = fir.convert [[CB_REF]] : (memref<?x?xcomplex<f32>>) -> memref<?x?x2xf32>
// CHECK:             %[[CB_C_RE:.*]] = arith.constant 0 : index
// CHECK:             %[[CB_C_SZF32:.*]] = arith.constant 4 : index
// CHECK:             %[[CB_C_DIM1:.*]] = arith.constant 1 : index
// CHECK:             [[CB_BD0:%[0-9]+]]:3 = fir.box_dims %arg0, %[[CB_C_DIM1]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// CHECK:             [[CB_STR0:%[0-9]+]] = arith.divsi [[CB_BD0]]#2, %[[CB_C_SZF32]] : index
// CHECK:             %[[CB_C_DIM0:.*]] = arith.constant 0 : index
// CHECK:             [[CB_BD1:%[0-9]+]]:3 = fir.box_dims %arg0, %[[CB_C_DIM0]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// CHECK:             [[CB_STR1:%[0-9]+]] = arith.divsi [[CB_BD1]]#2, %[[CB_C_SZF32]] : index
// CHECK-NOT:         fir.box_elesize %arg0
// Complex box (not embox): strides from fir.box_dims on %arg0; divisor sizeof(f32).
// CHECK:             %[[CB_C_PAIR:.*]] = arith.constant 2 : index
// CHECK:             %[[CB_C_COMP_STRIDE:.*]] = arith.constant 1 : index
// CHECK:             %[[CB_C_OFF:.*]] = arith.constant 0 : index
// CHECK:             [[CB_VIEW:%.*]] = memref.reinterpret_cast [[CB_COMP]] to offset: [%[[CB_C_OFF]]], sizes: [[[CB_BD0]]#1, [[CB_BD1]]#1, %[[CB_C_PAIR]]], strides: [[[CB_STR0]], [[CB_STR1]], %[[CB_C_COMP_STRIDE]]] : memref<?x?x2xf32> to memref<?x?x?xf32, strided<
// Row-major memref indices are [j, i, 0] (see projected_slice_2d).
// CHECK:             [[CB_LOAD:%[0-9]+]] = memref.load [[CB_VIEW]][[[CB_J]], [[CB_I]], %[[CB_C_RE]]] : memref<?x?x?xf32, strided<
func.func @projected_slice_complex_box(%arg0: !fir.box<!fir.array<?x?xcomplex<f32>>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
  %dim1:3 = fir.box_dims %arg0, %c1 : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
  %shape = fir.shape %dim0#1, %dim1#1 : (index, index) -> !fir.shape<2>
  %slice = fir.slice %c1, %dim0#1, %c1, %c1, %dim1#1, %c1 path %c0 : (index, index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %i = %c1 to %dim0#1 step %c1 unordered {
    fir.do_loop %j = %c1 to %dim1#1 step %c1 unordered {
      %coor = fir.array_coor %arg0 [%slice] %i, %j : (!fir.box<!fir.array<?x?xcomplex<f32>>>, !fir.slice<2>, index, index) -> !fir.ref<f32>
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
