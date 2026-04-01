// Verify fir.array_coor with explicit shape_shift and slice correctly
// computes 0-based memref indices.
//
// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// A(0:9) = 1  (lower bound 0)
// The Fortran index 0 must map to memref index 0, not -1.
// CHECK-LABEL: func.func @array_coor_slice_shift_1d
// CHECK:       memref.store
// CHECK-NOT:   fir.array_coor
func.func @array_coor_slice_shift_1d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFEa"}
  %1 = fir.shape_shift %c0, %c10 : (index, index) -> !fir.shapeshift<1>
  %2 = fir.declare %0(%1) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<10xi32>>
  %3 = fir.slice %c0, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  // Index %c0 is Fortran index 0 (= lower bound). Must produce memref index 0.
  %4 = fir.array_coor %2(%1) [%3] %c0 : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
  fir.store %c1_i32 to %4 : !fir.ref<i32>
  return
}

// A(0:9, -1:8) = 1  (lower bounds 0 and -1)
// Fortran indices (0, -1) must map to memref indices (0, 0).
// CHECK-LABEL: func.func @array_coor_slice_shift_2d
// CHECK:       memref.store {{%.+}}, {{%.+}}[{{%.+}}, {{%.+}}] : memref<10x10xi32>
// CHECK-NOT:   fir.array_coor
func.func @array_coor_slice_shift_2d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c_neg1 = arith.constant -1 : index
  %c8 = arith.constant 8 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca !fir.array<10x10xi32> {bindc_name = "a", uniq_name = "_QFEa"}
  %1 = fir.shape_shift %c0, %c10, %c_neg1, %c10 : (index, index, index, index) -> !fir.shapeshift<2>
  %2 = fir.declare %0(%1) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10x10xi32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<10x10xi32>>
  %3 = fir.slice %c0, %c10, %c1, %c_neg1, %c8, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  // Fortran indices (0, -1) = lower bounds => memref indices must be (0, 0).
  %4 = fir.array_coor %2(%1) [%3] %c0, %c_neg1 : (!fir.ref<!fir.array<10x10xi32>>, !fir.shapeshift<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
  fir.store %c1_i32 to %4 : !fir.ref<i32>
  return
}

// A(1:6, 1:9) with section A(:, 2:4).  (default lb=1, slice starts at 2)
// Index (1, 1) = lower bounds => memref indices must be (1, 0).
// The slice offset for dim 2 (sliceLb=2, shift=1 => offset=1) must be
// preserved, not cancelled out.
// CHECK-LABEL: func.func @array_coor_slice_shift_section
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// The dim 2 offset = sliceLb - shift = 2 - 1 = 1:
// CHECK:       arith.subi %[[C2]], %[[C1]] : index
// CHECK:       memref.store
// CHECK-NOT:   fir.array_coor
func.func @array_coor_slice_shift_section() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %c9 = arith.constant 9 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca !fir.array<6x9xi32> {bindc_name = "a", uniq_name = "_QFEa"}
  %1 = fir.shape_shift %c1, %c6, %c1, %c9 : (index, index, index, index) -> !fir.shapeshift<2>
  %2 = fir.declare %0(%1) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<6x9xi32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<6x9xi32>>
  // Slice: full range in dim 1, section 2:4 in dim 2.
  %3 = fir.slice %c1, %c6, %c1, %c2, %c4, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  // Index (1, 1) in shape_shift space. Dim 2 slice starts at 2,
  // so memref index for dim 2 must be (1-1)+(2-1) = 1, not 0.
  %4 = fir.array_coor %2(%1) [%3] %c1, %c1 : (!fir.ref<!fir.array<6x9xi32>>, !fir.shapeshift<2>, !fir.slice<2>, index, index) -> !fir.ref<i32>
  fir.store %c1_i32 to %4 : !fir.ref<i32>
  return
}
