// Verify fir.array_coor with explicit shape_shift and slice correctly
// computes 0-based memref indices when lower bounds are non-default.
// This pattern arises after inlining canonicalizes fir.embox+fir.array_coor
// into a single fir.array_coor with explicit shape and slice operands,
// where the indices become Fortran indices rather than 1-based section indices.
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
