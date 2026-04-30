// RUN: fir-opt --canonicalize --split-input-file %s | FileCheck %s

// Verify that fir.rebox is NOT folded into fir.array_coor when the rebox is
// defined outside a cuf.kernel and the array_coor is inside that kernel.
// CUF lowering converts such a rebox into a managed-memory descriptor that
// the kernel needs to receive as its argument; folding the rebox away would
// leave the kernel capturing the host-side descriptor directly, causing
// illegal device dereferences at runtime.
// CHECK-LABEL:   func.func @test_cuf_kernel_preserves_captured_rebox(
// CHECK-SAME:        %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>
// CHECK-SAME:        %[[VAL:.*]]: f32
// CHECK:           %[[REBOX:.*]] = fir.rebox %[[BOX]]
// CHECK:           cuf.kernel
// CHECK:             %[[COOR:.*]] = fir.array_coor %[[REBOX]]
// CHECK:             fir.store %[[VAL]] to %[[COOR]]
func.func @test_cuf_kernel_preserves_captured_rebox(%arg0: !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, %arg1: f32) {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg2 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %1 = fir.array_coor %0 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    fir.store %arg1 to %1 : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// When both the rebox and the array_coor are inside the same cuf.kernel, the
// capture boundary is not crossed and folding must still happen.
// CHECK-LABEL:   func.func @test_cuf_kernel_folds_local_rebox(
// CHECK-SAME:        %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>
// CHECK-SAME:        %[[VAL:.*]]: f32
// CHECK-NOT:       fir.rebox
// CHECK:           cuf.kernel
// CHECK-NOT:         fir.rebox
// CHECK:             %[[COOR:.*]] = fir.array_coor %[[BOX]]
// CHECK:             fir.store %[[VAL]] to %[[COOR]]
func.func @test_cuf_kernel_folds_local_rebox(%arg0: !fir.box<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>}, %arg1: f32) {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%arg2 : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
    %1 = fir.array_coor %0 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    fir.store %arg1 to %1 : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  return
}

// -----

// Without cuf.kernel, rebox SHOULD be folded normally (regression: the guard
// must not affect ordinary host code).
// CHECK-LABEL:   func.func @test_no_cuf_folds_rebox(
// CHECK-SAME:        %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>) -> !fir.ref<f32> {
// CHECK-NOT:       fir.rebox
// CHECK:           %[[COOR:.*]] = fir.array_coor %[[BOX]]
// CHECK:           return %[[COOR]] : !fir.ref<f32>
func.func @test_no_cuf_folds_rebox(%arg0: !fir.box<!fir.array<?xf32>>) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %1 = fir.array_coor %0 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  return %1 : !fir.ref<f32>
}
