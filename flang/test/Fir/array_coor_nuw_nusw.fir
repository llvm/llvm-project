// RUN: fir-opt --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

// Verify when nuw/nusw are added for array coordinate lowering where appropriate.

func.func @ext_array_coor_nuw_eligible(%arg0: !fir.ref<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : i64
  %0 = fircg.ext_array_coor %arg0(%c0) <%c0> : (!fir.ref<!fir.array<?xi32>>, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_nuw_eligible(
// CHECK:         %[[VAL_0:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[VAL_1:.*]] = llvm.mul %[[VAL_0]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[VAL_2:.*]] = llvm.mul %[[VAL_1]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[VAL_3:.*]] = llvm.add %[[VAL_2]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %{{.*}} = llvm.getelementptr nusw|nuw %{{.*}}[%[[VAL_3]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32

// -----

func.func @ext_array_coor_shifted_nuw_ineligble(%arg0: !fir.ref<!fir.array<?xi32>>) {
  %extent = arith.constant 10 : i64
  %shift  = arith.constant 1 : i64
  %idx    = arith.constant 3 : i64
  %0 = fircg.ext_array_coor %arg0(%extent) origin %shift <%idx>
      : (!fir.ref<!fir.array<?xi32>>, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_shifted_nuw_ineligble(
// CHECK:         %[[VAL_0:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw> : i64
// CHECK:         %[[VAL_1:.*]] = llvm.mul %[[VAL_0]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[VAL_2:.*]] = llvm.mul %[[VAL_1]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[VAL_3:.*]] = llvm.add %[[VAL_2]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %{{.*}} = llvm.getelementptr nusw|nuw %{{.*}}[%[[VAL_3]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32

// -----

// Reverse section: known negative step — nsw only (result may be < 0).

func.func @ext_array_coor_reverse_slice(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %extent = arith.constant 10 : i64
  %lb = arith.constant 10 : i64
  %ub = arith.constant 1 : i64
  %step = arith.constant -1 : i64
  %idx = arith.constant 1 : i64
  %0 = fircg.ext_array_coor %arg0(%extent)[%lb, %ub, %step]<%idx>
      : (!fir.ref<!fir.array<10xi32>>, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_reverse_slice(
// CHECK:         %[[IDX:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[ADJ:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ]] overflow<nsw> : i64
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[DIFF1]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[STRIDE]], %{{.*}} overflow<nsw> : i64
// CHECK:         %{{.*}} = llvm.getelementptr nusw|nuw %{{.*}}[%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32

// -----

// Unknown step: may be negative, so drop nuw; nsw still holds for in-bounds indexing.

func.func @ext_array_coor_unknown_step(%arg0: !fir.ref<!fir.array<?xi32>>, %step: i64, %idx: i64) {
  %extent = arith.constant 10 : i64
  %lb = arith.constant 1 : i64
  %ub = arith.constant 10 : i64
  %0 = fircg.ext_array_coor %arg0(%extent)[%lb, %ub, %step]<%idx>
      : (!fir.ref<!fir.array<?xi32>>, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_unknown_step(
// CHECK:         %[[IDX:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[ADJ:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ]] overflow<nsw> : i64
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[DIFF1]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[STRIDE]], %{{.*}} overflow<nsw> : i64

// -----

// Known positive step > 1: nsw|nuw.

func.func @ext_array_coor_step_pos2(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %extent = arith.constant 10 : i64
  %lb = arith.constant 1 : i64
  %ub = arith.constant 9 : i64
  %step = arith.constant 2 : i64
  %idx = arith.constant 1 : i64
  %0 = fircg.ext_array_coor %arg0(%extent)[%lb, %ub, %step]<%idx>
      : (!fir.ref<!fir.array<10xi32>>, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_step_pos2(
// CHECK:         %[[IDX:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[ADJ:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ]] overflow<nsw, nuw> : i64
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[DIFF1]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[STRIDE]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %{{.*}} = llvm.getelementptr nusw|nuw %{{.*}}[%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32

// -----

// Known negative step < -1: nsw only.

func.func @ext_array_coor_step_neg2(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %extent = arith.constant 10 : i64
  %lb = arith.constant 9 : i64
  %ub = arith.constant 1 : i64
  %step = arith.constant -2 : i64
  %idx = arith.constant 1 : i64
  %0 = fircg.ext_array_coor %arg0(%extent)[%lb, %ub, %step]<%idx>
      : (!fir.ref<!fir.array<10xi32>>, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_step_neg2(
// CHECK:         %[[IDX:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[ADJ:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ]] overflow<nsw> : i64
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[DIFF1]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[STRIDE]], %{{.*}} overflow<nsw> : i64
// CHECK:         %{{.*}} = llvm.getelementptr nusw|nuw %{{.*}}[%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32

// -----

// Per-dimension flags: dim0 step +2 → nsw|nuw, dim1 step -1 → nsw.

func.func @ext_array_coor_mixed_steps(%arg0: !fir.ref<!fir.array<10x10xi32>>) {
  %e = arith.constant 10 : i64
  %lb0 = arith.constant 1 : i64
  %ub0 = arith.constant 9 : i64
  %s0 = arith.constant 2 : i64
  %lb1 = arith.constant 10 : i64
  %ub1 = arith.constant 1 : i64
  %s1 = arith.constant -1 : i64
  %i = arith.constant 1 : i64
  %0 = fircg.ext_array_coor %arg0(%e, %e)[%lb0, %ub0, %s0, %lb1, %ub1, %s1]<%i, %i>
      : (!fir.ref<!fir.array<10x10xi32>>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_mixed_steps(
// CHECK:         %[[IDX0:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX0]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[ADJ0:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ0]] overflow<nsw, nuw> : i64
// CHECK:         %[[STRIDE0:.*]] = llvm.mul %[[DIFF1]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[OFFSET0:.*]] = llvm.add %[[STRIDE0]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[EXT:.*]] = llvm.mul %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[IDX1:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF2:.*]] = llvm.mul %[[IDX1]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[ADJ1:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF3:.*]] = llvm.add %[[DIFF2]], %[[ADJ1]] overflow<nsw> : i64
// CHECK:         %[[STRIDE1:.*]] = llvm.mul %[[DIFF3]], %[[EXT]] overflow<nsw> : i64
// CHECK:         %[[OFFSET1:.*]] = llvm.add %[[STRIDE1]], %[[OFFSET0]] overflow<nsw> : i64
// CHECK:         %{{.*}} = llvm.getelementptr nusw|nuw %{{.*}}[%[[OFFSET1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32

// -----

// Boxed reverse slice: indexFlags nsw on idx*step/diff+adj; byte stride mul/add always nsw.

func.func @ext_array_coor_boxed_reverse(%arg0: !fir.box<!fir.array<?xi32>>) {
  %extent = arith.constant 10 : i64
  %lb = arith.constant 10 : i64
  %ub = arith.constant 1 : i64
  %step = arith.constant -1 : i64
  %idx = arith.constant 1 : i64
  %0 = fircg.ext_array_coor %arg0(%extent)[%lb, %ub, %step]<%idx>
      : (!fir.box<!fir.array<?xi32>>, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor_boxed_reverse(
// CHECK:         %[[IDX:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[ADJ:.*]] = llvm.sub %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ]] overflow<nsw> : i64
// CHECK:         %[[STRIDE:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i64
// CHECK:         %[[SC:.*]] = llvm.mul %[[DIFF1]], %[[STRIDE]] overflow<nsw> : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[SC]], %{{.*}} overflow<nsw> : i64
// CHECK:         %[[BASE:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %{{.*}} = llvm.getelementptr nusw %[[BASE]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8

