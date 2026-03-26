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

