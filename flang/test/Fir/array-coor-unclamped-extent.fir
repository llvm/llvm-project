// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// Verify that array coordinate strides of contiguous arrays use the extent
// before it is clamped to zero: an element can only be addressed when the array
// is not empty, so max(x, 0) == x for any valid access.

func.func @maxsi_extent(%arg0: !fir.ref<!fir.array<?x?xf64>>, %x: i64, %y: i64, %i: i64, %j: i64) {
  %c0 = arith.constant 0 : i64
  %e0 = arith.maxsi %x, %c0 : i64
  %e1 = arith.maxsi %y, %c0 : i64
  %0 = fircg.ext_array_coor %arg0(%e0, %e1) <%i, %j> : (!fir.ref<!fir.array<?x?xf64>>, i64, i64, i64, i64) -> !fir.ref<f64>
  return
}

// CHECK-LABEL: llvm.func @maxsi_extent(
// CHECK-SAME:      %[[BASE:[a-z0-9]+]]: !llvm.ptr, %[[X:[a-z0-9]+]]: i64, %[[Y:[a-z0-9]+]]: i64, %[[I:[a-z0-9]+]]: i64, %[[J:[a-z0-9]+]]: i64
// CHECK:         %[[E0:.*]] = llvm.intr.smax(%[[X]], %{{.*}}) : (i64, i64) -> i64
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT:     %[[E0]]{{[^0-9]}}
// CHECK:         %[[STRIDE1:.*]] = llvm.mul %[[ONE]], %[[X]] overflow<nsw, nuw> : i64
// CHECK-NOT:     %[[E0]]{{[^0-9]}}
// CHECK:         %[[SC1:.*]] = llvm.mul %{{.*}}, %[[STRIDE1]] overflow<nsw, nuw> : i64
// CHECK:         %[[OFF:.*]] = llvm.add %[[SC1]], %{{.*}} overflow<nsw, nuw> : i64
// CHECK-NOT:     %[[E0]]{{[^0-9]}}
// CHECK:         llvm.getelementptr nusw|nuw %[[BASE]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64

// -----

// The cmp/select form built by fir::factory::genMaxWithZero.

func.func @select_extent(%arg0: !fir.ref<!fir.array<?x?xf64>>, %x: i64, %y: i64, %i: i64, %j: i64) {
  %c0 = arith.constant 0 : i64
  %gt = arith.cmpi sgt, %x, %c0 : i64
  %e0 = arith.select %gt, %x, %c0 : i64
  %0 = fircg.ext_array_coor %arg0(%e0, %y) <%i, %j> : (!fir.ref<!fir.array<?x?xf64>>, i64, i64, i64, i64) -> !fir.ref<f64>
  return
}

// CHECK-LABEL: llvm.func @select_extent(
// CHECK-SAME:      %[[BASE:[a-z0-9]+]]: !llvm.ptr, %[[X:[a-z0-9]+]]: i64, %[[Y:[a-z0-9]+]]: i64, %[[I:[a-z0-9]+]]: i64, %[[J:[a-z0-9]+]]: i64
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[STRIDE1:.*]] = llvm.mul %[[ONE]], %[[X]] overflow<nsw, nuw> : i64
// CHECK:         %[[SC1:.*]] = llvm.mul %{{.*}}, %[[STRIDE1]] overflow<nsw, nuw> : i64

// -----

// A clamp to a non-zero value is not an emptiness guard and must be kept.

func.func @maxsi_nonzero_extent(%arg0: !fir.ref<!fir.array<?x?xf64>>, %x: i64, %y: i64, %i: i64, %j: i64) {
  %c1 = arith.constant 1 : i64
  %e0 = arith.maxsi %x, %c1 : i64
  %0 = fircg.ext_array_coor %arg0(%e0, %y) <%i, %j> : (!fir.ref<!fir.array<?x?xf64>>, i64, i64, i64, i64) -> !fir.ref<f64>
  return
}

// CHECK-LABEL: llvm.func @maxsi_nonzero_extent(
// CHECK-SAME:      %[[BASE:[a-z0-9]+]]: !llvm.ptr, %[[X:[a-z0-9]+]]: i64
// CHECK:         %[[E0:.*]] = llvm.intr.smax(%[[X]], %{{.*}}) : (i64, i64) -> i64
// CHECK:         %[[STRIDE1:.*]] = llvm.mul %{{.*}}, %[[E0]] overflow<nsw, nuw> : i64
