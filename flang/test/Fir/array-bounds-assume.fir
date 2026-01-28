// Test that array bounds assumes are generated for fir.array_coor operations.
// This tests that Flang emits llvm.assume intrinsics for array bounds constraints.

// RUN: tco %s | FileCheck %s

// Test 1D array with constant shape.
func.func @array_coor_1d(%addr : !fir.ref<!fir.array<10xi32>>, %idx : index) -> i32 {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %ref = fir.array_coor %addr(%shape) %idx : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %val = fir.load %ref : !fir.ref<i32>
  return %val : i32
}

// CHECK-LABEL: define i32 @array_coor_1d
// CHECK: %[[LB_CHECK:.*]] = icmp sge i64 %{{.*}}, 1
// CHECK: %[[UB_CHECK:.*]] = icmp sle i64 %{{.*}}, 10
// CHECK: call void @llvm.assume(i1 %[[LB_CHECK]]){{.*}}!llvm.array.bounds
// CHECK: call void @llvm.assume(i1 %[[UB_CHECK]]){{.*}}!llvm.array.bounds

// Test 2D array with constant shape.
func.func @array_coor_2d(%addr : !fir.ref<!fir.array<10x20xi32>>, %i : index, %j : index) -> i32 {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %ref = fir.array_coor %addr(%shape) %i, %j : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
  %val = fir.load %ref : !fir.ref<i32>
  return %val : i32
}

// CHECK-LABEL: define i32 @array_coor_2d
// First dimension (i) bounds check: 1 <= i <= 10.
// CHECK: %[[I_LB_CHECK:.*]] = icmp sge i64 %{{.*}}, 1
// CHECK: %[[I_UB_CHECK:.*]] = icmp sle i64 %{{.*}}, 10
// CHECK: call void @llvm.assume(i1 %[[I_LB_CHECK]]){{.*}}!llvm.array.bounds
// CHECK: call void @llvm.assume(i1 %[[I_UB_CHECK]]){{.*}}!llvm.array.bounds
// Second dimension (j) bounds check: 1 <= j <= 20.
// CHECK: %[[J_LB_CHECK:.*]] = icmp sge i64 %{{.*}}, 1
// CHECK: %[[J_UB_CHECK:.*]] = icmp sle i64 %{{.*}}, 20
// CHECK: call void @llvm.assume(i1 %[[J_LB_CHECK]]){{.*}}!llvm.array.bounds
// CHECK: call void @llvm.assume(i1 %[[J_UB_CHECK]]){{.*}}!llvm.array.bounds

// Test array with shifted lower bounds.
func.func @array_coor_shifted(%addr : !fir.ref<!fir.array<10xi32>>, %idx : index) -> i32 {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape_shift %c5, %c10 : (index, index) -> !fir.shapeshift<1>
  %ref = fir.array_coor %addr(%shape) %idx : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
  %val = fir.load %ref : !fir.ref<i32>
  return %val : i32
}

// CHECK-LABEL: define i32 @array_coor_shifted
// Bounds check with lower bound 5: 5 <= idx <= 14.
// CHECK: %[[LB_CHECK:.*]] = icmp sge i64 %{{.*}}, 5
// CHECK: %[[UB_CHECK:.*]] = icmp sle i64 %{{.*}}, 14
// CHECK: call void @llvm.assume(i1 %[[LB_CHECK]]){{.*}}!llvm.array.bounds
// CHECK: call void @llvm.assume(i1 %[[UB_CHECK]]){{.*}}!llvm.array.bounds
