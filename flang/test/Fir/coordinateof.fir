// RUN: tco -emit-fir %s | tco | FileCheck %s

// tests on coordinate_of op

// CHECK-LABEL: @foo1
func @foo1(%i : i32, %j : i32, %k : i32) -> !fir.ref<f32> {
  %1 = fir.alloca !fir.array<10 x 20 x 30 x f32>
  // CHECK: %[[ptr:.*]] = bitcast [30 x [20 x [10 x
  %2 = fir.convert %1 : (!fir.ref<!fir.array<10 x 20 x 30 x f32>>) -> !fir.ref<!fir.array<10 x 20 x ? x f32>>
  // CHECK: getelementptr [20 x [10 x float]], [20 x [10 x float]]* %[[ptr]]
  %3 = fir.coordinate_of %2, %i, %j, %k : (!fir.ref<!fir.array<10 x 20 x ? x f32>>, i32, i32, i32) -> !fir.ref<f32>
  return %3 : !fir.ref<f32>
}

// CHECK-LABEL: @foo2
func @foo2(%i : i32, %j : i32, %k : i32) -> !fir.ref<f32> {
  %1 = fir.alloca !fir.array<10 x 20 x 30 x f32>
  // CHECK: %[[ptr:.*]] = bitcast [30 x [20 x [10 x
  %2 = fir.convert %1 : (!fir.ref<!fir.array<10 x 20 x 30 x f32>>) -> !fir.ref<f32>
  // CHECK: getelementptr float, float* %[[ptr]]
  %3 = fir.coordinate_of %2, %i : (!fir.ref<f32>, i32) -> !fir.ref<f32>
  return %3 : !fir.ref<f32>
}

// CHECK-LABEL: @foo3
func @foo3(%box : !fir.box<!fir.array<?xi32>>, %i : i32) -> i32 {
  // CHECK: %[[cvt:.*]] = sext i32 %
  %ii = fir.convert %i : (i32) -> index
  // CHECK: %[[gep0:.*]] = getelementptr { i32*
  // CHECK: %[[boxptr:.*]] = load i32*, i32** %[[gep0]]
  // CHECK: %[[gep1:.*]] = getelementptr i32, i32* %[[boxptr]], i64 %
  %1 = fir.coordinate_of %box, %ii : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  // CHECK: load i32, i32* %[[gep1]]
  %rv = fir.load %1 : !fir.ref<i32>
  return %rv : i32
}

// CHECK-LABEL: @foo4
func @foo4(%a : !fir.ptr<!fir.array<5x15x25xi32>>, %i : i32, %j : i64, %k : index) -> i32 {
  // CHECK: getelementptr [25 x [15 x [5 x
  %1 = fir.coordinate_of %a, %k : (!fir.ptr<!fir.array<5x15x25xi32>>, index) -> !fir.ref<!fir.array<5x15xi32>>
  // CHECK: getelementptr [15 x [5 x
  %2 = fir.coordinate_of %1, %j : (!fir.ref<!fir.array<5x15xi32>>, i64) -> !fir.ref<!fir.array<5xi32>>
  // CHECK: %[[ref:.*]] = getelementptr [5 x
  %3 = fir.coordinate_of %2, %i : (!fir.ref<!fir.array<5xi32>>, i32) -> !fir.ref<i32>
  // CHECK: load i32, i32* %[[ref]]
  %4 = fir.load %3 : !fir.ref<i32>
  return %4 : i32
}
