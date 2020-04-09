// RUN: tco -emit-fir %s | tco | FileCheck %s

// CHECK-LABEL: @foo1
func @foo1(%i : i32, %j : i32, %k : i32) -> !fir.ref<f32> {
  %1 = fir.alloca !fir.array<10 x 20 x 30 x f32>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<10 x 20 x 30 x f32>>) -> !fir.ref<!fir.array<10 x 20 x ? x f32>>
  // CHECK: getelementptr [20 x [10 x float]], [20 x [10 x float]]* %
  %3 = fir.coordinate_of %2, %i, %j, %k : (!fir.ref<!fir.array<10 x 20 x ? x f32>>, i32, i32, i32) -> !fir.ref<f32>
  return %3 : !fir.ref<f32>
}

// CHECK-LABEL: @foo2
func @foo2(%i : i32, %j : i32, %k : i32) -> !fir.ref<f32> {
  %1 = fir.alloca !fir.array<10 x 20 x 30 x f32>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<10 x 20 x 30 x f32>>) -> !fir.ref<f32>
  // CHECK: getelementptr float, float* %
  %3 = fir.coordinate_of %2, %i : (!fir.ref<f32>, i32) -> !fir.ref<f32>
  return %3 : !fir.ref<f32>
}
