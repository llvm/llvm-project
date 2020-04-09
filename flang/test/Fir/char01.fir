// RUN: tco -emit-fir %s | tco | FileCheck %s

// CHECK-LABEL: @test
func @test(%arg0 : !fir.ref<!fir.char<1>>, %arg1 : !fir.ref<!fir.char<1>>, %arg2 : i32) {
  // CHECK: getelementptr i8, i8*
  %1 = fir.coordinate_of %arg1, %arg2 : (!fir.ref<!fir.char<1>>, i32) -> !fir.ref<!fir.char<1>>
  // CHECK: load i8, i8*
  %2 = fir.load %1 : !fir.ref<!fir.char<1>>
  // CHECK: store i8
  fir.store %2 to %arg0 : !fir.ref<!fir.char<1>>
  // CHECK: ret void
  return
}
