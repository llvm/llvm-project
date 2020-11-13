// RUN: tco -emit-fir %s | tco | FileCheck %s

// CHECK-LABEL: @test
func @test(%arg0 : !fir.ref<!fir.char<1>>, %arg1 : !fir.ref<!fir.char<1,?>>, %arg2 : i32) {
  %0 = fir.convert %arg1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  // CHECK: getelementptr [1 x i8], [1 x i8]*
  %1 = fir.coordinate_of %0, %arg2 : (!fir.ref<!fir.array<?x!fir.char<1>>>, i32) -> !fir.ref<!fir.char<1>>
  // CHECK: load [1 x i8], [1 x i8]*
  %2 = fir.load %1 : !fir.ref<!fir.char<1>>
  // CHECK: store [1 x i8]
  fir.store %2 to %arg0 : !fir.ref<!fir.char<1>>
  // CHECK: ret void
  return
}
