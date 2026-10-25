// RUN: fir-opt %s --convert-hlfir-to-fir --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s

// Test that the code materializing the HLFIR base gets line 0 while the
// fir.declare keeps the line of the variable.

func.func @test_embox(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: index, %arg2: index) {
  %0 = fir.shape_shift %arg1, %arg2 : (index, index) -> !fir.shapeshift<1>
  %1:2 = hlfir.declare %arg0(%0) uniq_name("_QFtestEa") : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>) loc("test.f90":3:14)
  return
}
// CHECK-LABEL: func.func @test_embox(
// CHECK: fir.declare %{{.*}} loc("test.f90":3:14)
// CHECK: fir.embox %{{.*}} loc("test.f90":0:0)

func.func @test_rebox(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: index) {
  %0 = fir.shift %arg1 : (index) -> !fir.shift<1>
  %1:2 = hlfir.declare %arg0(%0) uniq_name("_QFtestEx") : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>) loc("test.f90":4:11)
  return
}
// CHECK-LABEL: func.func @test_rebox(
// CHECK: fir.declare %{{.*}} loc("test.f90":4:11)
// CHECK: fir.rebox %{{.*}} loc("test.f90":0:0)

func.func @test_emboxchar(%arg0: !fir.ref<!fir.char<1,?>>, %arg1: index) {
  %1:2 = hlfir.declare %arg0 typeparams %arg1 uniq_name("_QFtestEc") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>) loc("test.f90":5:23)
  return
}
// CHECK-LABEL: func.func @test_emboxchar(
// CHECK: fir.declare %{{.*}} loc("test.f90":5:23)
// CHECK: fir.emboxchar %{{.*}} loc("test.f90":0:0)
