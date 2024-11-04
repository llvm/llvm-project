// Test hlfir.count operation lowering with different default integer kinds.
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i8l4r4", fir.kindmap = ""} {
  func.func @test_i8(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: i64) {
    %4 = hlfir.count %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i64) -> !hlfir.expr<?xi64>
    return
  }
}
// CHECK-LABEL: func.func @test_i8
// CHECK: %[[KIND:.*]] = arith.constant 8 : i32
// CHECK: fir.call @_FortranACountDim(%{{.*}}, %{{.*}}, %{{.*}}, %[[KIND]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = ""} {
  func.func @test_i4(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: i64) {
    %4 = hlfir.count %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i64) -> !hlfir.expr<?xi32>
    return
  }
}
// CHECK-LABEL: func.func @test_i4
// CHECK: %[[KIND:.*]] = arith.constant 4 : i32
// CHECK: fir.call @_FortranACountDim(%{{.*}}, %{{.*}}, %{{.*}}, %[[KIND]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none

module attributes {fir.defaultkind = "a1c4d8i2l4r4", fir.kindmap = ""} {
  func.func @test_i2(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: i64) {
    %4 = hlfir.count %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i64) -> !hlfir.expr<?xi16>
    return
  }
}
// CHECK-LABEL: func.func @test_i2
// CHECK: %[[KIND:.*]] = arith.constant 2 : i32
// CHECK: fir.call @_FortranACountDim(%{{.*}}, %{{.*}}, %{{.*}}, %[[KIND]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none

module attributes {fir.defaultkind = "a1c4d8i1l4r4", fir.kindmap = ""} {
  func.func @test_i1(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: i64) {
    %4 = hlfir.count %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i64) -> !hlfir.expr<?xi8>
    return
  }
}
// CHECK-LABEL: func.func @test_i1
// CHECK: %[[KIND:.*]] = arith.constant 1 : i32
// CHECK: fir.call @_FortranACountDim(%{{.*}}, %{{.*}}, %{{.*}}, %[[KIND]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none
