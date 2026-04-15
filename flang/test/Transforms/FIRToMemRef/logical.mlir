// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s
// CHECK-LABEL: func.func @load_scalar
// CHECK:       [[DUMMY:%0]] = fir.undefined !fir.dscope
// CHECK-NEXT:  [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]]
// CHECK-NEXT:  [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.logical<4>>) -> memref<i32>
// CHECK-NEXT:  [[LOAD:%[0-9]+]] = memref.load [[CONVERT]][] : memref<i32>
// CHECK-NEXT:  fir.bitcast [[LOAD]] : (i32) -> !fir.logical<4>
func.func @load_scalar(%arg0: !fir.ref<!fir.logical<4>>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "a"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %2 = fir.load %1 : !fir.ref<!fir.logical<4>>
  return
}

// CHECK-LABEL: func.func @store_scalar
// CHECK:       [[CONSTTRUE:%.+]] = arith.constant true
// CHECK:       [[DUMMY:%0]] = fir.undefined !fir.dscope
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]]
// CHECK-NEXT:  [[CONVERT:%[0-9]+]] = fir.convert [[CONSTTRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:  [[CONVERT1:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.logical<4>>) -> memref<i32>
// CHECK-NEXT:  [[INT:%[0-9]+]] = fir.bitcast [[CONVERT]] : (!fir.logical<4>) -> i32
// CHECK-NEXT:  memref.store [[INT]], [[CONVERT1]][] : memref<i32>
func.func @store_scalar(%arg0: !fir.ref<!fir.logical<4>>) {
  %true = arith.constant true
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "b"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %2 = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %2 to %1 : !fir.ref<!fir.logical<4>>
  return
}

// CHECK-LABEL: func.func @logical_and_logical
// CHECK:       fir.logical_and {{.*}} : i32
func.func @logical_and_logical(%arg0: !fir.logical<4>, %arg1: !fir.logical<4>) -> !fir.logical<4> {
  %0 = fir.logical_and %arg0, %arg1 : !fir.logical<4>
  return %0 : !fir.logical<4>
}

// CHECK-LABEL: func.func @logical_or_logical
// CHECK:       fir.logical_or {{.*}} : i32
func.func @logical_or_logical(%arg0: !fir.logical<4>, %arg1: !fir.logical<4>) -> !fir.logical<4> {
  %0 = fir.logical_or %arg0, %arg1 : !fir.logical<4>
  return %0 : !fir.logical<4>
}

// CHECK-LABEL: func.func @eqv_logical
// CHECK:       fir.eqv {{.*}} : i32
func.func @eqv_logical(%arg0: !fir.logical<4>, %arg1: !fir.logical<4>) -> !fir.logical<4> {
  %0 = fir.eqv %arg0, %arg1 : !fir.logical<4>
  return %0 : !fir.logical<4>
}

// CHECK-LABEL: func.func @neqv_logical
// CHECK:       fir.neqv {{.*}} : i32
func.func @neqv_logical(%arg0: !fir.logical<4>, %arg1: !fir.logical<4>) -> !fir.logical<4> {
  %0 = fir.neqv %arg0, %arg1 : !fir.logical<4>
  return %0 : !fir.logical<4>
}
