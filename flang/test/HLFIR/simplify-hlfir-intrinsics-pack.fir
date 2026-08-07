// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

func.func @pack_scalar_true_mask(%arg0: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32> {
  %true = arith.constant 1 : i1
  %mask = fir.convert %true : (i1) -> !fir.logical<4>
  %res = hlfir.pack %arg0 %mask : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL: func.func @pack_scalar_true_mask
// CHECK-NOT: hlfir.pack
// CHECK: hlfir.elemental

func.func @pack_var_mask(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<!fir.logical<4>>) -> !hlfir.expr<?xi32> {
  %mask = fir.load %arg1 : !fir.ref<!fir.logical<4>>
  %res = hlfir.pack %arg0 %mask : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL: func.func @pack_var_mask
// CHECK: hlfir.pack
// CHECK-NOT: hlfir.reshape

func.func @pack_with_vector(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32> {
  %true = arith.constant 1 : i1
  %mask = fir.convert %true : (i1) -> !fir.logical<4>
  %res = hlfir.pack %arg0 %mask vector %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL: func.func @pack_with_vector
// CHECK: hlfir.pack
// CHECK-NOT: hlfir.reshape
