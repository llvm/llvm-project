// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// The base address comes from a descriptor, so each dimension's element stride
// is its byte stride divided by the element size, not a product of the inner
// extents. The storage is contiguous, so the division is exact.
// CHECK-LABEL: func.func @box_addr_strides_from_descriptor
// CHECK:       [[ESIZE:%[0-9]+]] = fir.box_elesize
// CHECK:       [[DIMS2:%[0-9]+]]:3 = fir.box_dims
// CHECK:       [[STR2:%[0-9]+]] = arith.divsi [[DIMS2]]#2, [[ESIZE]] exact
// CHECK:       [[DIMS1:%[0-9]+]]:3 = fir.box_dims
// CHECK:       [[STR1:%[0-9]+]] = arith.divsi [[DIMS1]]#2, [[ESIZE]] exact
// CHECK:       memref.reinterpret_cast {{.+}}strides: {{\[}}[[STR2]], [[STR1]], %c1{{[_0-9]*}}]
// CHECK-NOT:   fir.array_coor
func.func @box_addr_strides_from_descriptor(%box: !fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, %i: index, %j: index, %k: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 1.0 : f32
  %addr = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>) -> !fir.heap<!fir.array<?x?x?xf32>>
  %d0:3 = fir.box_dims %box, %c0 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
  %d1:3 = fir.box_dims %box, %c1 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
  %d2:3 = fir.box_dims %box, %c2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
  %shape = fir.shape_shift %d0#0, %d0#1, %d1#0, %d1#1, %d2#0, %d2#1 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
  %elem = fir.array_coor %addr(%shape) %i, %j, %k : (!fir.heap<!fir.array<?x?x?xf32>>, !fir.shapeshift<3>, index, index, index) -> !fir.ref<f32>
  fir.store %cst to %elem : !fir.ref<f32>
  return
}

// No descriptor behind the base, so the outer stride is still the product of
// the inner extents.
// CHECK-LABEL: func.func @raw_ref_strides_from_extents
// CHECK:       [[MUL:%[0-9]+]] = arith.muli %arg{{[0-9]+}}, %arg{{[0-9]+}} : index
// CHECK:       memref.reinterpret_cast {{.+}}strides: {{\[}}[[MUL]], %arg{{[0-9]+}}, %c1{{[_0-9]*}}]
// CHECK-NOT:   fir.box_dims
// CHECK-NOT:   fir.array_coor
func.func @raw_ref_strides_from_extents(%ref: !fir.ref<!fir.array<?x?x?xf32>>, %e0: index, %e1: index, %e2: index, %i: index, %j: index, %k: index) {
  %cst = arith.constant 1.0 : f32
  %shape = fir.shape %e0, %e1, %e2 : (index, index, index) -> !fir.shape<3>
  %elem = fir.array_coor %ref(%shape) %i, %j, %k : (!fir.ref<!fir.array<?x?x?xf32>>, !fir.shape<3>, index, index, index) -> !fir.ref<f32>
  fir.store %cst to %elem : !fir.ref<f32>
  return
}
