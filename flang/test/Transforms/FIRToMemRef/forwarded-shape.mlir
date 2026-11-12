// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

func.func @forwarded_shape_store(%pred : i1, %n1 : index, %n2 : index,
    %arg0: !fir.ref<!fir.array<?xi32>>) {
  cf.cond_br %pred, ^bb1, ^bb2
^bb1:
  %sh1 = fir.shape %n1 : (index) -> !fir.shape<1>
  cf.br ^bb3(%sh1 : !fir.shape<1>)
^bb2:
  %sh2 = fir.shape %n2 : (index) -> !fir.shape<1>
  cf.br ^bb3(%sh2 : !fir.shape<1>)
^bb3(%phi : !fir.shape<1>):
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  %elt = fir.array_coor %arg0(%phi) %c1
      : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c42 to %elt : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @forwarded_shape_store
// CHECK:       fir.shape_extents
// CHECK:       memref.store
// CHECK-NOT:   fir.array_coor
