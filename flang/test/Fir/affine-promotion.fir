// Test affine promotion pass

// RUN: fir-opt --split-input-file --promote-to-affine --affine-loop-invariant-code-motion --cse %s | FileCheck %s

!arr_d1 = type !fir.ref<!fir.array<?xf32>>
#arr_len = affine_map<()[j1,k1] -> (k1 - j1 + 1)>

func @loop_with_load_and_store(%a1: !arr_d1, %a2: !arr_d1, %a3: !arr_d1) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %len = constant 100 : index
  %dims = fir.shape %len : (index) -> !fir.shape<1>
  %siz = affine.apply #arr_len()[%c1,%len]
  %t1 = fir.alloca !fir.array<?xf32>, %siz

  fir.do_loop %i = %c1 to %len step %c1 {
    %a1_idx = fir.array_coor %a1(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %a1_v = fir.load %a1_idx : !fir.ref<f32>

    %a2_idx = fir.array_coor %a2(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %a2_v = fir.load %a2_idx : !fir.ref<f32>

    %v = addf %a1_v, %a2_v : f32
    %t1_idx = fir.array_coor %t1(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>

    fir.store %v to %t1_idx : !fir.ref<f32>
  }
  fir.do_loop %i = %c1 to %len step %c1 {
    %t1_idx = fir.array_coor %t1(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %t1_v = fir.load %t1_idx : !fir.ref<f32>

    %a2_idx = fir.array_coor %a2(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %a2_v = fir.load %a2_idx : !fir.ref<f32>

    %v = mulf %t1_v, %a2_v : f32
    %a3_idx = fir.array_coor %a3(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>

    fir.store %v to %a3_idx : !fir.ref<f32>
  }
  return
}

// CHECK:  func @loop_with_load_and_store(%[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<?xf32>>) {
// CHECK:    %[[VAL_3:.*]] = constant 1 : index
// CHECK:    %[[VAL_4:.*]] = constant 100 : index
// CHECK:    %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:    %[[VAL_6:.*]] = affine.apply #map0(){{\[}}%[[VAL_3]], %[[VAL_4]]]
// CHECK:    %[[VAL_7:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_6]]
// CHECK:    %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    %[[VAL_10:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    affine.for %[[VAL_11:.*]] = %[[VAL_3]] to #map1(){{\[}}%[[VAL_4]]] {
// CHECK:      %[[VAL_12:.*]] = affine.apply #map2(%[[VAL_11]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_13:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:      %[[VAL_14:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:      %[[VAL_15:.*]] = addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:      affine.store %[[VAL_15]], %[[VAL_10]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:    }
// CHECK:    %[[VAL_16:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    affine.for %[[VAL_17:.*]] = %[[VAL_3]] to #map1(){{\[}}%[[VAL_4]]] {
// CHECK:      %[[VAL_18:.*]] = affine.apply #map2(%[[VAL_17]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_19:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_18]]] : memref<?xf32>
// CHECK:      %[[VAL_20:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xf32>
// CHECK:      %[[VAL_21:.*]] = mulf %[[VAL_19]], %[[VAL_20]] : f32
// CHECK:      affine.store %[[VAL_21]], %[[VAL_16]]{{\[}}%[[VAL_18]]] : memref<?xf32>
// CHECK:    }
// CHECK:    return
// CHECK:  }

// -----

!arr_d1 = type !fir.ref<!fir.array<?xf32>>
#arr_len = affine_map<()[j1,k1] -> (k1 - j1 + 1)>

func @loop_with_if(%a: !arr_d1, %v: f32) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %len = constant 100 : index
  %dims = fir.shape %len : (index) -> !fir.shape<1>

  fir.do_loop %i = %c1 to %len step %c1 {
    fir.do_loop %j = %c1 to %len step %c1 {
      fir.do_loop %k = %c1 to %len step %c1 {
        %im2 = subi %i, %c2 : index
        %cond = cmpi "sgt", %im2, %c0 : index
        fir.if %cond {
          %a_idx = fir.array_coor %a(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
          fir.store %v to %a_idx : !fir.ref<f32>
        }
          %aj_idx = fir.array_coor %a(%dims) %j
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
          fir.store %v to %aj_idx : !fir.ref<f32>
          %ak_idx = fir.array_coor %a(%dims) %k
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
          fir.store %v to %ak_idx : !fir.ref<f32>
      }
    }
  }
  return
}

// CHECK: func @loop_with_if(%[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_1:.*]]: f32) {
// CHECK:   %[[VAL_2:.*]] = constant 0 : index
// CHECK:   %[[VAL_3:.*]] = constant 1 : index
// CHECK:   %[[VAL_4:.*]] = constant 2 : index
// CHECK:   %[[VAL_5:.*]] = constant 100 : index
// CHECK:   %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_7:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:   affine.for %[[VAL_8:.*]] = %[[VAL_3]] to #map0(){{\[}}%[[VAL_5]]] {
// CHECK:     %[[VAL_9:.*]] = affine.apply #map1(%[[VAL_8]]){{\[}}%[[VAL_3]], %[[VAL_5]], %[[VAL_3]]]
// CHECK:     affine.store %[[VAL_1]], %[[VAL_7]]{{\[}}%[[VAL_9]]] : memref<?xf32>
// CHECK:   }
// CHECK:   affine.for %[[VAL_10:.*]] = %[[VAL_3]] to #map0(){{\[}}%[[VAL_5]]] {
// CHECK:     %[[VAL_11:.*]] = affine.apply #map1(%[[VAL_10]]){{\[}}%[[VAL_3]], %[[VAL_5]], %[[VAL_3]]]
// CHECK:     affine.store %[[VAL_1]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<?xf32>
// CHECK:   }
// CHECK:   affine.for %[[VAL_12:.*]] = %[[VAL_3]] to #map0(){{\[}}%[[VAL_5]]] {
// CHECK:     %[[VAL_13:.*]] = subi %[[VAL_12]], %[[VAL_4]] : index
// CHECK:     affine.if #set(%[[VAL_12]]) {
// CHECK:       %[[VAL_14:.*]] = affine.apply #map1(%[[VAL_12]]){{\[}}%[[VAL_3]], %[[VAL_5]], %[[VAL_3]]]
// CHECK:       affine.store %[[VAL_1]], %[[VAL_7]]{{\[}}%[[VAL_14]]] : memref<?xf32>
// CHECK:     }
// CHECK:   }
// CHECK:   return
// CHECK: }
