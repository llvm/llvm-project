// Test affine demotion pass

// RUN: fir-opt --split-input-file --demote-affine %s | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (s1 - s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
#map2 = affine_map<(d0)[s0, s1, s2] -> (d0 * s2 - s0)>
module  {
  func @calc(%arg0: !fir.ref<!fir.array<?xf32>>, %arg1: !fir.ref<!fir.array<?xf32>>, %arg2: !fir.ref<!fir.array<?xf32>>) {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = fir.shape %c100 : (index) -> !fir.shape<1>
    %1 = affine.apply #map0()[%c1, %c100]
    %2 = fir.alloca !fir.array<?xf32>, %1
    %3 = fir.convert %arg0 : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    %4 = fir.convert %arg1 : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    %5 = fir.convert %2 : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    affine.for %arg3 = %c1 to #map1()[%c100] {
      %7 = affine.apply #map2(%arg3)[%c1, %c100, %c1]
      %8 = affine.load %3[%7] : memref<?xf32>
      %9 = affine.load %4[%7] : memref<?xf32>
      %10 = arith.addf %8, %9 : f32
      affine.store %10, %5[%7] : memref<?xf32>
    }
    %6 = fir.convert %arg2 : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    affine.for %arg3 = %c1 to #map1()[%c100] {
      %7 = affine.apply #map2(%arg3)[%c1, %c100, %c1]
      %8 = affine.load %5[%7] : memref<?xf32>
      %9 = affine.load %4[%7] : memref<?xf32>
      %10 = arith.mulf %8, %9 : f32
      affine.store %10, %6[%7] : memref<?xf32>
    }
    return
  }
}

// CHECK:  func @calc(%[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<?xf32>>) {
// CHECK:    %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:    %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:    %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:    %[[VAL_6:.*]] = arith.constant 100 : index
// CHECK:    %[[VAL_7:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_6]]
// CHECK:    %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:    %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:    %[[VAL_10:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:    affine.for %[[VAL_11:.*]] = 1 to 101 {
// CHECK:      %[[VAL_12:.*]] = affine.apply #map(%[[VAL_11]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_12]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:      %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
// CHECK:      %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_12]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:      %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<f32>
// CHECK:      %[[VAL_17:.*]] = arith.addf %[[VAL_14]], %[[VAL_16]] : f32
// CHECK:      %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_12]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:      fir.store %[[VAL_17]] to %[[VAL_18]] : !fir.ref<f32>
// CHECK:    }
// CHECK:    %[[VAL_19:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:    affine.for %[[VAL_20:.*]] = 1 to 101 {
// CHECK:      %[[VAL_21:.*]] = affine.apply #map(%[[VAL_20]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_21]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:      %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<f32>
// CHECK:      %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_21]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:      %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<f32>
// CHECK:      %[[VAL_26:.*]] = arith.mulf %[[VAL_23]], %[[VAL_25]] : f32
// CHECK:      %[[VAL_27:.*]] = fir.coordinate_of %[[VAL_19]], %[[VAL_21]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:      fir.store %[[VAL_26]] to %[[VAL_27]] : !fir.ref<f32>
// CHECK:    }
// CHECK:    return
// CHECK:  }
