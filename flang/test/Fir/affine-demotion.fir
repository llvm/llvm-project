// Test affine demotion pass

// RUN: fir-opt --split-input-file --demote-affine %s | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (s1 - s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
#map2 = affine_map<(d0) -> (d0 - 1)>
module  {
  func.func @calc(%arg0: !fir.ref<!fir.array<?xf32>>, %arg1: !fir.ref<!fir.array<?xf32>>, %arg2: !fir.ref<!fir.array<?xf32>>) {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = fir.shape %c100 : (index) -> !fir.shape<1>
    %a = fir.declare %arg0(%0) {uniq_name = "a"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
    %b = fir.declare %arg1(%0) {uniq_name = "b"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
    %1 = affine.apply #map0()[%c1, %c100]
    %2 = fir.alloca !fir.array<?xf32>, %1
    %t = fir.declare %2(%0) {uniq_name = "t"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
    %3 = fir.convert %a : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    %4 = fir.convert %b : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    %5 = fir.convert %t : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    affine.for %arg3 = %c1 to #map1()[%c100] {
      %7 = affine.apply #map2(%arg3)
      %8 = affine.load %3[%7] : memref<?xf32>
      %9 = affine.load %4[%7] : memref<?xf32>
      %10 = arith.addf %8, %9 : f32
      affine.store %10, %5[%7] : memref<?xf32>
    }
    %c = fir.declare %arg2(%0) {uniq_name = "c"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
    %6 = fir.convert %c : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    affine.for %arg3 = %c1 to #map1()[%c100] {
      %7 = affine.apply #map2(%arg3)
      %8 = affine.load %5[%7] : memref<?xf32>
      %9 = affine.load %4[%7] : memref<?xf32>
      %10 = arith.mulf %8, %9 : f32
      affine.store %10, %6[%7] : memref<?xf32>
    }
    return
  }
}

// CHECK:  func @calc(%[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<?xf32>>, %[[ARG2:.*]]: !fir.ref<!fir.array<?xf32>>) {
// CHECK:    %[[C1:.*]] = arith.constant 1 : index
// CHECK:    %[[C100:.*]] = arith.constant 100 : index
// CHECK:    %[[SHP:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
// CHECK:    %[[A:.*]] = fir.declare %[[ARG0]](%[[SHP]]) {uniq_name = "a"}
// CHECK:    %[[B:.*]] = fir.declare %[[ARG1]](%[[SHP]]) {uniq_name = "b"}
// CHECK:    %[[ALLOCSZ:.*]] = arith.constant 100 : index
// CHECK:    %[[ALLOC:.*]] = fir.alloca !fir.array<?xf32>, %[[ALLOCSZ]]
// CHECK:    %[[T:.*]] = fir.declare %[[ALLOC]](%[[SHP]]) {uniq_name = "t"}
// fir.convert removed — affine.load/store demoted to fir.array_coor + fir.load/store:
// CHECK:    affine.for %[[IV1:.*]] = 1 to 101 {
// CHECK:      %[[IDX1:.*]] = affine.apply #{{.*}}(%[[IV1]])
// 0-based → 1-based for a(i):
// CHECK:      %[[C1_A:.*]] = arith.constant 1 : index
// CHECK:      %[[FI_A:.*]] = arith.addi %[[IDX1]], %[[C1_A]] : index
// CHECK:      %[[A_COOR:.*]] = fir.array_coor %[[A]](%[[SHP]]) %[[FI_A]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:      %[[A_VAL:.*]] = fir.load %[[A_COOR]] : !fir.ref<f32>
// 0-based → 1-based for b(i):
// CHECK:      %[[C1_B:.*]] = arith.constant 1 : index
// CHECK:      %[[FI_B:.*]] = arith.addi %[[IDX1]], %[[C1_B]] : index
// CHECK:      %[[B_COOR1:.*]] = fir.array_coor %[[B]](%[[SHP]]) %[[FI_B]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:      %[[B_VAL1:.*]] = fir.load %[[B_COOR1]] : !fir.ref<f32>
// CHECK:      %[[ADD:.*]] = arith.addf %[[A_VAL]], %[[B_VAL1]] : f32
// 0-based → 1-based for t(i):
// CHECK:      %[[C1_T:.*]] = arith.constant 1 : index
// CHECK:      %[[FI_T:.*]] = arith.addi %[[IDX1]], %[[C1_T]] : index
// CHECK:      %[[T_COOR1:.*]] = fir.array_coor %[[T]](%[[SHP]]) %[[FI_T]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:      fir.store %[[ADD]] to %[[T_COOR1]] : !fir.ref<f32>
// CHECK:    }
// CHECK:    %[[C:.*]] = fir.declare %[[ARG2]](%[[SHP]]) {uniq_name = "c"}
// CHECK:    affine.for %[[IV2:.*]] = 1 to 101 {
// CHECK:      %[[IDX2:.*]] = affine.apply #{{.*}}(%[[IV2]])
// 0-based → 1-based for t(i):
// CHECK:      %[[C1_T2:.*]] = arith.constant 1 : index
// CHECK:      %[[FI_T2:.*]] = arith.addi %[[IDX2]], %[[C1_T2]] : index
// CHECK:      %[[T_COOR2:.*]] = fir.array_coor %[[T]](%[[SHP]]) %[[FI_T2]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:      %[[T_VAL:.*]] = fir.load %[[T_COOR2]] : !fir.ref<f32>
// 0-based → 1-based for b(i):
// CHECK:      %[[C1_B2:.*]] = arith.constant 1 : index
// CHECK:      %[[FI_B2:.*]] = arith.addi %[[IDX2]], %[[C1_B2]] : index
// CHECK:      %[[B_COOR2:.*]] = fir.array_coor %[[B]](%[[SHP]]) %[[FI_B2]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:      %[[B_VAL2:.*]] = fir.load %[[B_COOR2]] : !fir.ref<f32>
// CHECK:      %[[MUL:.*]] = arith.mulf %[[T_VAL]], %[[B_VAL2]] : f32
// 0-based → 1-based for c(i):
// CHECK:      %[[C1_C:.*]] = arith.constant 1 : index
// CHECK:      %[[FI_C:.*]] = arith.addi %[[IDX2]], %[[C1_C]] : index
// CHECK:      %[[C_COOR:.*]] = fir.array_coor %[[C]](%[[SHP]]) %[[FI_C]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:      fir.store %[[MUL]] to %[[C_COOR]] : !fir.ref<f32>
// CHECK:    }
// CHECK:    return
// CHECK:  }

// -----

// Test: 2D nested loop demotion with static-shape arrays.
#map2 = affine_map<()[s0] -> (s0 + 1)>
#map3 = affine_map<(d0) -> (d0 - 1)>
module {
  func.func @calc_2d_static(%arg0: !fir.ref<!fir.array<100x100xf32>>, %arg1: !fir.ref<!fir.array<100x100xf32>>) {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
    %1 = fir.convert %arg0 : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
    %2 = fir.convert %arg1 : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
    affine.for %arg2 = %c1 to #map2()[%c100] {
      %3 = affine.apply #map3(%arg2)
      affine.for %arg3 = %c1 to #map2()[%c100] {
        %4 = affine.apply #map3(%arg3)
        %5 = affine.load %1[%3, %4] : memref<100x100xf32>
        affine.store %5, %2[%3, %4] : memref<100x100xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func @calc_2d_static(
// CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<100x100xf32>>, %[[B:.*]]: !fir.ref<!fir.array<100x100xf32>>)
// CHECK:    %[[C1:.*]] = arith.constant 1 : index
// CHECK:    %[[C100:.*]] = arith.constant 100 : index
// CHECK:    %[[SHP:.*]] = fir.shape %[[C100]], %[[C100]] : (index, index) -> !fir.shape<2>
// fir.convert removed — static arrays use fir.coordinate_of directly:
// CHECK:    affine.for %[[I:.*]] = 1 to 101 {
// CHECK:      %[[IDXI:.*]] = affine.apply #{{.*}}(%[[I]])
// CHECK:      affine.for %[[J:.*]] = 1 to 101 {
// CHECK:        %[[IDXJ:.*]] = affine.apply #{{.*}}(%[[J]])
// Indices reversed (row-major memref → column-major Fortran):
// CHECK:        %[[A_COOR:.*]] = fir.coordinate_of %[[A]], %[[IDXJ]], %[[IDXI]] : (!fir.ref<!fir.array<100x100xf32>>, index, index) -> !fir.ref<f32>
// CHECK:        %[[A_VAL:.*]] = fir.load %[[A_COOR]] : !fir.ref<f32>
// CHECK:        %[[B_COOR:.*]] = fir.coordinate_of %[[B]], %[[IDXJ]], %[[IDXI]] : (!fir.ref<!fir.array<100x100xf32>>, index, index) -> !fir.ref<f32>
// CHECK:        fir.store %[[A_VAL]] to %[[B_COOR]] : !fir.ref<f32>
// CHECK:      }
// CHECK:    }
// CHECK:    return

// -----

// Test: Triangular loop demotion — inner bound depends on outer IV.
#map4 = affine_map<()[s0] -> (s0 + 1)>
#map5 = affine_map<(d0) -> (d0 + 1)>
#map6 = affine_map<(d0) -> (d0 - 1)>
module {
  func.func @triangular_demotion(%arg0: !fir.ref<!fir.array<100x100xf32>>, %arg1: !fir.ref<!fir.array<100x100xf32>>) {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
    %1 = fir.convert %arg0 : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
    %2 = fir.convert %arg1 : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
    affine.for %arg2 = %c1 to #map4()[%c100] {
      %3 = affine.apply #map6(%arg2)
      affine.for %arg3 = %c1 to #map5(%arg2) {
        %4 = affine.apply #map6(%arg3)
        %5 = affine.load %1[%3, %4] : memref<100x100xf32>
        affine.store %5, %2[%3, %4] : memref<100x100xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func @triangular_demotion(
// CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<100x100xf32>>, %[[B:.*]]: !fir.ref<!fir.array<100x100xf32>>)
// Outer: constant bound; Inner: IV-dependent bound
// CHECK:       affine.for %[[I:.*]] = 1 to 101 {
// CHECK:         affine.for %[[J:.*]] = 1 to #{{.*}}(%[[I]]) {
// CHECK:           fir.coordinate_of %[[A]], %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<100x100xf32>>, index, index) -> !fir.ref<f32>
// CHECK:           fir.load %{{.*}} : !fir.ref<f32>
// CHECK:           fir.coordinate_of %[[B]], %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<100x100xf32>>, index, index) -> !fir.ref<f32>
// CHECK:           fir.store %{{.*}} to %{{.*}} : !fir.ref<f32>
// CHECK:         }
// CHECK:       }
// CHECK:       return

// -----

// Test: box-based demotion (allocatable / assumed-shape dummy).
// When only a `fir.box` is available (case 3 of findShapeForBase), the
// element address must be computed through the box rather than assuming
// a 1-based lower bound. AffineDemotion is expected to:
//   * read each dimension's runtime lower bound via `fir.box_dims`,
//   * pack those bounds into a `fir.shift`,
//   * pass the shift to `fir.array_coor`, and
//   * shift the 0-based affine indices by lb to get user-visible
//     Fortran indices.
#map_box = affine_map<()[s0] -> (s0 + 1)>
#map_box_id = affine_map<(d0) -> (d0 - 1)>
module {
  func.func @box_path(%arg0: !fir.box<!fir.array<?xf32>>) {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %base = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
    %m = fir.convert %base : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
    affine.for %arg1 = %c1 to #map_box()[%c100] {
      %i = affine.apply #map_box_id(%arg1)
      %v = affine.load %m[%i] : memref<?xf32>
      affine.store %v, %m[%i] : memref<?xf32>
    }
    return
  }
}

// CHECK-LABEL: func @box_path(
// CHECK-SAME:    %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>)
// CHECK:       %[[BASE:.*]] = fir.box_addr %[[BOX]]
// fir.convert is removed by the pass; affine.load/store on the box
// path must read each dim's lb from the box and pass an explicit shift.
// CHECK:       affine.for %[[IV:.*]] = 1 to 101 {
// CHECK:         %[[IDX0:.*]] = affine.apply #{{.*}}(%[[IV]])
// Load: read lb from box, build shift, address through array_coor.
// CHECK:         %[[LD_DIM:.*]] = arith.constant 0 : i32
// CHECK:         %[[LD_DIMS:.*]]:3 = fir.box_dims %[[BOX]], %[[LD_DIM]] : (!fir.box<!fir.array<?xf32>>, i32) -> (index, index, index)
// CHECK:         %[[LD_SHIFT:.*]] = fir.shift %[[LD_DIMS]]#0 : (index) -> !fir.shift<1>
// CHECK:         %[[LD_FI:.*]] = arith.addi %[[IDX0]], %[[LD_DIMS]]#0 : index
// CHECK:         %[[LD_COOR:.*]] = fir.array_coor %[[BOX]](%[[LD_SHIFT]]) %[[LD_FI]] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
// CHECK:         %[[V:.*]] = fir.load %[[LD_COOR]] : !fir.ref<f32>
// Store: same pattern.
// CHECK:         %[[ST_DIM:.*]] = arith.constant 0 : i32
// CHECK:         %[[ST_DIMS:.*]]:3 = fir.box_dims %[[BOX]], %[[ST_DIM]] : (!fir.box<!fir.array<?xf32>>, i32) -> (index, index, index)
// CHECK:         %[[ST_SHIFT:.*]] = fir.shift %[[ST_DIMS]]#0 : (index) -> !fir.shift<1>
// CHECK:         %[[ST_FI:.*]] = arith.addi %[[IDX0]], %[[ST_DIMS]]#0 : index
// CHECK:         %[[ST_COOR:.*]] = fir.array_coor %[[BOX]](%[[ST_SHIFT]]) %[[ST_FI]] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
// CHECK:         fir.store %[[V]] to %[[ST_COOR]] : !fir.ref<f32>
// CHECK:       }
// CHECK:       return
