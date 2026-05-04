// Test affine promotion pass

// RUN: fir-opt --split-input-file --promote-to-affine --affine-loop-invariant-code-motion --cse %s | FileCheck %s

!arr_d1 = !fir.ref<!fir.array<?xf32>>
#arr_len = affine_map<()[j1,k1] -> (k1 - j1 + 1)>

func.func @loop_with_load_and_store(%a1: !arr_d1, %a2: !arr_d1, %a3: !arr_d1) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %len = arith.constant 100 : index
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

    %v = arith.addf %a1_v, %a2_v : f32
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

    %v = arith.mulf %t1_v, %a2_v : f32
    %a3_idx = fir.array_coor %a3(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>

    fir.store %v to %a3_idx : !fir.ref<f32>
  }
  return
}

// CHECK:  func @loop_with_load_and_store(%[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<?xf32>>) {
// CHECK:    %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:    %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:    %[[VAL_5:.*]] = affine.apply #{{.*}}(){{\[}}%[[VAL_3]], %[[VAL_4]]]
// CHECK:    %[[VAL_6:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_5]]
// CHECK:    %[[VAL_7:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    affine.for %[[VAL_10:.*]] = %[[VAL_3]] to #{{.*}}(){{\[}}%[[VAL_4]]] {
// CHECK:      %[[VAL_11:.*]] = affine.apply #{{.*}}(%[[VAL_10]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_12:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<?xf32>
// CHECK:      %[[VAL_13:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_11]]] : memref<?xf32>
// CHECK:      %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:      affine.store %[[VAL_14]], %[[VAL_9]]{{\[}}%[[VAL_11]]] : memref<?xf32>
// CHECK:    }
// CHECK:    %[[VAL_15:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    affine.for %[[VAL_16:.*]] = %[[VAL_3]] to #{{.*}}(){{\[}}%[[VAL_4]]] {
// CHECK:      %[[VAL_17:.*]] = affine.apply #{{.*}}(%[[VAL_16]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_18:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_17]]] : memref<?xf32>
// CHECK:      %[[VAL_19:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xf32>
// CHECK:      %[[VAL_20:.*]] = arith.mulf %[[VAL_18]], %[[VAL_19]] : f32
// CHECK:      affine.store %[[VAL_20]], %[[VAL_15]]{{\[}}%[[VAL_17]]] : memref<?xf32>
// CHECK:    }
// CHECK:    return
// CHECK:  }

// -----

!arr_d1 = !fir.ref<!fir.array<?xf32>>
#arr_len = affine_map<()[j1,k1] -> (k1 - j1 + 1)>

func.func @loop_with_if(%a: !arr_d1, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %len = arith.constant 100 : index
  %dims = fir.shape %len : (index) -> !fir.shape<1>

  fir.do_loop %i = %c1 to %len step %c1 {
    fir.do_loop %j = %c1 to %len step %c1 {
      fir.do_loop %k = %c1 to %len step %c1 {
        %im2 = arith.subi %i, %c2 : index
        %cond = arith.cmpi "sgt", %im2, %c0 : index
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
// CHECK:   %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_3:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:   affine.for %[[VAL_5:.*]] = %[[VAL_2]] to #{{.*}}(){{\[}}%[[VAL_3]]] {
// CHECK:     %[[VAL_6:.*]] = affine.apply #{{.*}}(%[[VAL_5]]){{\[}}%[[VAL_2]], %[[VAL_3]], %[[VAL_2]]]
// CHECK:     affine.store %[[VAL_1]], %[[VAL_4]]{{\[}}%[[VAL_6]]] : memref<?xf32>
// CHECK:   }
// CHECK:   affine.for %[[VAL_7:.*]] = %[[VAL_2]] to #{{.*}}(){{\[}}%[[VAL_3]]] {
// CHECK:     %[[VAL_8:.*]] = affine.apply #{{.*}}(%[[VAL_7]]){{\[}}%[[VAL_2]], %[[VAL_3]], %[[VAL_2]]]
// CHECK:     affine.store %[[VAL_1]], %[[VAL_4]]{{\[}}%[[VAL_8]]] : memref<?xf32>
// CHECK:   }
// CHECK:   affine.for %[[VAL_9:.*]] = %[[VAL_2]] to #{{.*}}(){{\[}}%[[VAL_3]]] {
// CHECK:     affine.if #set(%[[VAL_9]]) {
// CHECK:       %[[VAL_10:.*]] = affine.apply #{{.*}}(%[[VAL_9]]){{\[}}%[[VAL_2]], %[[VAL_3]], %[[VAL_2]]]
// CHECK:       affine.store %[[VAL_1]], %[[VAL_4]]{{\[}}%[[VAL_10]]] : memref<?xf32>
// CHECK:     }
// CHECK:   }
// CHECK:   return
// CHECK: }

func.func @loop_with_result(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100x100xf32>>, %arg2: !fir.ref<!fir.array<100xf32>>) -> f32 {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
  %2 = fir.alloca i32
  %3:2 = fir.do_loop %arg3 = %c1 to %c100 step %c1 iter_args(%arg4 = %cst) -> (index, f32) {
    %8 = fir.array_coor %arg0(%0) %arg3 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %9 = fir.load %8 : !fir.ref<f32>
    %10 = arith.addf %arg4, %9 fastmath<contract> : f32
    %11 = arith.addi %arg3, %c1 overflow<nsw> : index
    fir.result %11, %10 : index, f32
  }
  %4:2 = fir.do_loop %arg3 = %c1 to %c100 step %c1 iter_args(%arg4 = %3#1) -> (index, f32) {
    %8 = fir.array_coor %arg1(%1) %c1, %arg3 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
    %9 = fir.convert %8 : (!fir.ref<f32>) -> !fir.ref<!fir.array<100xf32>>
    %10 = fir.do_loop %arg5 = %c1 to %c100 step %c1 iter_args(%arg6 = %arg4) -> (f32) {
      %12 = fir.array_coor %9(%0) %arg5 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
      %13 = fir.load %12 : !fir.ref<f32>
      %14 = arith.addf %arg6, %13 fastmath<contract> : f32
      fir.result %14 : f32
    }
    %11 = arith.addi %arg3, %c1 overflow<nsw> : index
    fir.result %11, %10 : index, f32
  }
  %5:2 = fir.do_loop %arg3 = %c1 to %c100 step %c1 iter_args(%arg4 = %4#1, %arg5 = %cst) -> (f32, f32) {
    %8 = fir.array_coor %arg0(%0) %arg3 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %9 = fir.load %8 : !fir.ref<f32>
    %10 = arith.addf %arg4, %9 fastmath<contract> : f32
    %11 = fir.array_coor %arg2(%0) %arg3 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    %12 = fir.load %11 : !fir.ref<f32>
    %13 = arith.addf %arg5, %12 fastmath<contract> : f32
    fir.result %10, %13 : f32, f32
  }
  %6 = arith.addf %5#0, %5#1 fastmath<contract> : f32
  %7 = fir.convert %4#0 : (index) -> i32
  fir.store %7 to %2 : !fir.ref<i32>
  return %6 : f32
}

// CHECK-LABEL:   func.func @loop_with_result(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<100x100xf32>>,
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<!fir.array<100xf32>>) -> f32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = fir.alloca i32
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<100xf32>>) -> memref<?xf32>
// CHECK:           %[[VAL_6:.*]] = affine.for %[[VAL_7:.*]] = %[[VAL_0]] to #{{.*}}(){{\[}}%[[VAL_2]]] iter_args(%[[VAL_8:.*]] = %[[VAL_1]]) -> (f32) {
// CHECK:             %[[VAL_9:.*]] = affine.apply #{{.*}}(%[[VAL_7]]){{\[}}%[[VAL_0]], %[[VAL_2]], %[[VAL_0]]]
// CHECK:             %[[VAL_10:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<?xf32>
// CHECK:             %[[VAL_11:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] fastmath<contract> : f32
// CHECK:             affine.yield %[[VAL_11]] : f32
// CHECK:           }
// CHECK:           %[[VAL_12:.*]]:2 = fir.do_loop %[[VAL_13:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_0]] iter_args(%[[VAL_14:.*]] = %[[VAL_6]]) -> (index, f32) {
// CHECK:             %[[VAL_15:.*]] = fir.array_coor %[[ARG1]](%[[VAL_3]]) %[[VAL_0]], %[[VAL_13]] : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<f32>) -> !fir.ref<!fir.array<100xf32>>
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.array<100xf32>>) -> memref<?xf32>
// CHECK:             %[[VAL_18:.*]] = affine.for %[[VAL_19:.*]] = %[[VAL_0]] to #{{.*}}(){{\[}}%[[VAL_2]]] iter_args(%[[VAL_20:.*]] = %[[VAL_14]]) -> (f32) {
// CHECK:               %[[VAL_21:.*]] = affine.apply #{{.*}}(%[[VAL_19]]){{\[}}%[[VAL_0]], %[[VAL_2]], %[[VAL_0]]]
// CHECK:               %[[VAL_22:.*]] = affine.load %[[VAL_17]]{{\[}}%[[VAL_21]]] : memref<?xf32>
// CHECK:               %[[VAL_23:.*]] = arith.addf %[[VAL_20]], %[[VAL_22]] fastmath<contract> : f32
// CHECK:               affine.yield %[[VAL_23]] : f32
// CHECK:             }
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_13]], %[[VAL_0]] overflow<nsw> : index
// CHECK:             fir.result %[[VAL_24]], %[[VAL_18]] : index, f32
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[ARG2]] : (!fir.ref<!fir.array<100xf32>>) -> memref<?xf32>
// CHECK:           %[[VAL_26:.*]]:2 = affine.for %[[VAL_27:.*]] = %[[VAL_0]] to #{{.*}}(){{\[}}%[[VAL_2]]] iter_args(%[[VAL_28:.*]] = %[[VAL_12]]#1, %[[VAL_29:.*]] = %[[VAL_1]]) -> (f32, f32) {
// CHECK:             %[[VAL_30:.*]] = affine.apply #{{.*}}(%[[VAL_27]]){{\[}}%[[VAL_0]], %[[VAL_2]], %[[VAL_0]]]
// CHECK:             %[[VAL_31:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_30]]] : memref<?xf32>
// CHECK:             %[[VAL_32:.*]] = arith.addf %[[VAL_28]], %[[VAL_31]] fastmath<contract> : f32
// CHECK:             %[[VAL_33:.*]] = affine.load %[[VAL_25]]{{\[}}%[[VAL_30]]] : memref<?xf32>
// CHECK:             %[[VAL_34:.*]] = arith.addf %[[VAL_29]], %[[VAL_33]] fastmath<contract> : f32
// CHECK:             affine.yield %[[VAL_32]], %[[VAL_34]] : f32, f32
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = arith.addf %[[VAL_26]]#0, %[[VAL_26]]#1 fastmath<contract> : f32
// CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_12]]#0 : (index) -> i32
// CHECK:           fir.store %[[VAL_36]] to %[[VAL_4]] : !fir.ref<i32>
// CHECK:           return %[[VAL_35]] : f32
// CHECK:         }
