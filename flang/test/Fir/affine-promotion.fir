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
// CHECK:    %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:    %[[VAL_6:.*]] = affine.apply #{{.*}}(){{\[}}%[[VAL_3]], %[[VAL_4]]]
// CHECK:    %[[VAL_7:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_6]]
// CHECK:    %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    %[[VAL_10:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    affine.for %[[VAL_11:.*]] = %[[VAL_3]] to #{{.*}}(){{\[}}%[[VAL_4]]] {
// CHECK:      %[[VAL_12:.*]] = affine.apply #{{.*}}(%[[VAL_11]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_13:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:      %[[VAL_14:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:      %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:      affine.store %[[VAL_15]], %[[VAL_10]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:    }
// CHECK:    %[[VAL_16:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    affine.for %[[VAL_17:.*]] = %[[VAL_3]] to #{{.*}}(){{\[}}%[[VAL_4]]] {
// CHECK:      %[[VAL_18:.*]] = affine.apply #{{.*}}(%[[VAL_17]]){{\[}}%[[VAL_3]], %[[VAL_4]], %[[VAL_3]]]
// CHECK:      %[[VAL_19:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_18]]] : memref<?xf32>
// CHECK:      %[[VAL_20:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xf32>
// CHECK:      %[[VAL_21:.*]] = arith.mulf %[[VAL_19]], %[[VAL_20]] : f32
// CHECK:      affine.store %[[VAL_21]], %[[VAL_16]]{{\[}}%[[VAL_18]]] : memref<?xf32>
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
// CHECK:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:   %[[VAL_5:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_7:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:   affine.for %[[VAL_8:.*]] = %[[VAL_3]] to #{{.*}}(){{\[}}%[[VAL_5]]] {
// CHECK:     %[[VAL_9:.*]] = affine.apply #{{.*}}(%[[VAL_8]]){{\[}}%[[VAL_3]], %[[VAL_5]], %[[VAL_3]]]
// CHECK:     affine.store %[[VAL_1]], %[[VAL_7]]{{\[}}%[[VAL_9]]] : memref<?xf32>
// CHECK:   }
// CHECK:   affine.for %[[VAL_10:.*]] = %[[VAL_3]] to #{{.*}}(){{\[}}%[[VAL_5]]] {
// CHECK:     %[[VAL_11:.*]] = affine.apply #{{.*}}(%[[VAL_10]]){{\[}}%[[VAL_3]], %[[VAL_5]], %[[VAL_3]]]
// CHECK:     affine.store %[[VAL_1]], %[[VAL_7]]{{\[}}%[[VAL_11]]] : memref<?xf32>
// CHECK:   }
// CHECK:   affine.for %[[VAL_12:.*]] = %[[VAL_3]] to #{{.*}}(){{\[}}%[[VAL_5]]] {
// CHECK:     %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_4]] : index
// CHECK:     affine.if #set(%[[VAL_12]]) {
// CHECK:       %[[VAL_14:.*]] = affine.apply #{{.*}}(%[[VAL_12]]){{\[}}%[[VAL_3]], %[[VAL_5]], %[[VAL_3]]]
// CHECK:       affine.store %[[VAL_1]], %[[VAL_7]]{{\[}}%[[VAL_14]]] : memref<?xf32>
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
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]] = fir.alloca i32
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<100xf32>>) -> memref<?xf32>
// CHECK:           %[[VAL_7:.*]] = affine.for %[[VAL_8:.*]] = %[[VAL_0]] to #{{.*}}(){{\[}}%[[VAL_2]]] iter_args(%[[VAL_9:.*]] = %[[VAL_1]]) -> (f32) {
// CHECK:             %[[VAL_10:.*]] = affine.apply #{{.*}}(%[[VAL_8]]){{\[}}%[[VAL_0]], %[[VAL_2]], %[[VAL_0]]]
// CHECK:             %[[VAL_11:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf32>
// CHECK:             %[[VAL_12:.*]] = arith.addf %[[VAL_9]], %[[VAL_11]] fastmath<contract> : f32
// CHECK:             affine.yield %[[VAL_12]] : f32
// CHECK:           }
// CHECK:           %[[VAL_13:.*]]:2 = fir.do_loop %[[VAL_14:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_0]] iter_args(%[[VAL_15:.*]] = %[[VAL_7]]) -> (index, f32) {
// CHECK:             %[[VAL_16:.*]] = fir.array_coor %[[ARG1]](%[[VAL_4]]) %[[VAL_0]], %[[VAL_14]] : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<f32>) -> !fir.ref<!fir.array<100xf32>>
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.array<100xf32>>) -> memref<?xf32>
// CHECK:             %[[VAL_19:.*]] = affine.for %[[VAL_20:.*]] = %[[VAL_0]] to #{{.*}}(){{\[}}%[[VAL_2]]] iter_args(%[[VAL_21:.*]] = %[[VAL_15]]) -> (f32) {
// CHECK:               %[[VAL_22:.*]] = affine.apply #{{.*}}(%[[VAL_20]]){{\[}}%[[VAL_0]], %[[VAL_2]], %[[VAL_0]]]
// CHECK:               %[[VAL_23:.*]] = affine.load %[[VAL_18]]{{\[}}%[[VAL_22]]] : memref<?xf32>
// CHECK:               %[[VAL_24:.*]] = arith.addf %[[VAL_21]], %[[VAL_23]] fastmath<contract> : f32
// CHECK:               affine.yield %[[VAL_24]] : f32
// CHECK:             }
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_14]], %[[VAL_0]] overflow<nsw> : index
// CHECK:             fir.result %[[VAL_25]], %[[VAL_19]] : index, f32
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[ARG2]] : (!fir.ref<!fir.array<100xf32>>) -> memref<?xf32>
// CHECK:           %[[VAL_27:.*]]:2 = affine.for %[[VAL_28:.*]] = %[[VAL_0]] to #{{.*}}(){{\[}}%[[VAL_2]]] iter_args(%[[VAL_29:.*]] = %[[VAL_30:.*]]#1, %[[VAL_31:.*]] = %[[VAL_1]]) -> (f32, f32) {
// CHECK:             %[[VAL_32:.*]] = affine.apply #{{.*}}(%[[VAL_28]]){{\[}}%[[VAL_0]], %[[VAL_2]], %[[VAL_0]]]
// CHECK:             %[[VAL_33:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_32]]] : memref<?xf32>
// CHECK:             %[[VAL_34:.*]] = arith.addf %[[VAL_29]], %[[VAL_33]] fastmath<contract> : f32
// CHECK:             %[[VAL_35:.*]] = affine.load %[[VAL_26]]{{\[}}%[[VAL_32]]] : memref<?xf32>
// CHECK:             %[[VAL_36:.*]] = arith.addf %[[VAL_31]], %[[VAL_35]] fastmath<contract> : f32
// CHECK:             affine.yield %[[VAL_34]], %[[VAL_36]] : f32, f32
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = arith.addf %[[VAL_38:.*]]#0, %[[VAL_38]]#1 fastmath<contract> : f32
// CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_40:.*]]#0 : (index) -> i32
// CHECK:           fir.store %[[VAL_39]] to %[[VAL_5]] : !fir.ref<i32>
// CHECK:           return %[[VAL_37]] : f32
// CHECK:         }
