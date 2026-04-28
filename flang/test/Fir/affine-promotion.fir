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
// CHECK:    affine.for %[[VAL_11:.*]] = 1 to 101 {
// CHECK:      %[[VAL_12:.*]] = affine.apply #{{.*}}(%[[VAL_11]])
// CHECK:      %[[VAL_13:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:      %[[VAL_14:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:      %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:      affine.store %[[VAL_15]], %[[VAL_10]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:    }
// CHECK:    %[[VAL_16:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:    affine.for %[[VAL_17:.*]] = 1 to 101 {
// CHECK:      %[[VAL_18:.*]] = affine.apply #{{.*}}(%[[VAL_17]])
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
// Loops containing fir.if are not promoted (if-promotion deferred to a later patch).
// The entire nest stays as fir.do_loop.
// CHECK: func @loop_with_if(%[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_1:.*]]: f32) {
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:   %[[C100:.*]] = arith.constant 100 : index
// CHECK:   %[[DIMS:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
// CHECK:   fir.do_loop %[[I:.*]] = %[[C1]] to %[[C100]] step %[[C1]] {
// CHECK:     fir.do_loop %[[J:.*]] = %[[C1]] to %[[C100]] step %[[C1]] {
// CHECK:       fir.do_loop %[[K:.*]] = %[[C1]] to %[[C100]] step %[[C1]] {
// CHECK:         %[[IM2:.*]] = arith.subi %[[I]], %[[C2]] : index
// CHECK:         %[[COND:.*]] = arith.cmpi sgt, %[[IM2]], %[[C0]] : index
// CHECK:         fir.if %[[COND]] {
// CHECK:           fir.store %[[VAL_1]] to %{{.*}} : !fir.ref<f32>
// CHECK:         }
// CHECK:         fir.store %[[VAL_1]] to %{{.*}} : !fir.ref<f32>
// CHECK:         fir.store %[[VAL_1]] to %{{.*}} : !fir.ref<f32>
// CHECK:       }
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
// First loop promoted — memref<100xf32> (static shape):
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<100xf32>>) -> memref<100xf32>
// CHECK:           %[[VAL_7:.*]] = affine.for %[[VAL_8:.*]] = 1 to 101 iter_args(%[[VAL_9:.*]] = %[[VAL_1]]) -> (f32) {
// CHECK:             %[[VAL_10:.*]] = affine.apply #{{.*}}(%[[VAL_8]])
// CHECK:             %[[VAL_11:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<100xf32>
// CHECK:             %[[VAL_12:.*]] = arith.addf %[[VAL_9]], %[[VAL_11]] fastmath<contract> : f32
// CHECK:             affine.yield %[[VAL_12]] : f32
// CHECK:           }
// Middle loop stays as fir.do_loop (non-promotable: fir.convert reinterprets pointer):
// CHECK:           %[[VAL_13:.*]]:2 = fir.do_loop %[[VAL_14:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_0]] iter_args(%[[VAL_15:.*]] = %[[VAL_7]]) -> (index, f32) {
// CHECK:             %[[VAL_16:.*]] = fir.array_coor %[[ARG1]](%[[VAL_4]]) %[[VAL_0]], %[[VAL_14]] : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<f32>) -> !fir.ref<!fir.array<100xf32>>
// Inner loop also stays as fir.do_loop:
// CHECK:             %[[VAL_19:.*]] = fir.do_loop %[[VAL_20:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_0]] iter_args(%[[VAL_21:.*]] = %[[VAL_15]]) -> (f32) {
// CHECK:               %[[VAL_22:.*]] = fir.array_coor %[[VAL_17]](%[[VAL_3]]) %[[VAL_20]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<f32>
// CHECK:               %[[VAL_24:.*]] = arith.addf %[[VAL_21]], %[[VAL_23]] fastmath<contract> : f32
// CHECK:               fir.result %[[VAL_24]] : f32
// CHECK:             }
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_14]], %[[VAL_0]] overflow<nsw> : index
// CHECK:             fir.result %[[VAL_25]], %[[VAL_19]] : index, f32
// CHECK:           }
// Last loop promoted — dual reduction:
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[ARG2]] : (!fir.ref<!fir.array<100xf32>>) -> memref<100xf32>
// CHECK:           %[[VAL_27:.*]]:2 = affine.for %[[VAL_28:.*]] = 1 to 101 iter_args(%[[VAL_29:.*]] = %[[VAL_30:.*]]#1, %[[VAL_31:.*]] = %[[VAL_1]]) -> (f32, f32) {
// CHECK:             %[[VAL_32:.*]] = affine.apply #{{.*}}(%[[VAL_28]])
// CHECK:             %[[VAL_33:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_32]]] : memref<100xf32>
// CHECK:             %[[VAL_34:.*]] = arith.addf %[[VAL_29]], %[[VAL_33]] fastmath<contract> : f32
// CHECK:             %[[VAL_35:.*]] = affine.load %[[VAL_26]]{{\[}}%[[VAL_32]]] : memref<100xf32>
// CHECK:             %[[VAL_36:.*]] = arith.addf %[[VAL_31]], %[[VAL_35]] fastmath<contract> : f32
// CHECK:             affine.yield %[[VAL_34]], %[[VAL_36]] : f32, f32
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = arith.addf %[[VAL_38:.*]]#0, %[[VAL_38]]#1 fastmath<contract> : f32
// CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_40:.*]]#0 : (index) -> i32
// CHECK:           fir.store %[[VAL_39]] to %[[VAL_5]] : !fir.ref<i32>
// CHECK:           return %[[VAL_37]] : f32
// CHECK:         }


// -----

// Test: Simple matrix multiplication C(j,i) += A(k,i) * B(j,k)
// Triple-nested loop with all three arrays using 2D indexing.
// Fortran: do i=1,N; do j=1,N; do k=1,N; C(j,i) = C(j,i) + A(k,i)*B(j,k); end do; end do; end do
func.func @matmul(%a: !fir.ref<!fir.array<100x100xf32>>, %b: !fir.ref<!fir.array<100x100xf32>>, %c: !fir.ref<!fir.array<100x100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %shp = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
  fir.do_loop %i = %c1 to %c100 step %c1 {
    fir.do_loop %j = %c1 to %c100 step %c1 {
      fir.do_loop %k = %c1 to %c100 step %c1 {
        %k32 = fir.convert %k : (index) -> i32
        %k64 = fir.convert %k32 : (i32) -> i64
        %i32 = fir.convert %i : (index) -> i32
        %i64 = fir.convert %i32 : (i32) -> i64
        %j32 = fir.convert %j : (index) -> i32
        %j64 = fir.convert %j32 : (i32) -> i64
        %a_idx = fir.array_coor %a(%shp) %k64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
        %a_val = fir.load %a_idx : !fir.ref<f32>
        %b_idx = fir.array_coor %b(%shp) %j64, %k64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
        %b_val = fir.load %b_idx : !fir.ref<f32>
        %mul = arith.mulf %a_val, %b_val fastmath<contract> : f32
        %c_idx = fir.array_coor %c(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
        %c_val = fir.load %c_idx : !fir.ref<f32>
        %sum = arith.addf %c_val, %mul fastmath<contract> : f32
        fir.store %sum to %c_idx : !fir.ref<f32>
      }
    }
  }
  return
}

// CHECK-LABEL: func @matmul(
// CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<100x100xf32>>, %[[B:.*]]: !fir.ref<!fir.array<100x100xf32>>, %[[C:.*]]: !fir.ref<!fir.array<100x100xf32>>)
// CHECK-DAG:   %[[C100:.*]] = arith.constant 100 : index
// CHECK:       %[[SHP:.*]] = fir.shape %[[C100]], %[[C100]]
// CHECK:       %[[AM:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
// CHECK:       %[[BM:.*]] = fir.convert %[[B]] : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
// CHECK:       %[[CM:.*]] = fir.convert %[[C]] : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
// CHECK:       affine.for %[[I:.*]] = 1 to 101 {
// CHECK:         %[[IDXI:.*]] = affine.apply #{{.*}}(%[[I]])
// CHECK:         affine.for %[[J:.*]] = 1 to 101 {
// CHECK:           %[[IDXJ:.*]] = affine.apply #{{.*}}(%[[J]])
// CHECK:           %[[CVAL:.*]] = affine.load %[[CM]][%[[IDXI]], %[[IDXJ]]] : memref<100x100xf32>
// CHECK:           affine.for %[[K:.*]] = 1 to 101 {
// CHECK:             %[[IDXK:.*]] = affine.apply #{{.*}}(%[[K]])
// CHECK:             %[[AVAL:.*]] = affine.load %[[AM]][%[[IDXI]], %[[IDXK]]] : memref<100x100xf32>
// CHECK:             %[[BVAL:.*]] = affine.load %[[BM]][%[[IDXK]], %[[IDXJ]]] : memref<100x100xf32>
// CHECK:             %[[MUL:.*]] = arith.mulf %[[AVAL]], %[[BVAL]] fastmath<contract> : f32
// CHECK:             %[[SUM:.*]] = arith.addf %[[CVAL]], %[[MUL]] fastmath<contract> : f32
// CHECK:             affine.store %[[SUM]], %[[CM]][%[[IDXI]], %[[IDXJ]]] : memref<100x100xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       return

// -----

// Test: Triangular loop promoted to affine.for with dimension-based upper bound.
// Inner bound j=1..i uses affine_map<(d0) -> (d0 + 1)>(%outer_iv) — a dimension,
// not a symbol, because the outer IV is a loop induction variable.
func.func @triangular_promotion(%a: !fir.ref<!fir.array<100x100xf32>>, %b: !fir.ref<!fir.array<100x100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %shp = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
  fir.do_loop %i = %c1 to %c100 step %c1 {
    fir.do_loop %j = %c1 to %i step %c1 {
      %j32 = fir.convert %j : (index) -> i32
      %j64 = fir.convert %j32 : (i32) -> i64
      %i32 = fir.convert %i : (index) -> i32
      %i64 = fir.convert %i32 : (i32) -> i64
      %a_idx = fir.array_coor %a(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      %a_val = fir.load %a_idx : !fir.ref<f32>
      %b_idx = fir.array_coor %b(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      fir.store %a_val to %b_idx : !fir.ref<f32>
    }
  }
  return
}

// CHECK-LABEL: func @triangular_promotion(
// CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<100x100xf32>>, %[[B:.*]]: !fir.ref<!fir.array<100x100xf32>>)
// CHECK-DAG:   %[[AM:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
// CHECK-DAG:   %[[BM:.*]] = fir.convert %[[B]] : (!fir.ref<!fir.array<100x100xf32>>) -> memref<100x100xf32>
// Outer loop: constant bounds (folded by AffineIndexBuilder)
// CHECK:       affine.for %[[I:.*]] = 1 to 101 {
// Inner loop: upper bound is affine_map<(d0) -> (d0 + 1)>(%outer_iv) — dimension
// CHECK:         affine.for %[[J:.*]] = 1 to #{{.*}}(%[[I]]) {
// CHECK:           affine.load %[[AM]][%{{.*}}, %{{.*}}] : memref<100x100xf32>
// CHECK:           affine.store %{{.*}}, %[[BM]][%{{.*}}, %{{.*}}] : memref<100x100xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       return

// -----

// Test 6: Loop over an array whose element type is not a valid MLIR
// MemRef element type (e.g. `!fir.logical<n>`, `!fir.char<...>`,
// derived `!fir.type<...>`) must NOT be promoted. Promotion would
// otherwise build a `MemRefType` with an unsupported element type and
// trip the verifier in `mlir::MemRefType::get`.
func.func @rejects_invalid_memref_element(%arg0: !fir.ref<!fir.array<100x!fir.logical<4>>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %true = arith.constant true
  %tlog = fir.convert %true : (i1) -> !fir.logical<4>
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  fir.do_loop %i = %c1 to %c100 step %c1 {
    %1 = fir.array_coor %arg0(%0) %i : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
    fir.store %tlog to %1 : !fir.ref<!fir.logical<4>>
  }
  return
}

// CHECK-LABEL: func @rejects_invalid_memref_element(
// Element type !fir.logical<4> is not accepted by MemRefType, so the
// loop must stay as fir.do_loop and never become affine.for:
// CHECK-NOT:   affine.for
// CHECK:       fir.do_loop
// CHECK-NOT:   affine.for
// CHECK:       return
