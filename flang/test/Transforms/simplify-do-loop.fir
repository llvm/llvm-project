// Test simplify-fir-loop pass
// Canonicalizes fir.do_loop nests by removing shadow iter_args, forwarding
// IV loads, and emitting final IV value stores after the outermost loop.

// RUN: fir-opt --split-input-file --simplify-fir-loop -cse %s | FileCheck %s

// -----

// Test 1: Simple 1D loop — iter_arg removed, IV loads forwarded, final store
// emitted after the loop.
// Fortran: do i = 1, 100; b(i) = a(i); end do
func.func @simple_1d(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %2 = fir.declare %1 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.convert %c1 : (index) -> i32
  %4 = fir.do_loop %arg2 = %c1 to %c100 step %c1 iter_args(%arg3 = %3) -> (i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %5 = fir.load %2 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> i64
    %7 = fir.array_coor %arg0(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    %9 = fir.array_coor %arg1(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    fir.store %8 to %9 : !fir.ref<f32>
    %10 = fir.load %2 : !fir.ref<i32>
    %11 = arith.addi %10, %3 overflow<nsw> : i32
    fir.result %11 : i32
  }
  fir.store %4 to %2 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @simple_1d(
// CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<100xf32>>, %[[B:.*]]: !fir.ref<!fir.array<100xf32>>)
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C100:.*]] = arith.constant 100 : index
// CHECK:       %[[SHAPE:.*]] = fir.shape %[[C100]]
// CHECK:       %[[IV_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i"
// CHECK:       %[[IV_DECL:.*]] = fir.declare %[[IV_ALLOCA]]
// iter_args must be removed:
// CHECK:       fir.do_loop %[[I:.*]] = %[[C1]] to %[[C100]] step %[[C1]] {
// CHECK-NOT:     iter_args
// IV loads must be forwarded to fir.convert(IV):
// CHECK:         %[[I_I32:.*]] = fir.convert %[[I]] : (index) -> i32
// CHECK:         %[[I_I64:.*]] = fir.convert %[[I_I32]] : (i32) -> i64
// CHECK:         %[[A_IDX:.*]] = fir.array_coor %[[A]](%[[SHAPE]]) %[[I_I64]]
// CHECK:         %[[A_VAL:.*]] = fir.load %[[A_IDX]]
// CHECK:         %[[B_IDX:.*]] = fir.array_coor %[[B]](%[[SHAPE]]) %[[I_I64]]
// CHECK:         fir.store %[[A_VAL]] to %[[B_IDX]]
// CHECK:       }
// Final IV value: lb + max(0, (ub - lb + step) / step) * step
// CHECK:       %[[SUB:.*]] = arith.subi %[[C100]], %[[C1]] : index
// CHECK:       %[[ADD:.*]] = arith.addi %[[SUB]], %[[C1]] : index
// CHECK:       %[[DIV:.*]] = arith.divsi %[[ADD]], %[[C1]] : index
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[CMP:.*]] = arith.cmpi sgt, %[[DIV]], %[[C0]] : index
// CHECK:       %[[TRIP:.*]] = arith.select %[[CMP]], %[[DIV]], %[[C0]] : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[TRIP]], %[[C1]] : index
// CHECK:       %[[FINALIDX:.*]] = arith.addi %[[C1]], %[[MUL]] : index
// CHECK:       %[[FINAL:.*]] = fir.convert %[[FINALIDX]] : (index) -> i32
// CHECK:       fir.store %[[FINAL]] to %[[IV_DECL]]
// CHECK:       return

// -----

// Test 2: Nested 2D loop — both iter_args removed, final stores for i and j.
// Fortran: do i = 1,100; do j = 1,100; c(j,i) = a(j,i) + b(j,i); end do; end do
func.func @nested_2d(%a: !fir.ref<!fir.array<100x100xf32>>, %b: !fir.ref<!fir.array<100x100xf32>>, %c: !fir.ref<!fir.array<100x100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %shp = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
  %ad = fir.declare %a(%shp) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<100x100xf32>>
  %bd = fir.declare %b(%shp) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<100x100xf32>>
  %cd = fir.declare %c(%shp) {uniq_name = "_QFEc"} : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<100x100xf32>>
  %i_alloca = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %i_decl = fir.declare %i_alloca {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %j_alloca = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
  %j_decl = fir.declare %j_alloca {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %init = fir.convert %c1 : (index) -> i32
  %outer = fir.do_loop %i = %c1 to %c100 step %c1 iter_args(%i_arg = %init) -> (i32) {
    fir.store %i_arg to %i_decl : !fir.ref<i32>
    %inner = fir.do_loop %j = %c1 to %c100 step %c1 iter_args(%j_arg = %init) -> (i32) {
      fir.store %j_arg to %j_decl : !fir.ref<i32>
      %jv = fir.load %j_decl : !fir.ref<i32>
      %j64 = fir.convert %jv : (i32) -> i64
      %iv = fir.load %i_decl : !fir.ref<i32>
      %i64 = fir.convert %iv : (i32) -> i64
      %a_idx = fir.array_coor %ad(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      %a_val = fir.load %a_idx : !fir.ref<f32>
      %b_idx = fir.array_coor %bd(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      %b_val = fir.load %b_idx : !fir.ref<f32>
      %sum = arith.addf %a_val, %b_val fastmath<contract> : f32
      %c_idx = fir.array_coor %cd(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      fir.store %sum to %c_idx : !fir.ref<f32>
      %jv2 = fir.load %j_decl : !fir.ref<i32>
      %j_next = arith.addi %jv2, %init overflow<nsw> : i32
      fir.result %j_next : i32
    }
    fir.store %inner to %j_decl : !fir.ref<i32>
    %iv2 = fir.load %i_decl : !fir.ref<i32>
    %i_next = arith.addi %iv2, %init overflow<nsw> : i32
    fir.result %i_next : i32
  }
  fir.store %outer to %i_decl : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @nested_2d(
// Both loops must have no iter_args:
// CHECK:       fir.do_loop %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         fir.do_loop %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       iter_args
// IV loads forwarded — j and i accessed via fir.convert of loop IVs:
// CHECK:           %[[J_I32:.*]] = fir.convert %[[J]] : (index) -> i32
// CHECK:           %[[J_I64:.*]] = fir.convert %[[J_I32]] : (i32) -> i64
// CHECK:           %[[I_I32:.*]] = fir.convert %[[I]] : (index) -> i32
// CHECK:           %[[I_I64:.*]] = fir.convert %[[I_I32]] : (i32) -> i64
// CHECK:         }
// CHECK:       }
// Final stores for both i and j after the outermost loop:
// CHECK:       fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK:       fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK:       return

// -----

// Test 3: Triangular loop — inner bound depends on outer IV.
// Fortran: do i = 1,100; do j = 1,i; c(j,i) = a(j,i) + b(j,i); end do; end do
func.func @triangular(%a: !fir.ref<!fir.array<100x100xf32>>, %b: !fir.ref<!fir.array<100x100xf32>>, %c: !fir.ref<!fir.array<100x100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %shp = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
  %ad = fir.declare %a(%shp) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<100x100xf32>>
  %bd = fir.declare %b(%shp) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<100x100xf32>>
  %cd = fir.declare %c(%shp) {uniq_name = "_QFEc"} : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<100x100xf32>>
  %i_alloca = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %i_decl = fir.declare %i_alloca {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %j_alloca = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
  %j_decl = fir.declare %j_alloca {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %init = fir.convert %c1 : (index) -> i32
  %outer = fir.do_loop %i = %c1 to %c100 step %c1 iter_args(%i_arg = %init) -> (i32) {
    fir.store %i_arg to %i_decl : !fir.ref<i32>
    %i_val = fir.load %i_decl : !fir.ref<i32>
    %i_idx = fir.convert %i_val : (i32) -> index
    %inner = fir.do_loop %j = %c1 to %i_idx step %c1 iter_args(%j_arg = %init) -> (i32) {
      fir.store %j_arg to %j_decl : !fir.ref<i32>
      %jv = fir.load %j_decl : !fir.ref<i32>
      %j64 = fir.convert %jv : (i32) -> i64
      %iv = fir.load %i_decl : !fir.ref<i32>
      %i64 = fir.convert %iv : (i32) -> i64
      %a_idx = fir.array_coor %ad(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      %a_val = fir.load %a_idx : !fir.ref<f32>
      %b_idx = fir.array_coor %bd(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      %b_val = fir.load %b_idx : !fir.ref<f32>
      %sum = arith.addf %a_val, %b_val fastmath<contract> : f32
      %c_idx = fir.array_coor %cd(%shp) %j64, %i64 : (!fir.ref<!fir.array<100x100xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
      fir.store %sum to %c_idx : !fir.ref<f32>
      %jv2 = fir.load %j_decl : !fir.ref<i32>
      %j_next = arith.addi %jv2, %init overflow<nsw> : i32
      fir.result %j_next : i32
    }
    fir.store %inner to %j_decl : !fir.ref<i32>
    %iv2 = fir.load %i_decl : !fir.ref<i32>
    %i_next = arith.addi %iv2, %init overflow<nsw> : i32
    fir.result %i_next : i32
  }
  fir.store %outer to %i_decl : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @triangular(
// Both loops transformed — no iter_args:
// CHECK:       fir.do_loop %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// Inner loop bound uses the outer IV (through fir.convert chain):
// CHECK:         fir.do_loop %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       iter_args
// CHECK:         }
// CHECK:       }
// Final stores for both i and j after outermost loop:
// CHECK:       fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK:       fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK:       return

// -----

// Test 4: Non-unit step — do i = 1, 100, 3
func.func @non_unit_step(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %2 = fir.declare %1 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.convert %c1 : (index) -> i32
  %4 = fir.do_loop %arg2 = %c1 to %c100 step %c3 iter_args(%arg3 = %3) -> (i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %5 = fir.load %2 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> i64
    %7 = fir.array_coor %arg0(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    %9 = fir.array_coor %arg1(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    fir.store %8 to %9 : !fir.ref<f32>
    %10 = fir.convert %c3 : (index) -> i32
    %11 = fir.load %2 : !fir.ref<i32>
    %12 = arith.addi %11, %10 overflow<nsw> : i32
    fir.result %12 : i32
  }
  fir.store %4 to %2 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @non_unit_step(
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C100:.*]] = arith.constant 100 : index
// Loop with step 3 and no iter_args:
// CHECK:       fir.do_loop %[[I:.*]] = %[[C1]] to %[[C100]] step %[[C3]] {
// CHECK-NOT:     iter_args
// CHECK:       }
// Final IV value: lb + trip_count * step = 1 + 34 * 3 = 103
// CHECK:       %[[FINAL_I32:.*]] = fir.convert %{{.*}} : (index) -> i32
// CHECK:       fir.store %[[FINAL_I32]] to %{{.*}} : !fir.ref<i32>
// CHECK:       return

// -----

// Test 5: Rejection — loop with fir.call must NOT be transformed.
func.func @rejection_with_call(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %2 = fir.declare %1 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.convert %c1 : (index) -> i32
  %4 = fir.do_loop %arg2 = %c1 to %c100 step %c1 iter_args(%arg3 = %3) -> (i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %5 = fir.load %2 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> i64
    %7 = fir.array_coor %arg0(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    fir.call @_QPuser_sub(%7) fastmath<contract> : (!fir.ref<f32>) -> ()
    %9 = fir.array_coor %arg1(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    fir.store %8 to %9 : !fir.ref<f32>
    %10 = fir.load %2 : !fir.ref<i32>
    %11 = arith.addi %10, %3 overflow<nsw> : i32
    fir.result %11 : i32
  }
  fir.store %4 to %2 : !fir.ref<i32>
  return
}
func.func private @_QPuser_sub(!fir.ref<f32>)

// CHECK-LABEL: func.func @rejection_with_call(
// Loop must remain UNCHANGED — iter_args still present:
// CHECK:       fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32) {
// CHECK:         fir.call @_QPuser_sub
// CHECK:         fir.result %{{.*}} : i32
// CHECK:       }
// CHECK:       return

// -----

// Test 6: Rejection — iter_arg carries a reduction value (sum), not the IV.
// The pass must NOT transform this loop because the iter_arg is not a shadow
// of the induction variable.
// Fortran: sum = 0.0; do i = 1, 100; sum = sum + a(i); end do
func.func @rejection_reduction(%arg0: !fir.ref<!fir.array<100xf32>>, %sum_ref: !fir.ref<f32>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = fir.do_loop %arg1 = %c1 to %c100 step %c1 iter_args(%sum = %cst) -> (f32) {
    %2 = fir.convert %arg1 : (index) -> i64
    %3 = fir.array_coor %arg0(%0) %2 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    %4 = fir.load %3 : !fir.ref<f32>
    %5 = arith.addf %sum, %4 fastmath<contract> : f32
    fir.result %5 : f32
  }
  fir.store %1 to %sum_ref : !fir.ref<f32>
  return
}

// CHECK-LABEL: func.func @rejection_reduction(
// Loop must remain UNCHANGED — iter_args still present (reduction, not IV):
// CHECK:       fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32) {
// CHECK:         arith.addf
// CHECK:         fir.result %{{.*}} : f32
// CHECK:       }
// CHECK:       fir.store
// CHECK:       return

// -----

// Test 7: Rejection — iter_arg is i32 but init value is NOT fir.convert(lb).
// The init is an arbitrary constant, not derived from the loop lower bound.
func.func @rejection_non_iv_init(%arg0: !fir.ref<!fir.array<100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c42_i32 = arith.constant 42 : i32
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
  %2 = fir.declare %1 {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.do_loop %arg1 = %c1 to %c100 step %c1 iter_args(%x = %c42_i32) -> (i32) {
    fir.store %x to %2 : !fir.ref<i32>
    %4 = fir.load %2 : !fir.ref<i32>
    %c2_i32 = arith.constant 2 : i32
    %5 = arith.addi %4, %c2_i32 overflow<nsw> : i32
    fir.result %5 : i32
  }
  fir.store %3 to %2 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @rejection_non_iv_init(
// Loop must remain UNCHANGED — init is 42, not fir.convert(lb):
// CHECK:       %[[C42:.*]] = arith.constant 42 : i32
// CHECK:       fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[C42]]) -> (i32) {
// CHECK:         fir.result %{{.*}} : i32
// CHECK:       }
// CHECK:       return

// -----

// Test 8: Loop attributes preserved across rewrite.
// `unordered` (independence hint) and `loopAnnotation` (user pragmas such as
// `!dir$ unroll`, `!$omp simd`) must survive removal of the shadow iter_args.
#loop_unroll = #llvm.loop_unroll<disable = false, full = true>
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll>
func.func @preserves_attributes(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %2 = fir.declare %1 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.convert %c1 : (index) -> i32
  %4 = fir.do_loop %arg2 = %c1 to %c100 step %c1 unordered iter_args(%arg3 = %3) -> (i32) attributes {loopAnnotation = #loop_annotation} {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %5 = fir.load %2 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> i64
    %7 = fir.array_coor %arg0(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    %9 = fir.array_coor %arg1(%0) %6 : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    fir.store %8 to %9 : !fir.ref<f32>
    %10 = fir.load %2 : !fir.ref<i32>
    %11 = arith.addi %10, %3 overflow<nsw> : i32
    fir.result %11 : i32
  }
  fir.store %4 to %2 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @preserves_attributes(
// iter_args must be removed but `unordered` and `loopAnnotation` must survive:
// CHECK:       fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered
// CHECK-NOT:     iter_args
// CHECK-SAME:    attributes {loopAnnotation = #loop_annotation}
