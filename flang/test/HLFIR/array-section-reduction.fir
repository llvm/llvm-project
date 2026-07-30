// Test the loop-invariant array-section reduction promotion pass. It rewrites a
// whole-section reduction (a(:,i) = 0; do j; a(:,i) = a(:,i) + ...) to
// accumulate into a constant-shape local temporary and, by default, annotates
// the loop to enable vectorization. Positive cases check the rewrite; negative
// cases check that each precondition blocks promotion.
// RUN: fir-opt --split-input-file --array-section-reduction %s | FileCheck %s
// RUN: fir-opt --split-input-file --array-section-reduction=force-vectorize=false %s | FileCheck %s --check-prefix=NOVEC

// Fortran source of the canonical promotable reduction (@_QPpos):
//   subroutine pos(a, x, xx, i, n)
//     real(8) :: a(:,:), x(:), xx(3)
//     integer :: i, n, j
//     a(:,i) = 0.0
//     do j = 1, n
//       a(:,i) = a(:,i) + xx*x(j)   ! reduction into a loop-invariant section
//     end do
//   end subroutine
// Each case below is this pattern with one change: positives must still promote,
// negatives must be blocked by the noted precondition.

// @_QPpos: all preconditions hold -> promoted.
// The pass hoists a constant-shape temporary %[[T]], folds the init into it,
// redirects the in-loop read-modify-write to it, copies it back after the loop,
// and annotates the loop for vectorization.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos
// CHECK:         %[[T:.*]] = fir.alloca !fir.array<3xf64>
// CHECK:         hlfir.assign %{{.*}} to %[[T]] : f64, !fir.ref<!fir.array<3xf64>>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
// CHECK:         hlfir.designate %[[T]] (%{{.*}}) : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
// CHECK:         hlfir.assign %{{.*}} to %[[T]] : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
// CHECK:         hlfir.assign %[[T]] to %{{.*}} : !fir.ref<!fir.array<3xf64>>, !fir.box<!fir.array<?xf64>>
// With -force-vectorize=false the section is still promoted to the temporary,
// but the loop is not annotated: the vectorization hint is separable.
// NOVEC-LABEL: func.func @_QPpos
// NOVEC:         %[[T:.*]] = fir.alloca !fir.array<3xf64>
// NOVEC:         hlfir.assign %{{.*}} to %[[T]] : f64, !fir.ref<!fir.array<3xf64>>
// NOVEC:         hlfir.assign %[[T]] to %{{.*}} : !fir.ref<!fir.array<3xf64>>, !fir.box<!fir.array<?xf64>>
// NOVEC-NOT:     loopAnnotation
func.func @_QPpos(%arg0: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "x"}, %arg2: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}, %arg4: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFposEa"} : (!fir.box<!fir.array<?x?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>)
  %2:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFposEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFposEj"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFposEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFposEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFposEx"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>)
  %c3 = arith.constant 3 : index
  %7 = fir.shape %c3 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %arg2(%7) dummy_scope %0 arg 3 {uniq_name = "_QFposExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %9:3 = fir.box_dims %1#1, %c0 : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
  %c1_0 = arith.constant 1 : index
  %c0_1 = arith.constant 0 : index
  %10 = arith.subi %9#1, %c1 : index
  %11 = arith.addi %10, %c1_0 : index
  %12 = arith.divsi %11, %c1_0 : index
  %13 = arith.cmpi sgt, %12, %c0_1 : index
  %14 = arith.select %13, %12, %c0_1 : index
  %15 = fir.load %2#0 : !fir.ref<i32>
  %16 = fir.convert %15 : (i32) -> i64
  %17 = fir.shape %14 : (index) -> !fir.shape<1>
  %18 = hlfir.designate %1#0 (%c1:%9#1:%c1_0, %16)  shape %17 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  hlfir.assign %cst to %18 : f64, !fir.box<!fir.array<?xf64>>
  %c1_i32 = arith.constant 1 : i32
  %19 = fir.convert %c1_i32 : (i32) -> index
  %20 = fir.load %5#0 : !fir.ref<i32>
  %21 = fir.convert %20 : (i32) -> index
  %c1_2 = arith.constant 1 : index
  %22 = fir.convert %19 : (index) -> i32
  %23 = fir.do_loop %arg5 = %19 to %21 step %c1_2 iter_args(%arg6 = %22) -> (i32) {
    fir.store %arg6 to %4#0 : !fir.ref<i32>
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %24:3 = fir.box_dims %1#1, %c0_4 : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
    %c1_5 = arith.constant 1 : index
    %c0_6 = arith.constant 0 : index
    %25 = arith.subi %24#1, %c1_3 : index
    %26 = arith.addi %25, %c1_5 : index
    %27 = arith.divsi %26, %c1_5 : index
    %28 = arith.cmpi sgt, %27, %c0_6 : index
    %29 = arith.select %28, %27, %c0_6 : index
    %30 = fir.load %2#0 : !fir.ref<i32>
    %31 = fir.convert %30 : (i32) -> i64
    %32 = fir.shape %29 : (index) -> !fir.shape<1>
    %33 = hlfir.designate %1#0 (%c1_3:%24#1:%c1_5, %31)  shape %32 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
    %34 = fir.load %4#0 : !fir.ref<i32>
    %35 = fir.convert %34 : (i32) -> i64
    %36 = hlfir.designate %6#0 (%35)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %37 = fir.load %36 : !fir.ref<f64>
    %c1_7 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    %c3_9 = arith.constant 3 : index
    %38 = fir.shape %c3_9 : (index) -> !fir.shape<1>
    %39 = hlfir.designate %8#0 (%c1_7:%c3:%c1_8)  shape %38 : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %40 = hlfir.elemental %38 unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%arg7: index):
      %55 = hlfir.designate %39 (%arg7)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %56 = fir.load %55 : !fir.ref<f64>
      %57 = arith.mulf %37, %56 fastmath<contract> : f64
      hlfir.yield_element %57 : f64
    }
    %41 = hlfir.elemental %38 unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%arg7: index):
      %55 = hlfir.designate %33 (%arg7)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
      %56 = hlfir.apply %40, %arg7 : (!hlfir.expr<3xf64>, index) -> f64
      %57 = fir.load %55 : !fir.ref<f64>
      %58 = arith.addf %57, %56 fastmath<contract> : f64
      hlfir.yield_element %58 : f64
    }
    %c1_10 = arith.constant 1 : index
    %c0_11 = arith.constant 0 : index
    %42:3 = fir.box_dims %1#1, %c0_11 : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
    %c1_12 = arith.constant 1 : index
    %c0_13 = arith.constant 0 : index
    %43 = arith.subi %42#1, %c1_10 : index
    %44 = arith.addi %43, %c1_12 : index
    %45 = arith.divsi %44, %c1_12 : index
    %46 = arith.cmpi sgt, %45, %c0_13 : index
    %47 = arith.select %46, %45, %c0_13 : index
    %48 = fir.load %2#0 : !fir.ref<i32>
    %49 = fir.convert %48 : (i32) -> i64
    %50 = fir.shape %47 : (index) -> !fir.shape<1>
    %51 = hlfir.designate %1#0 (%c1_10:%42#1:%c1_12, %49)  shape %50 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
    hlfir.assign %41 to %51 : !hlfir.expr<3xf64>, !fir.box<!fir.array<?xf64>>
    hlfir.destroy %41 : !hlfir.expr<3xf64>
    hlfir.destroy %40 : !hlfir.expr<3xf64>
    %52 = fir.convert %c1_2 : (index) -> i32
    %53 = fir.load %4#0 : !fir.ref<i32>
    %54 = arith.addi %53, %52 overflow<nsw> : i32
    fir.result %54 : i32
  }
  fir.store %23 to %4#0 : !fir.ref<i32>
  return
}
// -----

// @_QPpos_mixed: mixed-precision RMW, minimized to the pattern that triggers
// promotion. a is real(4) but the reduction expression is real(8), so the RHS
// is !hlfir.expr<3xf64> while the section is f32. The promoted temporary must
// take the section element type (f32), not the RHS's (f64), or the redirected
// reads would be mis-typed. Source: real(4) a(3); real(8) xx(3);
//   a = 0.0; do j = 1, n; a = a + xx; end do   (a promotes to real(8)).
// CHECK-LABEL: func.func @_QPpos_mixed
// CHECK:         %[[T:.*]] = fir.alloca !fir.array<3xf32>
// CHECK:         hlfir.assign %{{.*}} to %[[T]] : f32, !fir.ref<!fir.array<3xf32>>
// CHECK:         hlfir.designate %[[T]] (%{{.*}}) : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
// CHECK:         hlfir.assign %{{.*}} to %[[T]] : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf32>>
// CHECK:         hlfir.assign %[[T]] to %{{.*}} : !fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>
func.func @_QPpos_mixed(%arg0: !fir.ref<!fir.array<3xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFpos_mixedEa"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_mixedExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_mixedEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  hlfir.assign %cst to %init : f32, !fir.ref<!fir.array<3xf32>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
      %av = fir.load %ae : !fir.ref<f32>
      %acv = fir.convert %av : (f32) -> f64
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %acv, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf32>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
    // Unrelated dynamic heap free: its !fir.heap value must reach alias() raw
    // (not wrapped in hlfir::Entity, which would assert on the non-entity type).
    %h = fir.allocmem !fir.array<?xf64>, %ub
    fir.freemem %h : !fir.heap<!fir.array<?xf64>>
  }
  return
}
// -----

// a(2:3,i) = a(2:3,i) + ...   (partial slice, not the whole section a(:,i))
// P2: no dominating full-section define matches -> not promoted.
// CHECK-LABEL: func.func @_QPneg_partial
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_partial(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %shp = fir.shape %c2 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_partialEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_partialExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_partialEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_partialEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    // Partial slice a(2:3,i), not the whole section a(:,i) the init defines.
    %rd = hlfir.designate %a#0 (%c2:%c3:%c1, %icl)  shape %shp : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<2xf64>>
    %xxs = hlfir.designate %xx#0 (%c2:%c3:%c1)  shape %shp : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<2xf64>>
    %el = hlfir.elemental %shp unordered : (!fir.shape<1>) -> !hlfir.expr<2xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<2xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<2xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c2:%c3:%c1, %icl)  shape %shp : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<2xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<2xf64>, !fir.ref<!fir.array<2xf64>>
    hlfir.destroy %el : !hlfir.expr<2xf64>
  }
  return
}
// -----

// No  a(:,i) = 0  before the loop.
// P2: no dominating full-section define -> not promoted.
// CHECK-LABEL: func.func @_QPneg_noinit
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_noinit(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_noinitEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_noinitExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_noinitEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_noinitEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  // No a(:,i) = 0 init before the loop.
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}
// -----

// x(j) = a(1,i)   an extra element read of the section, outside the RMW.
// P3 (reads): that read would see stale memory once promoted -> not promoted.
// CHECK-LABEL: func.func @_QPneg_extra_read
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_extra_read(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}, %arg4: !fir.ref<f64> {fir.bindc_name = "res"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_extra_readEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_extra_readExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_extra_readEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_extra_readEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %res:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFneg_extra_readEres"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    // An extra element read a(1,i) outside the RMW: P3-reads rejects, since it
    // would observe stale memory once the reduction accumulates into T.
    %ivl0 = fir.load %i#0 : !fir.ref<i32>
    %icl0 = fir.convert %ivl0 : (i32) -> i64
    %er = hlfir.designate %a#0 (%c1, %icl0)  : (!fir.ref<!fir.array<3x4xf64>>, index, i64) -> !fir.ref<f64>
    %erv = fir.load %er : !fir.ref<f64>
    hlfir.assign %erv to %res#0 : f64, !fir.ref<f64>
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}
// -----

// call sink(a)   inside the loop, may modify or read the section.
// P3 (writes): a call touching the section bypasses the temporary -> not promoted.
// @_QPsink lives in this chunk so the split-out module resolves it.
// CHECK-LABEL: func.func @_QPneg_call
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func private @_QPsink(!fir.ref<!fir.array<3x4xf64>>)
func.func @_QPneg_call(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_callEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_callExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_callEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_callEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    // A call that may modify or read the section bypasses the temporary.
    fir.call @_QPsink(%a#0) : (!fir.ref<!fir.array<3x4xf64>>) -> ()
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}
// -----

// i = i + 1   inside the loop: the section address is not loop-invariant.
// P1: a subscript the section reads is rewritten mid-loop -> not promoted.
// CHECK-LABEL: func.func @_QPneg_index
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_index(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_indexEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_indexExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_indexEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_indexEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    // Reassign the section index i mid-loop, so the section address is not
    // loop-invariant and P1 rejects.
    %oi = fir.load %i#0 : !fir.ref<i32>
    %ni = arith.addi %oi, %c1_i32 : i32
    fir.store %ni to %i#0 : !fir.ref<i32>
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// if (j > 0) a(:,i) = a(:,i) + x(j)   (conditional RMW; init still unconditional)
// Promotes: the dominating init keeps promotion sound even when the RMW is guarded.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos_cond
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
func.func @_QPpos_cond(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFpos_condEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_condExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_condEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFpos_condEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    // Conditional RMW: the unconditional init still dominates, so it promotes.
    %cond = arith.cmpi sgt, %j, %c0 : index
    fir.if %cond {
      %ivl = fir.load %i#0 : !fir.ref<i32>
      %icl = fir.convert %ivl : (i32) -> i64
      %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
      %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
      ^bb0(%k: index):
        %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
        %av = fir.load %ae : !fir.ref<f64>
        %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
        %xv = fir.load %xe : !fir.ref<f64>
        %sum = arith.addf %av, %xv fastmath<contract> : f64
        hlfir.yield_element %sum : f64
      }
      %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
      hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
      hlfir.destroy %el : !hlfir.expr<3xf64>
    }
  }
  return
}

// -----

// a(:,i) = a(:,i) + x(j)   (whole-section reduction with no constant-shape xx,
// so the accumulator extent is a runtime value)
// P4: shape is not a compile-time constant -> detected but not rewritten.
// Kept descriptor-based since the extent must be a runtime value.
// CHECK-LABEL: func.func @_QPneg_dynamic
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_dynamic(%arg0: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "x"}, %arg2: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}, %arg4: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFneg_dynamicEa"} : (!fir.box<!fir.array<?x?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>)
  %2:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_dynamicEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFneg_dynamicEj"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFneg_dynamicEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFneg_dynamicEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFneg_dynamicEx"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %bd0:3 = fir.box_dims %1#1, %c0 : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
  %il0 = fir.load %2#0 : !fir.ref<i32>
  %ic0 = fir.convert %il0 : (i32) -> i64
  %sh0 = fir.shape %bd0#1 : (index) -> !fir.shape<1>
  %initd = hlfir.designate %1#0 (%c1:%bd0#1:%c1, %ic0)  shape %sh0 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  hlfir.assign %cst to %initd : f64, !fir.box<!fir.array<?xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nl = fir.load %5#0 : !fir.ref<i32>
  %ub = fir.convert %nl : (i32) -> index
  %st = arith.constant 1 : index
  %iv0 = fir.convert %lb : (index) -> i32
  %loop = fir.do_loop %arg5 = %lb to %ub step %st iter_args(%arg6 = %iv0) -> (i32) {
    fir.store %arg6 to %4#0 : !fir.ref<i32>
    %c1_l = arith.constant 1 : index
    %c0_l = arith.constant 0 : index
    %bd:3 = fir.box_dims %1#1, %c0_l : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
    %il = fir.load %2#0 : !fir.ref<i32>
    %ic = fir.convert %il : (i32) -> i64
    %sh = fir.shape %bd#1 : (index) -> !fir.shape<1>
    %rd = hlfir.designate %1#0 (%c1_l:%bd#1:%c1_l, %ic)  shape %sh : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
    %jl = fir.load %4#0 : !fir.ref<i32>
    %jc = fir.convert %jl : (i32) -> i64
    %xd = hlfir.designate %6#0 (%jc)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %xv = fir.load %xd : !fir.ref<f64>
    %e = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
    ^bb0(%k: index):
      %r = hlfir.designate %rd (%k)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
      %v = fir.load %r : !fir.ref<f64>
      %s = arith.addf %v, %xv fastmath<contract> : f64
      hlfir.yield_element %s : f64
    }
    %wd = hlfir.designate %1#0 (%c1_l:%bd#1:%c1_l, %ic)  shape %sh : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
    hlfir.assign %e to %wd : !hlfir.expr<?xf64>, !fir.box<!fir.array<?xf64>>
    hlfir.destroy %e : !hlfir.expr<?xf64>
    %ni = fir.convert %st : (index) -> i32
    %nj = fir.load %4#0 : !fir.ref<i32>
    %nn = arith.addi %nj, %ni overflow<nsw> : i32
    fir.result %nn : i32
  }
  fir.store %loop to %4#0 : !fir.ref<i32>
  return
}

// -----

// a(:,i) = 0; do j; a(:,i) = a(:,i) + xx*x(j)   (same source as @_QPpos, but
// lowered to an unstructured CFG with the init and loop in separate blocks)
// Promotes: the init still dominates the loop (cross-block dominance).
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos_cfg
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
func.func @_QPpos_cfg(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFpos_cfgEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_cfgExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_cfgEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFpos_cfgEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  // Init and loop live in separate blocks: cross-block dominance still promotes.
  cf.br ^bb1
^bb1:
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(:,i) = x(j)   (broadcast; the RHS never reads a(:,i), so not a reduction)
// Rejected: the RHS does not read the section, so this is not a read-modify-write.
// CHECK-LABEL: func.func @_QPneg_notrmw
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_notrmw(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_notrmwEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_notrmwExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_notrmwEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_notrmwEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %xe = hlfir.designate %xx#0 (%c1)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
    %xv = fir.load %xe : !fir.ref<f64>
    // Broadcast: the RHS never reads a(:,i), so this is not a read-modify-write.
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      hlfir.yield_element %xv : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(1,i) = 0   a second in-loop write that may alias the section a(:,i).
// P3 (writes): an aliasing in-loop assignment -> not promoted.
// CHECK-LABEL: func.func @_QPneg_alias_write
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_alias_write(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_alias_writeEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_alias_writeExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_alias_writeEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_alias_writeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    // An extra in-loop write a(1,i) aliases the section: P3-writes rejects.
    %ivl0 = fir.load %i#0 : !fir.ref<i32>
    %icl0 = fir.convert %ivl0 : (i32) -> i64
    %aw = hlfir.designate %a#0 (%c1, %icl0)  : (!fir.ref<!fir.array<3x4xf64>>, index, i64) -> !fir.ref<f64>
    hlfir.assign %cst to %aw : f64, !fir.ref<f64>
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// if (n > 0) a(:,i) = 0   (the init is conditional)
// P2: the init does not dominate the loop -> not promoted.
// CHECK-LABEL: func.func @_QPneg_cond_init
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_cond_init(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_cond_initEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_cond_initExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_cond_initEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_cond_initEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  // Conditional init: it does not dominate the loop, so P2 finds no init.
  %n0 = fir.load %n#0 : !fir.ref<i32>
  %n0i = fir.convert %n0 : (i32) -> index
  %cond0 = arith.cmpi sgt, %n0i, %c0 : index
  fir.if %cond0 {
    %iv = fir.load %i#0 : !fir.ref<i32>
    %ic = fir.convert %iv : (i32) -> i64
    %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  }
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(:,i) = 0; do j; a(:,i) = a(:,i) + xx*x(j)   then   b(:,i) = 0; do j; b(:,i) = b(:,i) + xx*x(j)
// Two independent reductions in one routine.  Both promote: two temporaries,
// two annotated loops.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos_multi
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
func.func @_QPpos_multi(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}, %argb: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "b"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFpos_multiEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_multiExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_multiEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFpos_multiEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %b:2 = hlfir.declare %argb(%sh2) dummy_scope %0 arg 5 {uniq_name = "_QFpos_multiEb"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  // First reduction on a(:,i).
  %inita = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %inita : f64, !fir.ref<!fir.array<3xf64>>
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rda = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %ela = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rda (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wra = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %ela to %wra : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %ela : !hlfir.expr<3xf64>
  }
  // Second reduction on the independent array b(:,i).
  %initb = hlfir.designate %b#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %initb : f64, !fir.ref<!fir.array<3xf64>>
  fir.do_loop %jb = %lb to %ub step %c1 {
    %ivlb = fir.load %i#0 : !fir.ref<i32>
    %iclb = fir.convert %ivlb : (i32) -> i64
    %rdb = hlfir.designate %b#0 (%c1:%c3:%c1, %iclb)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %elb = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%kb: index):
      %be = hlfir.designate %rdb (%kb)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %bv = fir.load %be : !fir.ref<f64>
      %xeb = hlfir.designate %xx#0 (%kb)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xvb = fir.load %xeb : !fir.ref<f64>
      %sumb = arith.addf %bv, %xvb fastmath<contract> : f64
      hlfir.yield_element %sumb : f64
    }
    %wrb = hlfir.designate %b#0 (%c1:%c3:%c1, %iclb)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %elb to %wrb : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %elb : !hlfir.expr<3xf64>
  }
  return
}

// -----

// x(i) = a(1,i)   read of the section between the init and the loop.
// P2 (interval): the folded-away init would leave that read seeing stale
// memory -> not promoted.
// CHECK-LABEL: func.func @_QPneg_read_between
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_read_between(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}, %arg4: !fir.ref<f64> {fir.bindc_name = "res"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_read_betweenEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_read_betweenExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_read_betweenEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_read_betweenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %res:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFneg_read_betweenEres"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  // Read a(1,i) between the init and the loop: P2 (interval) rejects.
  %rb_e = hlfir.designate %a#0 (%c1, %ic)  : (!fir.ref<!fir.array<3x4xf64>>, index, i64) -> !fir.ref<f64>
  %rb_v = fir.load %rb_e : !fir.ref<f64>
  hlfir.assign %rb_v to %res#0 : f64, !fir.ref<f64>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// if (c) x(i) = a(1,i)   a *conditional* read of the section between the init
// and the loop.  The read is dominated by the init but does not dominate the
// loop, so a dominates-the-loop interval test would miss it.
// P2 (interval): the folded-away init would leave that conditional read seeing
// stale memory -> not promoted.
// CHECK-LABEL: func.func @_QPneg_cond_read_between
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_cond_read_between(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}, %arg4: !fir.ref<f64> {fir.bindc_name = "res"}, %argc: !fir.ref<!fir.logical<4>> {fir.bindc_name = "c"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_cond_read_betweenEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_cond_read_betweenExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_cond_read_betweenEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_cond_read_betweenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %res:2 = hlfir.declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFneg_cond_read_betweenEres"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c:2 = hlfir.declare %argc dummy_scope %0 arg 6 {uniq_name = "_QFneg_cond_read_betweenEc"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  // Conditional read a(1,i) between the init and the loop: dominated by the
  // init but not dominating the loop, so P2 (interval) must still reject.
  %cl = fir.load %c#0 : !fir.ref<!fir.logical<4>>
  %cb = fir.convert %cl : (!fir.logical<4>) -> i1
  fir.if %cb {
    %rb_e = hlfir.designate %a#0 (%c1, %ic)  : (!fir.ref<!fir.array<3x4xf64>>, index, i64) -> !fir.ref<f64>
    %rb_v = fir.load %rb_e : !fir.ref<f64>
    hlfir.assign %rb_v to %res#0 : f64, !fir.ref<f64>
  }
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(1,i) = 0   write to the section between the init and the loop.
// P2 (interval): the post-loop copy-out would clobber that write -> not promoted.
// CHECK-LABEL: func.func @_QPneg_write_between
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_write_between(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_write_betweenEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_write_betweenExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_write_betweenEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_write_betweenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  // Write a(1,i) between the init and the loop: the post-loop copy-out would
  // clobber it, so P2 (interval) rejects.
  %wb_e = hlfir.designate %a#0 (%c1, %ic)  : (!fir.ref<!fir.array<3x4xf64>>, index, i64) -> !fir.ref<f64>
  hlfir.assign %cst to %wb_e : f64, !fir.ref<f64>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// i = i + 1   the section index is reassigned between the init and the loop, so
// the init defines a(:,i0) but the loop reduces a(:,i0+1).
// P2 (interval): a subscript the section reads is rewritten -> not promoted.
// CHECK-LABEL: func.func @_QPneg_index_between
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_index_between(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFneg_index_betweenEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_index_betweenExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_index_betweenEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_index_betweenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  // Reassign the section index i between the init and the loop: P2 (interval)
  // sees a subscript the section reads rewritten, so it rejects.
  %ib_v = fir.load %i#0 : !fir.ref<i32>
  %ib_n = arith.addi %ib_v, %c1_i32 : i32
  hlfir.assign %ib_n to %i#0 : i32, !fir.ref<i32>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(:,i) = 0; do j; a(:,i) = a(:,i) + xx*x(j)   (twice, on the SAME section a(:,i))
// Both must promote: findDominatingFullDef must pick each loop's own (nearest)
// init, not the prior loop's copy-out.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos_multi_same
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
func.func @_QPpos_multi_same(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFpos_multi_sameEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_multi_sameExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_multi_sameEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFpos_multi_sameEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  // First reduction on a(:,i).
  %inita = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %inita : f64, !fir.ref<!fir.array<3xf64>>
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rda = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %ela = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rda (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wra = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %ela to %wra : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %ela : !hlfir.expr<3xf64>
  }
  // Second reduction on the same a(:,i): each loop must pick its own nearest init.
  %inits = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %inits : f64, !fir.ref<!fir.array<3xf64>>
  fir.do_loop %js = %lb to %ub step %c1 {
    %ivls = fir.load %i#0 : !fir.ref<i32>
    %icls = fir.convert %ivls : (i32) -> i64
    %rds = hlfir.designate %a#0 (%c1:%c3:%c1, %icls)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %els = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%ks: index):
      %aes = hlfir.designate %rds (%ks)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %avs = fir.load %aes : !fir.ref<f64>
      %xes = hlfir.designate %xx#0 (%ks)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xvs = fir.load %xes : !fir.ref<f64>
      %sums = arith.addf %avs, %xvs fastmath<contract> : f64
      hlfir.yield_element %sums : f64
    }
    %wrs = hlfir.designate %a#0 (%c1:%c3:%c1, %icls)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %els to %wrs : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %els : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(:,i) = a(:,i) + xx*x(j)   (canonical reduction, as flang lowers it at O2)
// One designate is both the RMW target and the RHS read base (the shape the
// front end emits after CSE), so the read's users include the RMW assign. Must
// still promote AND annotate -- regression guard for the vectorize-hint gate.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos_shared
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
func.func @_QPpos_shared(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFpos_sharedEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_sharedExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_sharedEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFpos_sharedEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    // One designate is both the RMW read base and the write target (CSE'd).
    %rw = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rw (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xx#0 (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    hlfir.assign %el to %rw : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(:,i) = 0; do k; do j; a(:,i) = a(:,i) + x(j); end; end;  y = a(1,i)
// The reduction is in the inner loop but the init dominates the OUTER loop, so
// (the post-loop read blocks the inner match) it is promoted at the outer loop.
// The vectorize hint must still land on the inner loop, which is what vectorizes.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos_nested
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop
// CHECK-NOT:       loopAnnotation
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
func.func @_QPpos_nested(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}, %argm: !fir.ref<i32> {fir.bindc_name = "m"}, %argr: !fir.ref<f64> {fir.bindc_name = "res"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFpos_nestedEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_nestedExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %i:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_nestedEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFpos_nestedEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %m:2 = hlfir.declare %argm dummy_scope %0 arg 5 {uniq_name = "_QFpos_nestedEm"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %res:2 = hlfir.declare %argr dummy_scope %0 arg 6 {uniq_name = "_QFpos_nestedEres"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %ml = fir.load %m#0 : !fir.ref<i32>
  %ubk = fir.convert %ml : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ubj = fir.convert %nv : (i32) -> index
  // Reduction in the inner loop; init dominates the outer loop, so it promotes
  // at the outer loop while the vectorize hint lands on the inner loop.
  fir.do_loop %k = %lb to %ubk step %c1 {
    fir.do_loop %j = %lb to %ubj step %c1 {
      %ivl = fir.load %i#0 : !fir.ref<i32>
      %icl = fir.convert %ivl : (i32) -> i64
      %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
      %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
      ^bb0(%kk: index):
        %ae = hlfir.designate %rd (%kk)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
        %av = fir.load %ae : !fir.ref<f64>
        %xe = hlfir.designate %xx#0 (%kk)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
        %xv = fir.load %xe : !fir.ref<f64>
        %sum = arith.addf %av, %xv fastmath<contract> : f64
        hlfir.yield_element %sum : f64
      }
      %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
      hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
      hlfir.destroy %el : !hlfir.expr<3xf64>
    }
  }
  // Post-outer-loop read a(1,i) -> res: forces the match at the outer loop.
  %pr_iv = fir.load %i#0 : !fir.ref<i32>
  %pr_ic = fir.convert %pr_iv : (i32) -> i64
  %pr_e = hlfir.designate %a#0 (%c1, %pr_ic)  : (!fir.ref<!fir.array<3x4xf64>>, index, i64) -> !fir.ref<f64>
  %pr_v = fir.load %pr_e : !fir.ref<f64>
  hlfir.assign %pr_v to %res#0 : f64, !fir.ref<f64>
  return
}

// -----

// a(:,i) = a(:,i) + sum(a(:,i))   (the RHS reduces the whole section; sum() is
// inlined to a nested loop, as SimplifyHLFIRIntrinsics does at O2). Promoted,
// but NOT annotated: the nested loop reads the accumulator (a coupled
// recurrence), so the reduction loop is not vectorizable.
// CHECK-LABEL: func.func @_QPnovec_reduce
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK-NOT:     loopAnnotation
func.func @_QPnovec_reduce(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFnovec_reduceEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFnovec_reduceEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    // Inlined sum reads the accumulator (a coupled recurrence): promoted but
    // the reduction loop is not vectorizable, so it is left un-annotated.
    %c0f = arith.constant 0.000000e+00 : f64
    %s = fir.do_loop %ksum = %c1 to %c3 step %c1 iter_args(%acc = %c0f) -> (f64) {
      %rsum = hlfir.designate %rd (%ksum)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %vsum = fir.load %rsum : !fir.ref<f64>
      %nacc = arith.addf %acc, %vsum fastmath<contract> : f64
      fir.result %nacc : f64
    }
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %sum = arith.addf %av, %s fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// a(:,i) = 0; do j; s = 0; do k=1,3; s = s + xx(k); end; a(:,i) = a(:,i) + s
// The inner loop is a small constant-trip scratch loop that reads a local xx,
// not the accumulator, so it is fully unrolled before vectorization. Promoted
// AND annotated: a nested loop only blocks the hint if it reads the accumulator
// or is not fully unrollable.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPpos_scratch_loop
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
func.func @_QPpos_scratch_loop(%arg0: !fir.ref<!fir.array<3x4xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "i"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f64
  %c0f = arith.constant 0.000000e+00 : f64
  %sh2 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh2) dummy_scope %0 arg 1 {uniq_name = "_QFpos_scratch_loopEa"} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<3x4xf64>>, !fir.ref<!fir.array<3x4xf64>>)
  %i:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFpos_scratch_loopEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_scratch_loopEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %xx = fir.alloca !fir.array<3xf64> {bindc_name = "xx", uniq_name = "_QFpos_scratch_loopExx"}
  %xxd:2 = hlfir.declare %xx(%sh) {uniq_name = "_QFpos_scratch_loopExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %iv = fir.load %i#0 : !fir.ref<i32>
  %ic = fir.convert %iv : (i32) -> i64
  %init = hlfir.designate %a#0 (%c1:%c3:%c1, %ic)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %ivl = fir.load %i#0 : !fir.ref<i32>
    %icl = fir.convert %ivl : (i32) -> i64
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    // Small constant-trip scratch loop over the local xx, not the accumulator:
    // fully unrolled before vectorization, so it does not block the hint.
    %s = fir.do_loop %ksum = %c1 to %c3 step %c1 iter_args(%acc = %c0f) -> (f64) {
      %rsum = hlfir.designate %xxd#0 (%ksum)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %vsum = fir.load %rsum : !fir.ref<f64>
      %nacc = arith.addf %acc, %vsum fastmath<contract> : f64
      fir.result %nacc : f64
    }
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %sum = arith.addf %av, %s fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1, %icl)  shape %sh : (!fir.ref<!fir.array<3x4xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// type(pt) :: a(:,:);  do j; a(:,i) = a(:,i)   (derived-type section)
// The element type is not a trivial value type, so the rewrite (stack temporary
// + value copy) does not apply. P0 is checked before the dominating-init lookup,
// so no init is needed to reject.
// P0: non-trivial element type -> not promoted.
// CHECK-LABEL: func.func @_QPneg_derived
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_derived(%arg0: !fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c1_i32 = arith.constant 1 : i32
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFneg_derivedEa"} : (!fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>, !fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>)
  %n:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFneg_derivedEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QTpt{x:f64}>> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>, index) -> !fir.ref<!fir.type<_QTpt{x:f64}>>
      %av = fir.load %ae : !fir.ref<!fir.type<_QTpt{x:f64}>>
      hlfir.yield_element %av : !fir.type<_QTpt{x:f64}>
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>
    hlfir.assign %el to %wr : !hlfir.expr<3x!fir.type<_QTpt{x:f64}>>, !fir.ref<!fir.array<3x!fir.type<_QTpt{x:f64}>>>
    hlfir.destroy %el : !hlfir.expr<3x!fir.type<_QTpt{x:f64}>>
  }
  return
}

// -----

// real(8), volatile :: a(:,:);  do j; a(:,i) = a(:,i)   (volatile section)
// The element type is trivial, so the trivial-type check passes and the
// separate volatility check rejects. Like @_QPneg_derived, P0 is checked before
// the dominating-init lookup, so no init is needed to reject.
// P0: volatile section -> not promoted.
// CHECK-LABEL: func.func @_QPneg_volatile
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_volatile(%arg0: !fir.ref<!fir.array<3xf64>, volatile> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c1_i32 = arith.constant 1 : i32
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFneg_volatileEa"} : (!fir.ref<!fir.array<3xf64>, volatile>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>, volatile>, !fir.ref<!fir.array<3xf64>, volatile>)
  %n:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFneg_volatileEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>, volatile>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>, volatile>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>, volatile>, index) -> !fir.ref<f64, volatile>
      %av = fir.load %ae : !fir.ref<f64, volatile>
      hlfir.yield_element %av : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>, volatile>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>, volatile>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>, volatile>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// !$omp parallel do private(i,j): a(:,i) = 0; do j; a(:,i) = a(:,i) + xx. The
// section index i and the inner counter j are OpenMP privates. P1
// must keep the hlfir.declare address so AliasAnalysis classifies i as a
// private and does not treat the in-loop store to j as rewriting i. Promoted
// and annotated. The omp.private decls stay in this chunk.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[ANNOT:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]]>
// CHECK-LABEL: func.func @_QPompred
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[ANNOT]]}
omp.private {type = private} @_QFompredEi_private_i32 : i32
omp.private {type = private} @_QFompredEj_private_i32 : i32
func.func @_QPompred(%arg0: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i"}, %arg3: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFompredEa"} : (!fir.box<!fir.array<?x?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFompredEi"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg3 dummy_scope %0 {uniq_name = "_QFompredEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c3 = arith.constant 3 : index
  %sh3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg1(%sh3) dummy_scope %0 {uniq_name = "_QFompredExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %jalloc = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFompredEj"}
  %5:2 = hlfir.declare %jalloc {uniq_name = "_QFompredEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c0_i32 = arith.constant 0 : i32
  %nload = fir.load %3#0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %ubi = arith.subi %nload, %c1_i32 : i32
  omp.parallel {
    omp.wsloop private(@_QFompredEi_private_i32 %2#0 -> %argi, @_QFompredEj_private_i32 %5#0 -> %argj : !fir.ref<i32>, !fir.ref<i32>) {
      omp.loop_nest (%iv) : i32 = (%c0_i32) to (%ubi) inclusive step (%c1_i32) {
        %pi:2 = hlfir.declare %argi {uniq_name = "_QFompredEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        %pj:2 = hlfir.declare %argj {uniq_name = "_QFompredEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        hlfir.assign %iv to %pi#0 : i32, !fir.ref<i32>
        %cst = arith.constant 0.000000e+00 : f64
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %bd0:3 = fir.box_dims %1#1, %c0 : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
        %il0 = fir.load %pi#0 : !fir.ref<i32>
        %ic0 = fir.convert %il0 : (i32) -> i64
        %initd = hlfir.designate %1#0 (%c1:%bd0#1:%c1, %ic0)  shape %sh3 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
        hlfir.assign %cst to %initd : f64, !fir.box<!fir.array<?xf64>>
        %lb = arith.constant 1 : index
        %nl = fir.load %3#0 : !fir.ref<i32>
        %ubj = fir.convert %nl : (i32) -> index
        %st = arith.constant 1 : index
        %jiv0 = fir.convert %lb : (index) -> i32
        %jloop = fir.do_loop %arg5 = %lb to %ubj step %st iter_args(%arg6 = %jiv0) -> (i32) {
          fir.store %arg6 to %pj#0 : !fir.ref<i32>
          %c1_l = arith.constant 1 : index
          %c0_l = arith.constant 0 : index
          %bd:3 = fir.box_dims %1#1, %c0_l : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
          %il = fir.load %pi#0 : !fir.ref<i32>
          %ic = fir.convert %il : (i32) -> i64
          %rd = hlfir.designate %1#0 (%c1_l:%bd#1:%c1_l, %ic)  shape %sh3 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
          %xxsl = hlfir.designate %4#0 (%c1_l:%c3:%c1_l)  shape %sh3 : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
          %e = hlfir.elemental %sh3 unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
          ^bb0(%k: index):
            %r = hlfir.designate %rd (%k)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
            %xr = hlfir.designate %xxsl (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
            %v = fir.load %r : !fir.ref<f64>
            %xv = fir.load %xr : !fir.ref<f64>
            %sum = arith.addf %v, %xv fastmath<contract> : f64
            hlfir.yield_element %sum : f64
          }
          %wd = hlfir.designate %1#0 (%c1_l:%bd#1:%c1_l, %ic)  shape %sh3 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
          hlfir.assign %e to %wd : !hlfir.expr<3xf64>, !fir.box<!fir.array<?xf64>>
          hlfir.destroy %e : !hlfir.expr<3xf64>
          %ni = fir.convert %st : (index) -> i32
          %nj = fir.load %pj#0 : !fir.ref<i32>
          %nn = arith.addi %nj, %ni overflow<nsw> : i32
          fir.result %nn : i32
        }
        fir.store %jloop to %pj#0 : !fir.ref<i32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// -----

// @_QPneg_direct_store: an extra direct element write into the section via
// fir.array_coor + fir.store (not an hlfir.assign) inside the loop. P3-writes
// must reject -- promotion redirects the reduction to T while the direct store
// still updates a(:), so the post-loop copy-out would clobber it. Source:
//   real(8) a(3), xx(3); a = 0; do j=1,n; a = a + xx; a(2) = 1.0; end do
// CHECK-LABEL: func.func @_QPneg_direct_store
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_direct_store(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFneg_direct_storeEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_direct_storeExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_direct_storeEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
    %c2 = arith.constant 2 : index
    %elt = fir.array_coor %a#0(%sh) %c2 : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
    %one = arith.constant 1.000000e+00 : f64
    fir.store %one to %elt : !fir.ref<f64>
  }
  return
}

// -----

// @_QPneg_direct_read: a direct element read of the section via fir.array_coor
// + fir.load (not an hlfir.designate) inside the loop. P3-reads must reject --
// after promotion the reduction accumulates into T and the section is not
// updated in the loop, so the direct read would observe stale memory. Source:
//   real(8) a(3), xx(3), res; a = 0; do j=1,n; a = a + xx; res = a(2); end do
// CHECK-LABEL: func.func @_QPneg_direct_read
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_direct_read(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}, %arg3: !fir.ref<f64> {fir.bindc_name = "res"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFneg_direct_readEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_direct_readExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_direct_readEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %res:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFneg_direct_readEres"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
    %c2 = arith.constant 2 : index
    %elt = fir.array_coor %a#0(%sh) %c2 : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
    %v = fir.load %elt : !fir.ref<f64>
    fir.store %v to %res#0 : !fir.ref<f64>
  }
  return
}

// -----

// @_QPpos_reused_read: the section read designator %shared is loop-invariant
// (defined before the loop) and reused after the loop, mirroring CSE'd input
// where one a(:) designator is shared. Promotion must redirect only the in-loop
// use to T; the post-loop use must keep reading the copied-out section, not the
// stale temporary. Source: real(8) a(3), xx(3), res;
//   a = 0; do j=1,n; a = a + xx; end do; a = 99.0; res = a(2)
// CHECK-LABEL: func.func @_QPpos_reused_read
// CHECK:         %[[T:.*]] = fir.alloca !fir.array<3xf64>
// CHECK:         %[[SHARED:.*]] = hlfir.designate %{{.*}}#0 {{.*}} -> !fir.ref<!fir.array<3xf64>>
// CHECK:         hlfir.designate %[[T]] (%{{.*}}) : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
// CHECK:         hlfir.designate %[[SHARED]] (%{{.*}}) : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
func.func @_QPpos_reused_read(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}, %arg3: !fir.ref<f64> {fir.bindc_name = "res"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFpos_reused_readEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_reused_readExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_reused_readEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %res:2 = hlfir.declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFpos_reused_readEres"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %shared = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %shared (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  %c99 = arith.constant 9.900000e+01 : f64
  %w2 = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %c99 to %w2 : f64, !fir.ref<!fir.array<3xf64>>
  %c2 = arith.constant 2 : index
  %e2 = hlfir.designate %shared (%c2)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
  %v = fir.load %e2 : !fir.ref<f64>
  fir.store %v to %res#0 : !fir.ref<f64>
  return
}

// -----

// @_QPpos_novector: the loop already carries a NOVECTOR annotation (vectorize
// disable = true). Promotion still applies, but the vectorize hint must not
// override the user's directive -- the annotation is left unchanged.
// CHECK: #[[NVEC:.*]] = #llvm.loop_vectorize<disable = true>
// CHECK: #[[NOVEC_ANNO:.*]] = #llvm.loop_annotation<vectorize = #[[NVEC]]>
// CHECK-LABEL: func.func @_QPpos_novector
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[NOVEC_ANNO]]}
// NOVEC-LABEL: func.func @_QPpos_novector
// NOVEC:         fir.alloca !fir.array<3xf64>
// NOVEC:         fir.do_loop {{.*}}loopAnnotation
func.func @_QPpos_novector(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFpos_novectorEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_novectorExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_novectorEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 attributes {loopAnnotation = #llvm.loop_annotation<vectorize = #llvm.loop_vectorize<disable = true>>} {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// @_QPpos_preserve: the loop already carries an annotation with an unrelated
// field (mustProgress). Promotion sets only the vectorize-enable field and
// keeps every other field.
// CHECK: #[[VEC:.*]] = #llvm.loop_vectorize<disable = false>
// CHECK: #[[PRES:.*]] = #llvm.loop_annotation<vectorize = #[[VEC]], mustProgress = true>
// CHECK-LABEL: func.func @_QPpos_preserve
// CHECK:         fir.alloca !fir.array<3xf64>
// CHECK:         fir.do_loop {{.*}}attributes {loopAnnotation = #[[PRES]]}
// NOVEC-LABEL: func.func @_QPpos_preserve
// NOVEC:         fir.alloca !fir.array<3xf64>
// NOVEC:         fir.do_loop {{.*}}loopAnnotation
func.func @_QPpos_preserve(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFpos_preserveEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFpos_preserveExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFpos_preserveEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 attributes {loopAnnotation = #llvm.loop_annotation<mustProgress = true>} {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// @_QPneg_fir_copy: a fir.copy reads the whole section (its source) into
// another array inside the loop. fir.copy exposes the read via
// MemoryEffectOpInterface, not hlfir.designate/fir.load, so P3-reads must
// reject it -- P3-writes passes because the copy's write targets the
// non-aliasing destination. Source: real(8) a(3), xx(3), other(3);
//   a = 0; do j=1,n; a = a + xx; other = a; end do
// CHECK-LABEL: func.func @_QPneg_fir_copy
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_fir_copy(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}, %arg3: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "other"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFneg_fir_copyEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_fir_copyExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_fir_copyEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %other:2 = hlfir.declare %arg3(%sh) dummy_scope %0 arg 4 {uniq_name = "_QFneg_fir_copyEother"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
    fir.copy %a#0 to %other#0 : !fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>
  }
  return
}

// -----

// @_QPneg_copy_between: a fir.copy reads the whole section (its source) into
// another array *between* the init and the loop. Promotion folds the init away
// and only copies the reduction back after the loop, so this copy would read
// stale (uninitialized) section memory. P2 (interval) must reject it via the
// generic memory-effects read check, since fir.copy is neither an
// hlfir.designate nor a fir.load. Source: real(8) a(3), xx(3), other(3);
//   a = 0; other = a; do j=1,n; a = a + xx; end do
// CHECK-LABEL: func.func @_QPneg_copy_between
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_copy_between(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}, %arg3: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "other"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFneg_copy_betweenEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_copy_betweenExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_copy_betweenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %other:2 = hlfir.declare %arg3(%sh) dummy_scope %0 arg 4 {uniq_name = "_QFneg_copy_betweenEother"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  // Read the whole section between the init and the loop via fir.copy.
  fir.copy %a#0 to %other#0 : !fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

// -----

// @_QPneg_assign_between: like @_QPneg_copy_between, but the copy other = a is
// an hlfir.assign whose RHS designates the section (not a fir.copy). The LHS
// (other) does not alias the section, so P2's hlfir.assign special case must
// also check the RHS -- the section read between the init and the loop would
// see stale storage once the init is folded into the temporary. Source:
//   real(8) a(3), xx(3), other(3); a = 0; other = a; do j=1,n; a = a + xx; end do
// CHECK-LABEL: func.func @_QPneg_assign_between
// CHECK-NOT:     fir.alloca !fir.array
// CHECK-NOT:     loopAnnotation
func.func @_QPneg_assign_between(%arg0: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "xx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}, %arg3: !fir.ref<!fir.array<3xf64>> {fir.bindc_name = "other"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c3 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%sh) dummy_scope %0 arg 1 {uniq_name = "_QFneg_assign_betweenEa"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %xx:2 = hlfir.declare %arg1(%sh) dummy_scope %0 arg 2 {uniq_name = "_QFneg_assign_betweenExx"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %n:2 = hlfir.declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFneg_assign_betweenEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %other:2 = hlfir.declare %arg3(%sh) dummy_scope %0 arg 4 {uniq_name = "_QFneg_assign_betweenEother"} : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>)
  %cst = arith.constant 0.000000e+00 : f64
  %c1 = arith.constant 1 : index
  %init = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %cst to %init : f64, !fir.ref<!fir.array<3xf64>>
  // Read the whole section between the init and the loop via an hlfir.assign
  // whose RHS designates the section.
  %rb = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
  hlfir.assign %rb to %other#0 : !fir.ref<!fir.array<3xf64>>, !fir.ref<!fir.array<3xf64>>
  %c1_i32 = arith.constant 1 : i32
  %lb = fir.convert %c1_i32 : (i32) -> index
  %nv = fir.load %n#0 : !fir.ref<i32>
  %ub = fir.convert %nv : (i32) -> index
  fir.do_loop %j = %lb to %ub step %c1 {
    %rd = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %xxs = hlfir.designate %xx#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    %el = hlfir.elemental %sh unordered : (!fir.shape<1>) -> !hlfir.expr<3xf64> {
    ^bb0(%k: index):
      %ae = hlfir.designate %rd (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %av = fir.load %ae : !fir.ref<f64>
      %xe = hlfir.designate %xxs (%k)  : (!fir.ref<!fir.array<3xf64>>, index) -> !fir.ref<f64>
      %xv = fir.load %xe : !fir.ref<f64>
      %sum = arith.addf %av, %xv fastmath<contract> : f64
      hlfir.yield_element %sum : f64
    }
    %wr = hlfir.designate %a#0 (%c1:%c3:%c1)  shape %sh : (!fir.ref<!fir.array<3xf64>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf64>>
    hlfir.assign %el to %wr : !hlfir.expr<3xf64>, !fir.ref<!fir.array<3xf64>>
    hlfir.destroy %el : !hlfir.expr<3xf64>
  }
  return
}

