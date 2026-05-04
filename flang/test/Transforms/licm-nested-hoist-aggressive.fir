// RUN: fir-opt -flang-licm='hoist-from-nested-regions=aggressive' --split-input-file %s | FileCheck %s

// Tests for the "aggressive" nested hoisting mode, which hoists all safe
// invariant operations from nested regions (not just fir.convert).

// Test that pure ops (arith, fir.shape, fir.convert) inside scf.if are all
// hoisted out of the loop in aggressive mode.
// CHECK-LABEL:   func.func @test_aggressive_pure_ops(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>)
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK:         %[[CVT:.*]] = fir.convert %[[ARG0]]
// CHECK:         %[[SHP:.*]] = fir.shape %[[C10]]
// CHECK:         %[[IDX:.*]] = arith.addi %{{.*}}, %[[C5]]
// CHECK:         scf.for
// CHECK:           scf.if
// CHECK-NOT:         fir.convert
// CHECK-NOT:         fir.shape
// CHECK-NOT:         arith.addi
// CHECK:             memref.store %{{.*}}, %[[CVT]][%[[IDX]]]
func.func @test_aggressive_pure_ops(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    %cond = arith.cmpi slt, %i, %c5 : index
    scf.if %cond {
      %cvt = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
      %shp = fir.shape %c10 : (index) -> !fir.shape<1>
      %idx = arith.addi %c3, %c5 : index
      memref.store %cst, %cvt[%idx] : memref<10xf32>
    }
  }
  return
}

// -----
// Test that an invariant fir.load and its dependent fir.convert chain inside
// scf.if are hoisted in aggressive mode, when the loaded memory is not
// modified inside the loop.
// CHECK-LABEL:   func.func @test_aggressive_invariant_load(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"})
// CHECK:         %[[DECL:.*]] = fir.declare %[[ARG1]]
// CHECK:         %[[CVT:.*]] = fir.convert %[[ARG0]]
// CHECK:         %[[LOAD:.*]] = fir.load %[[DECL]]
// CHECK:         %[[IDX:.*]] = fir.convert %[[LOAD]] : (i32) -> index
// CHECK:         scf.for
// CHECK:           scf.if
// CHECK-NOT:         fir.convert
// CHECK-NOT:         fir.load
// CHECK:             memref.store %{{.*}}, %[[CVT]][%[[IDX]]]
func.func @test_aggressive_invariant_load(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  scf.for %i = %c0 to %c10 step %c1 {
    %cond = arith.cmpi slt, %i, %c5 : index
    scf.if %cond {
      %cvt = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
      %val = fir.load %1 : !fir.ref<i32>
      %idx = fir.convert %val : (i32) -> index
      memref.store %cst, %cvt[%idx] : memref<10xf32>
    }
  }
  return
}

// -----
// Test that a fir.load inside scf.if is NOT hoisted even in aggressive mode
// when the loaded memory is modified inside the loop.
// CHECK-LABEL:   func.func @test_aggressive_modified_load_not_hoisted(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32>)
// CHECK:         %[[CVT:.*]] = fir.convert %[[ARG0]]
// CHECK:         scf.for
// CHECK:           scf.if
// CHECK:             fir.load %[[ARG1]]
// CHECK:             fir.convert
// CHECK:             memref.store
// CHECK:           fir.store {{.*}} to %[[ARG1]]
func.func @test_aggressive_modified_load_not_hoisted(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    %cond = arith.cmpi slt, %i, %c5 : index
    scf.if %cond {
      %cvt = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
      %val = fir.load %arg1 : !fir.ref<i32>
      %idx = fir.convert %val : (i32) -> index
      memref.store %cst, %cvt[%idx] : memref<10xf32>
    }
    fir.store %c1_i32 to %arg1 : !fir.ref<i32>
  }
  return
}

// -----
// Test that aggressive mode hoists pure ops from deeply nested scf.if regions
// and through user-propagation chains.
// CHECK-LABEL:   func.func @test_aggressive_deeply_nested(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>)
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK:         %[[CVT:.*]] = fir.convert %[[ARG0]]
// CHECK:         %[[IDX:.*]] = arith.addi %[[C3]], %[[C5]]
// CHECK:         %[[SHP:.*]] = fir.shape %{{.*}}
// CHECK:         scf.for
// CHECK:           scf.if
// CHECK:             scf.if
// CHECK-NOT:           fir.convert
// CHECK-NOT:           arith.addi
// CHECK-NOT:           fir.shape
// CHECK:               memref.store
func.func @test_aggressive_deeply_nested(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    %cond1 = arith.cmpi slt, %i, %c5 : index
    scf.if %cond1 {
      %cond2 = arith.cmpi slt, %i, %c3 : index
      scf.if %cond2 {
        %cvt = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
        %idx = arith.addi %c3, %c5 : index
        %shp = fir.shape %c10 : (index) -> !fir.shape<1>
        memref.store %cst, %cvt[%idx] : memref<10xf32>
      }
    }
  }
  return
}

// -----
// Test that a fir.load of an OPTIONAL variable inside scf.if is NOT hoisted
// even in aggressive mode, because it is conditionally executed and
// isNonOptionalScalar fails for optional variables. The trip count shortcut
// (loop runs >= 1 iteration) cannot be used for conditionally-executed loads
// since the condition might never be true.
// CHECK-LABEL:   func.func @test_aggressive_conditional_optional_load(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32>)
// CHECK:         %[[DECL:.*]] = fir.declare %[[ARG1]]
// CHECK:         %[[CVT:.*]] = fir.convert %[[ARG0]]
// CHECK:         scf.for
// CHECK:           scf.if
// The fir.load must stay inside scf.if (conditionally executed + optional):
// CHECK:             fir.load %[[DECL]]
// CHECK:             fir.convert
// CHECK:             memref.store
func.func @test_aggressive_conditional_optional_load(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  scf.for %i = %c0 to %c10 step %c1 {
    %cond = arith.cmpi slt, %i, %c5 : index
    scf.if %cond {
      %cvt = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
      %val = fir.load %1 : !fir.ref<i32>
      %idx = fir.convert %val : (i32) -> index
      memref.store %cst, %cvt[%idx] : memref<10xf32>
    }
  }
  return
}

// -----
// Test that in cheap mode (default), only fir.convert is hoisted from nested
// regions, while other pure ops (arith.addi) remain inside. In aggressive mode,
// both would be hoisted.
// This test uses the default (cheap) mode as a cross-check.
// RUN: fir-opt -flang-licm %s --split-input-file 2>&1
// CHECK-LABEL:   func.func @test_cheap_vs_aggressive(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>)
// CHECK:         %[[CVT:.*]] = fir.convert %[[ARG0]]
// CHECK:         scf.for
// CHECK:           scf.if
// In aggressive mode, arith.addi is also hoisted:
// CHECK-NOT:         arith.addi
// CHECK:             memref.store %{{.*}}, %[[CVT]]
func.func @test_cheap_vs_aggressive(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    %cond = arith.cmpi slt, %i, %c5 : index
    scf.if %cond {
      %cvt = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
      %idx = arith.addi %c3, %c5 : index
      memref.store %cst, %cvt[%idx] : memref<10xf32>
    }
  }
  return
}
