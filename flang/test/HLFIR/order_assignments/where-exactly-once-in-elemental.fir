// Test lowering of hlfir.where with hlfir.exactly_once nested inside
// hlfir.elemental in the WHERE mask. This pattern arises from WHERE
// statements with array constructors containing implied-do loops,
// e.g. WHERE([(.NOT.ANY(mask == j), j = 1, SIZE(arr))]) arr = default
// The exactly_once must be inlined into the elemental body during lowering.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @_QMmPtest(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "arr"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "mask"}, %arg2: !fir.ref<f64> {fir.bindc_name = "default"}) {
  %true = arith.constant true
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QMmFtestEarr"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMmFtestEdefault"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  %5:2 = hlfir.declare %arg1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMmFtestEmask"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  hlfir.where {
    %6:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
    %7 = fir.convert %6#1 : (index) -> i64
    %8 = arith.maxsi %7, %c0_i64 : i64
    %9 = fir.convert %8 : (i64) -> index
    %10 = fir.shape %9 : (index) -> !fir.shape<1>
    %11 = fir.convert %c1_i64 : (i64) -> index
    %12 = fir.convert %c1_i64 : (i64) -> index
    %13 = hlfir.elemental %10 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
    ^bb0(%arg3: index):
      %14 = arith.subi %arg3, %c1 : index
      %15 = arith.muli %14, %12 : index
      %16 = arith.addi %11, %15 : index
      %17 = hlfir.exactly_once : !fir.logical<4> {
        %21 = fir.convert %16 : (index) -> i64
        %22 = fir.convert %21 : (i64) -> i32
        %23:3 = fir.box_dims %5#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
        %24 = fir.shape %23#1 : (index) -> !fir.shape<1>
        %25 = hlfir.elemental %24 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
        ^bb0(%arg4: index):
          %27 = hlfir.designate %5#0 (%arg4)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
          %28 = fir.load %27 : !fir.ref<i32>
          %29 = arith.cmpi eq, %28, %22 : i32
          %30 = fir.convert %29 : (i1) -> !fir.logical<4>
          hlfir.yield_element %30 : !fir.logical<4>
        }
        %26 = hlfir.any %25 : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
        hlfir.yield %26 : !fir.logical<4> cleanup {
          hlfir.destroy %25 : !hlfir.expr<?x!fir.logical<4>>
        }
      }
      %18 = fir.convert %17 : (!fir.logical<4>) -> i1
      %19 = arith.xori %18, %true : i1
      %20 = fir.convert %19 : (i1) -> !fir.logical<4>
      hlfir.yield_element %20 : !fir.logical<4>
    }
    hlfir.yield %13 : !hlfir.expr<?x!fir.logical<4>> cleanup {
      hlfir.destroy %13 : !hlfir.expr<?x!fir.logical<4>>
    }
  } do {
    hlfir.region_assign {
      %6 = fir.load %2#0 : !fir.ref<f64>
      hlfir.yield %6 : f64
    } to {
      hlfir.yield %1#0 : !fir.box<!fir.array<?xf64>>
    }
  }
  return
}

// Verify the hlfir.exactly_once is eliminated and its body (hlfir.any
// over an inner hlfir.elemental comparing mask elements) is properly
// inlined into the WHERE loop. The result feeds the NOT mask and a
// conditional assignment.
// CHECK-LABEL:   func.func @_QMmPtest(
// CHECK-NOT:     hlfir.exactly_once
// CHECK:         fir.do_loop
// CHECK:           hlfir.elemental
// CHECK:             arith.cmpi eq
// CHECK:           hlfir.any
// CHECK:           arith.xori
// CHECK:           fir.if
// CHECK:             hlfir.assign
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     hlfir.exactly_once
