// RUN: fir-opt %s --hlfir-expression-simplification | FileCheck %s

// Test removal of trim() calls.

//  logical function test_char_cmp(x, y) result(cmp)
//    character(*) :: x, y
//    cmp = trim(x) == trim(y)
//  end function

func.func @_QPtest_char_cmp(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"},
                            %arg1: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.logical<4> {bindc_name = "cmp", uniq_name = "_QFtest_char_cmpEcmp"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtest_char_cmpEcmp"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFtest_char_cmpEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %5:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %6:2 = hlfir.declare %5#0 typeparams %5#1 dummy_scope %0 {uniq_name = "_QFtest_char_cmpEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %7 = hlfir.char_trim %4#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %8 = hlfir.char_trim %6#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %9 = hlfir.cmpchar eq %7 %8 : (!hlfir.expr<!fir.char<1,?>>, !hlfir.expr<!fir.char<1,?>>) -> i1
  %10 = fir.convert %9 : (i1) -> !fir.logical<4>
  hlfir.assign %10 to %2#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  hlfir.destroy %8 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %7 : !hlfir.expr<!fir.char<1,?>>
  %11 = fir.load %2#0 : !fir.ref<!fir.logical<4>>
  return %11 : !fir.logical<4>
}

// CHECK-LABEL: func.func @_QPtest_char_cmp(
// CHECK-SAME:        %[[ARG_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "x"},
// CHECK-SAME:        %[[ARG_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
// CHECK:         %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.logical<4> {bindc_name = "cmp", uniq_name = "_QFtest_char_cmpEcmp"}
// CHECK:         %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_char_cmpEcmp"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:         %[[VAL_3:.*]]:2 = fir.unboxchar %[[ARG_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:         %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_char_cmpEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:         %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:         %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFtest_char_cmpEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:         %[[VAL_9:.*]] = hlfir.cmpchar eq %[[VAL_5]]#0 %[[VAL_8]]#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> i1
// CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i1) -> !fir.logical<4>
// CHECK:         hlfir.assign %[[VAL_10]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:         %[[VAL_11:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
// CHECK:         return %[[VAL_11]] : !fir.logical<4>
// CHECK:       }

// Check that trim() is not removed when its result is stored.

//  logical function test_char_cmp2(x, y) result(res)
//    character(*) :: x, y
//    character(:), allocatable :: tx
//
//    tx = trim(x)
//    res = tx == y
//  end function

func.func @_QPtest_char_cmp2(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.logical<4> {bindc_name = "res", uniq_name = "_QFtest_char_cmp2Eres"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtest_char_cmp2Eres"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %3 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "tx", uniq_name = "_QFtest_char_cmp2Etx"}
  %4 = fir.zero_bits !fir.heap<!fir.char<1,?>>
  %c0 = arith.constant 0 : index
  %5 = fir.embox %4 typeparams %c0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  fir.store %5 to %3 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %6:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_char_cmp2Etx"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
  %7:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %8:2 = hlfir.declare %7#0 typeparams %7#1 dummy_scope %0 {uniq_name = "_QFtest_char_cmp2Ex"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %9:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %10:2 = hlfir.declare %9#0 typeparams %9#1 dummy_scope %0 {uniq_name = "_QFtest_char_cmp2Ey"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %11 = hlfir.char_trim %8#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %11 to %6#0 realloc : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  hlfir.destroy %11 : !hlfir.expr<!fir.char<1,?>>
  %12 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %13 = fir.box_addr %12 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %14 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %15 = fir.box_elesize %14 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %16 = fir.emboxchar %13, %15 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  %17 = hlfir.cmpchar eq %16 %10#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> i1
  %18 = fir.convert %17 : (i1) -> !fir.logical<4>
  hlfir.assign %18 to %2#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  %19 = fir.load %2#0 : !fir.ref<!fir.logical<4>>
  %20 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %21 = fir.box_addr %20 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %22 = fir.convert %21 : (!fir.heap<!fir.char<1,?>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %23 = arith.cmpi ne, %22, %c0_i64 : i64
  fir.if %23 {
    %24 = fir.load %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    %25 = fir.box_addr %24 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
    fir.freemem %25 : !fir.heap<!fir.char<1,?>>
    %26 = fir.zero_bits !fir.heap<!fir.char<1,?>>
    %c0_0 = arith.constant 0 : index
    %27 = fir.embox %26 typeparams %c0_0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
    fir.store %27 to %6#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  }
  return %19 : !fir.logical<4>
}

// CHECK-LABEL: func.func @_QPtest_char_cmp2(
// CHECK:         hlfir.char_trim
