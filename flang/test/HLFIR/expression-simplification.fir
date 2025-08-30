// RUN: fir-opt %s --expression-simplification | FileCheck %s

// Test removal of trim() calls.

//  logical function cmp(x, y)
//    character(*) :: x, y
//    cmp = trim(x) == trim(y)
//  end function

func.func @test_char_cmp(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"},
                         %arg1: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  %1 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca !fir.logical<4> {bindc_name = "cmp", uniq_name = "_QFcmpEcmp"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFcmpEcmp"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %5:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %6:2 = hlfir.declare %5#0 typeparams %5#1 dummy_scope %2 {uniq_name = "_QFcmpEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %7:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %8:2 = hlfir.declare %7#0 typeparams %7#1 dummy_scope %2 {uniq_name = "_QFcmpEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %9 = fir.embox %6#1 typeparams %5#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  %10 = fir.zero_bits !fir.heap<!fir.char<1,?>>
  %c0 = arith.constant 0 : index
  %11 = fir.embox %10 typeparams %c0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  fir.store %11 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %12 = fir.address_of(@_QQclX710b72634dc58109795123e3cbc3b5ed) : !fir.ref<!fir.char<1,65>>
  %c5_i32 = arith.constant 5 : i32
  %13 = fir.convert %1 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  %14 = fir.convert %9 : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  %15 = fir.convert %12 : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
  fir.call @_FortranATrim(%13, %14, %15, %c5_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  %16 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %17 = fir.box_elesize %16 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %18 = fir.box_addr %16 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %19:2 = hlfir.declare %18 typeparams %17 {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
  %true = arith.constant true
  %20 = hlfir.as_expr %19#0 move %true : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
  %21 = fir.embox %8#1 typeparams %7#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  %22 = fir.zero_bits !fir.heap<!fir.char<1,?>>
  %c0_0 = arith.constant 0 : index
  %23 = fir.embox %22 typeparams %c0_0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  fir.store %23 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %24 = fir.address_of(@_QQclX710b72634dc58109795123e3cbc3b5ed) : !fir.ref<!fir.char<1,65>>
  %c5_i32_1 = arith.constant 5 : i32
  %25 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  %26 = fir.convert %21 : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  %27 = fir.convert %24 : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
  fir.call @_FortranATrim(%25, %26, %27, %c5_i32_1) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  %28 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %29 = fir.box_elesize %28 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %30 = fir.box_addr %28 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %31:2 = hlfir.declare %30 typeparams %29 {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
  %true_2 = arith.constant true
  %32 = hlfir.as_expr %31#0 move %true_2 : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
  %33:3 = hlfir.associate %20 typeparams %17 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  %34:3 = hlfir.associate %32 typeparams %29 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  %35 = fir.convert %33#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  %36 = fir.convert %34#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  %37 = fir.convert %17 : (index) -> i64
  %38 = fir.convert %29 : (index) -> i64
  %39 = fir.call @_FortranACharacterCompareScalar1(%35, %36, %37, %38) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
  %c0_i32 = arith.constant 0 : i32
  %40 = arith.cmpi eq, %39, %c0_i32 : i32
  hlfir.end_associate %33#1, %33#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.end_associate %34#1, %34#2 : !fir.ref<!fir.char<1,?>>, i1
  %41 = fir.convert %40 : (i1) -> !fir.logical<4>
  hlfir.assign %41 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  hlfir.destroy %32 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %20 : !hlfir.expr<!fir.char<1,?>>
  %42 = fir.load %4#0 : !fir.ref<!fir.logical<4>>
  return %42 : !fir.logical<4>
}

// CHECK-LABEL: func.func @test_char_cmp(
// CHECK-SAME:        %[[VAL_4:.*]]: !fir.boxchar<1> {fir.bindc_name = "x"},
// CHECK-SAME:        %[[VAL_7:.*]]: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
// CHECK:         %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.logical<4> {bindc_name = "cmp", uniq_name = "_QFcmpEcmp"}
// CHECK:         %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFcmpEcmp"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:         %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_4]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:         %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFcmpEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:         %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_7]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:         %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_0]] {uniq_name = "_QFcmpEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:         %[[VAL_13:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_9]], %[[VAL_11]], %[[VAL_10]], %[[VAL_12]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:         %[[VAL_14:.*]] = arith.constant 0 : i32
// CHECK:         %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i1) -> !fir.logical<4>
// CHECK:         hlfir.assign %[[VAL_16]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:         %[[VAL_17:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
// CHECK:         return %[[VAL_17]] : !fir.logical<4>
// CHECK:       }

// Check that trim() is not removed when its result is stored.

//  logical function eq_use3(x, y) result(res)
//    character(*) :: x, y
//    character(:), allocatable :: tx
//
//    tx = trim(x)
//    res = tx == y
//  end function

func.func @test_char_cmp2(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.alloca !fir.logical<4> {bindc_name = "res", uniq_name = "_QFcmpEres"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFcmpEres"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %4 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "tx", uniq_name = "_QFcmpEtx"}
  %5 = fir.zero_bits !fir.heap<!fir.char<1,?>>
  %c0 = arith.constant 0 : index
  %6 = fir.embox %5 typeparams %c0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  fir.store %6 to %4 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %7:2 = hlfir.declare %4 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFcmpEtx"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
  %8:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %9:2 = hlfir.declare %8#0 typeparams %8#1 dummy_scope %1 {uniq_name = "_QFcmpEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %10:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %11:2 = hlfir.declare %10#0 typeparams %10#1 dummy_scope %1 {uniq_name = "_QFcmpEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %12 = fir.embox %9#1 typeparams %8#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  %13 = fir.zero_bits !fir.heap<!fir.char<1,?>>
  %c0_0 = arith.constant 0 : index
  %14 = fir.embox %13 typeparams %c0_0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  fir.store %14 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %15 = fir.address_of(@_QQclX9c1101346143d1d3654395ce601d7e68) : !fir.ref<!fir.char<1,66>>
  %c5_i32 = arith.constant 5 : i32
  %16 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  %17 = fir.convert %12 : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  %18 = fir.convert %15 : (!fir.ref<!fir.char<1,66>>) -> !fir.ref<i8>
  fir.call @_FortranATrim(%16, %17, %18, %c5_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  %19 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %20 = fir.box_elesize %19 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %21 = fir.box_addr %19 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %22:2 = hlfir.declare %21 typeparams %20 {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
  %true = arith.constant true
  %23 = hlfir.as_expr %22#0 move %true : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %23 to %7#0 realloc : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  hlfir.destroy %23 : !hlfir.expr<!fir.char<1,?>>
  %24 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %25 = fir.box_addr %24 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %26 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %27 = fir.box_elesize %26 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %28 = fir.convert %25 : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<i8>
  %29 = fir.convert %11#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  %30 = fir.convert %27 : (index) -> i64
  %31 = fir.convert %10#1 : (index) -> i64
  %32 = fir.call @_FortranACharacterCompareScalar1(%28, %29, %30, %31) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
  %c0_i32 = arith.constant 0 : i32
  %33 = arith.cmpi eq, %32, %c0_i32 : i32
  %34 = fir.convert %33 : (i1) -> !fir.logical<4>
  hlfir.assign %34 to %3#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  %35 = fir.load %3#0 : !fir.ref<!fir.logical<4>>
  %36 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %37 = fir.box_addr %36 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  %38 = fir.convert %37 : (!fir.heap<!fir.char<1,?>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %39 = arith.cmpi ne, %38, %c0_i64 : i64
  fir.if %39 {
    %40 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    %41 = fir.box_addr %40 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
    fir.freemem %41 : !fir.heap<!fir.char<1,?>>
    %42 = fir.zero_bits !fir.heap<!fir.char<1,?>>
    %c0_1 = arith.constant 0 : index
    %43 = fir.embox %42 typeparams %c0_1 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
    fir.store %43 to %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  }
  return %35 : !fir.logical<4>
}

// CHECK-LABEL: func.func @test_char_cmp2(
// CHECK:         fir.call @_FortranATrim
