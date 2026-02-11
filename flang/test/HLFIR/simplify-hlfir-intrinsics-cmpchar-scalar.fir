// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s


//  function test_eq(x, y)
//    logical :: test_eq
//    character(len=*,kind=1) :: x, y
//    test_eq = x .eq. y
//  end function test_eq
  func.func @_QPtest_eq(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
// CHECK-LABEL:   func.func @_QPtest_eq(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "y"}) -> !fir.logical<4> {
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test_eq", uniq_name = "_QFtest_eqEtest_eq"}
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtest_eqEtest_eq"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]]#0 typeparams %[[VAL_8]]#1 dummy_scope %[[VAL_5]] {uniq_name = "_QFtest_eqEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_10:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]]#0 typeparams %[[VAL_10]]#1 dummy_scope %[[VAL_5]] {uniq_name = "_QFtest_eqEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca !fir.logical<4> {bindc_name = "test_eq", uniq_name = "_QFtest_eqEtest_eq"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFtest_eqEtest_eq"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFtest_eqEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %5:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %6:2 = hlfir.declare %5#0 typeparams %5#1 dummy_scope %0 {uniq_name = "_QFtest_eqEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %7 = hlfir.cmpchar eq %4#0 %6#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> i1
// CHECK:           %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_8]]#1, %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_8]]#1, %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %c1 to %[[VAL_13]] step %c1 iter_args(%[[VAL_16:.*]] = %c0_i8) -> (i8) {
// CHECK:             %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_16]], %c0_i8 : i8
// CHECK:             %[[VAL_18:.*]] = fir.if %[[VAL_17]] -> (i8) {
// CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_9]]#0  substr %[[VAL_15]], %[[VAL_15]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_21:.*]] = fir.extract_value %[[VAL_20]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_22:.*]] = hlfir.designate %[[VAL_11]]#0  substr %[[VAL_15]], %[[VAL_15]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_24:.*]] = fir.extract_value %[[VAL_23]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_25:.*]] = arith.cmpi ult, %[[VAL_21]], %[[VAL_24]] : i8
// CHECK:               %[[VAL_26:.*]] = arith.select %[[VAL_25]], %c-1_i8, %[[VAL_16]] : i8
// CHECK:               %[[VAL_27:.*]] = arith.cmpi ugt, %[[VAL_21]], %[[VAL_24]] : i8
// CHECK:               %[[VAL_28:.*]] = arith.select %[[VAL_27]], %c1_i8, %[[VAL_26]] : i8
// CHECK:               fir.result %[[VAL_28]] : i8
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_16]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_18]] : i8
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_8]]#1, %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_30:.*]] = fir.if %[[VAL_29]] -> (i8) {
// CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_8]]#1, %[[VAL_10]]#1 : index
// CHECK:             %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %c1 to %[[VAL_31]] step %c1 iter_args(%[[VAL_34:.*]] = %[[VAL_14]]) -> (i8) {
// CHECK:               %[[VAL_35:.*]] = arith.cmpi eq, %[[VAL_34]], %c0_i8 : i8
// CHECK:               %[[VAL_36:.*]] = fir.if %[[VAL_35]] -> (i8) {
// CHECK:                 %[[VAL_37:.*]] = arith.addi %[[VAL_10]]#1, %[[VAL_33]] : index
// CHECK:                 %[[VAL_38:.*]] = hlfir.designate %[[VAL_9]]#0  substr %[[VAL_37]], %[[VAL_37]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_40:.*]] = fir.extract_value %[[VAL_39]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_41:.*]] = arith.cmpi ult, %[[VAL_40]], %c32_i8 : i8
// CHECK:                 %[[VAL_42:.*]] = arith.select %[[VAL_41]], %c-1_i8, %[[VAL_34]] : i8
// CHECK:                 %[[VAL_43:.*]] = arith.cmpi ugt, %[[VAL_40]], %c32_i8 : i8
// CHECK:                 %[[VAL_44:.*]] = arith.select %[[VAL_43]], %c1_i8, %[[VAL_42]] : i8
// CHECK:                 fir.result %[[VAL_44]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_34]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_36]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_32]] : i8
// CHECK:           } else {
// CHECK:             %[[VAL_45:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_8]]#1 : index
// CHECK:             %[[VAL_46:.*]] = fir.do_loop %[[VAL_47:.*]] = %c1 to %[[VAL_45]] step %c1 iter_args(%[[VAL_48:.*]] = %[[VAL_14]]) -> (i8) {
// CHECK:               %[[VAL_49:.*]] = arith.cmpi eq, %[[VAL_48]], %c0_i8 : i8
// CHECK:               %[[VAL_50:.*]] = fir.if %[[VAL_49]] -> (i8) {
// CHECK:                 %[[VAL_51:.*]] = arith.addi %[[VAL_8]]#1, %[[VAL_47]] : index
// CHECK:                 %[[VAL_52:.*]] = hlfir.designate %[[VAL_11]]#0  substr %[[VAL_51]], %[[VAL_51]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_53:.*]] = fir.load %[[VAL_52]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_54:.*]] = fir.extract_value %[[VAL_53]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_55:.*]] = arith.cmpi ugt, %[[VAL_54]], %c32_i8 : i8
// CHECK:                 %[[VAL_56:.*]] = arith.select %[[VAL_55]], %c-1_i8, %[[VAL_48]] : i8
// CHECK:                 %[[VAL_57:.*]] = arith.cmpi ult, %[[VAL_54]], %c32_i8 : i8
// CHECK:                 %[[VAL_58:.*]] = arith.select %[[VAL_57]], %c1_i8, %[[VAL_56]] : i8
// CHECK:                 fir.result %[[VAL_58]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_48]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_50]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_46]] : i8
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = arith.cmpi eq, %[[VAL_30]], %c0_i8 : i8
// CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i1) -> !fir.logical<4>
    %8 = fir.convert %7 : (i1) -> !fir.logical<4>
    hlfir.assign %8 to %2#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    %9 = fir.load %2#0 : !fir.ref<!fir.logical<4>>
    return %9 : !fir.logical<4>
  }

//  function test_ne(c1, c2)
//    integer :: test_ne
//    character(len=1,kind=2) :: c1, c2
//    test_ne =  c1 /= c2
//  end function test_ne
  func.func @_QPtest_ne(%arg0: !fir.boxchar<2> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<2> {fir.bindc_name = "c2"}) -> i32 {
// CHECK-LABEL:   func.func @_QPtest_ne(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<2> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<2> {fir.bindc_name = "c2"}) -> i32 {
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %c1 dummy_scope %[[VAL_7]] {uniq_name = "_QFtest_neEc1"} : (!fir.ref<!fir.char<2>>, index, !fir.dscope) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
// CHECK:           %[[VAL_11:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] typeparams %c1 dummy_scope %[[VAL_7]] {uniq_name = "_QFtest_neEc2"} : (!fir.ref<!fir.char<2>>, index, !fir.dscope) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
// CHECK:           %[[VAL_14:.*]] = fir.alloca i32 {bindc_name = "test_ne", uniq_name = "_QFtest_neEtest_ne"}
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = "_QFtest_neEtest_ne"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %2 = fir.convert %1#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.char<2>>
    %c1 = arith.constant 1 : index
    %3:2 = hlfir.declare %2 typeparams %c1 dummy_scope %0 {uniq_name = "_QFtest_neEc1"} : (!fir.ref<!fir.char<2>>, index, !fir.dscope) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
    %4:2 = fir.unboxchar %arg1 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %5 = fir.convert %4#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.char<2>>
    %c1_0 = arith.constant 1 : index
    %6:2 = hlfir.declare %5 typeparams %c1_0 dummy_scope %0 {uniq_name = "_QFtest_neEc2"} : (!fir.ref<!fir.char<2>>, index, !fir.dscope) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
    %7 = fir.alloca i32 {bindc_name = "test_ne", uniq_name = "_QFtest_neEtest_ne"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFtest_neEtest_ne"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %9 = hlfir.cmpchar ne %3#0 %6#0 : (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>) -> i1
// CHECK:           %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %c1 to %c1 step %c1 iter_args(%[[VAL_18:.*]] = %c0_i16) -> (i16) {
// CHECK:             %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_18]], %c0_i16 : i16
// CHECK:             %[[VAL_20:.*]] = fir.if %[[VAL_19]] -> (i16) {
// CHECK:               %[[VAL_21:.*]] = hlfir.designate %[[VAL_10]]#0  substr %[[VAL_17]], %[[VAL_17]]  typeparams %c1 : (!fir.ref<!fir.char<2>>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_23:.*]] = fir.extract_value %[[VAL_22]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_13]]#0  substr %[[VAL_17]], %[[VAL_17]]  typeparams %c1 : (!fir.ref<!fir.char<2>>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_26:.*]] = fir.extract_value %[[VAL_25]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:               %[[VAL_27:.*]] = arith.cmpi ult, %[[VAL_23]], %[[VAL_26]] : i16
// CHECK:               %[[VAL_28:.*]] = arith.select %[[VAL_27]], %c-1_i16, %[[VAL_18]] : i16
// CHECK:               %[[VAL_29:.*]] = arith.cmpi ugt, %[[VAL_23]], %[[VAL_26]] : i16
// CHECK:               %[[VAL_30:.*]] = arith.select %[[VAL_29]], %c1_i16, %[[VAL_28]] : i16
// CHECK:               fir.result %[[VAL_30]] : i16
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_18]] : i16
// CHECK:             }
// CHECK:             fir.result %[[VAL_20]] : i16
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = fir.if %false -> (i16) {
// CHECK:             %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %c1 to %c0 step %c1 iter_args(%[[VAL_34:.*]] = %[[VAL_16]]) -> (i16) {
// CHECK:               %[[VAL_35:.*]] = arith.cmpi eq, %[[VAL_34]], %c0_i16 : i16
// CHECK:               %[[VAL_36:.*]] = fir.if %[[VAL_35]] -> (i16) {
// CHECK:                 %[[VAL_37:.*]] = arith.addi %[[VAL_33]], %c1 : index
// CHECK:                 %[[VAL_38:.*]] = hlfir.designate %[[VAL_10]]#0  substr %[[VAL_37]], %[[VAL_37]]  typeparams %c1 : (!fir.ref<!fir.char<2>>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:                 %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.char<2>>
// CHECK:                 %[[VAL_40:.*]] = fir.extract_value %[[VAL_39]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:                 %[[VAL_41:.*]] = arith.cmpi ult, %[[VAL_40]], %c32_i16 : i16
// CHECK:                 %[[VAL_42:.*]] = arith.select %[[VAL_41]], %c-1_i16, %[[VAL_34]] : i16
// CHECK:                 %[[VAL_43:.*]] = arith.cmpi ugt, %[[VAL_40]], %c32_i16 : i16
// CHECK:                 %[[VAL_44:.*]] = arith.select %[[VAL_43]], %c1_i16, %[[VAL_42]] : i16
// CHECK:                 fir.result %[[VAL_44]] : i16
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_34]] : i16
// CHECK:               }
// CHECK:               fir.result %[[VAL_36]] : i16
// CHECK:             }
// CHECK:             fir.result %[[VAL_32]] : i16
// CHECK:           } else {
// CHECK:             %[[VAL_45:.*]] = fir.do_loop %[[VAL_46:.*]] = %c1 to %c0 step %c1 iter_args(%[[VAL_47:.*]] = %[[VAL_16]]) -> (i16) {
// CHECK:               %[[VAL_48:.*]] = arith.cmpi eq, %[[VAL_47]], %c0_i16 : i16
// CHECK:               %[[VAL_49:.*]] = fir.if %[[VAL_48]] -> (i16) {
// CHECK:                 %[[VAL_50:.*]] = arith.addi %[[VAL_46]], %c1 : index
// CHECK:                 %[[VAL_51:.*]] = hlfir.designate %[[VAL_13]]#0  substr %[[VAL_50]], %[[VAL_50]]  typeparams %c1 : (!fir.ref<!fir.char<2>>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:                 %[[VAL_52:.*]] = fir.load %[[VAL_51]] : !fir.ref<!fir.char<2>>
// CHECK:                 %[[VAL_53:.*]] = fir.extract_value %[[VAL_52]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:                 %[[VAL_54:.*]] = arith.cmpi ugt, %[[VAL_53]], %c32_i16 : i16
// CHECK:                 %[[VAL_55:.*]] = arith.select %[[VAL_54]], %c-1_i16, %[[VAL_47]] : i16
// CHECK:                 %[[VAL_56:.*]] = arith.cmpi ult, %[[VAL_53]], %c32_i16 : i16
// CHECK:                 %[[VAL_57:.*]] = arith.select %[[VAL_56]], %c1_i16, %[[VAL_55]] : i16
// CHECK:                 fir.result %[[VAL_57]] : i16
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_47]] : i16
// CHECK:               }
// CHECK:               fir.result %[[VAL_49]] : i16
// CHECK:             }
// CHECK:             fir.result %[[VAL_45]] : i16
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = arith.cmpi ne, %[[VAL_31]], %c0_i16 : i16
// CHECK:           %[[VAL_59:.*]] = fir.convert %[[VAL_58]] : (i1) -> i32
    %10 = fir.convert %9 : (i1) -> i32
    hlfir.assign %10 to %8#0 : i32, !fir.ref<i32>
    %11 = fir.load %8#0 : !fir.ref<i32>
    return %11 : i32
  }

//  function test1
//    logical :: test1
//    character*1 :: c1, c2
//    c1 = ''
//    c2 = char(255)
//    test1  = c1 .gt. c2
//  end function test1
  func.func @_QPtest1() -> !fir.logical<4> {
// CHECK-LABEL:   func.func @_QPtest1() -> !fir.logical<4> {
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.char<1> {bindc_name = "c1", uniq_name = "_QFtest1Ec1"}
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %c1 {uniq_name = "_QFtest1Ec1"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.char<1> {bindc_name = "c2", uniq_name = "_QFtest1Ec2"}
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] typeparams %c1 {uniq_name = "_QFtest1Ec2"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_12:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test1", uniq_name = "_QFtest1Etest1"}
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFtest1Etest1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQclX) : !fir.ref<!fir.char<1,0>>
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] typeparams %c0 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX"} : (!fir.ref<!fir.char<1,0>>, index) -> (!fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1,0>>)
// CHECK:           hlfir.assign %[[VAL_15]]#0 to %[[VAL_9]]#0 : !fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_16:.*]] = fir.address_of(@_QQclXFF) : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclXFF"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           hlfir.assign %[[VAL_17]]#0 to %[[VAL_11]]#0 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
    %0 = fir.dummy_scope : !fir.dscope
    %c1 = arith.constant 1 : index
    %1 = fir.alloca !fir.char<1> {bindc_name = "c1", uniq_name = "_QFtest1Ec1"}
    %2:2 = hlfir.declare %1 typeparams %c1 {uniq_name = "_QFtest1Ec1"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %c1_0 = arith.constant 1 : index
    %3 = fir.alloca !fir.char<1> {bindc_name = "c2", uniq_name = "_QFtest1Ec2"}
    %4:2 = hlfir.declare %3 typeparams %c1_0 {uniq_name = "_QFtest1Ec2"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %5 = fir.alloca !fir.logical<4> {bindc_name = "test1", uniq_name = "_QFtest1Etest1"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFtest1Etest1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %7 = fir.address_of(@_QQclX) : !fir.ref<!fir.char<1,0>>
    %c0 = arith.constant 0 : index
    %8:2 = hlfir.declare %7 typeparams %c0 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX"} : (!fir.ref<!fir.char<1,0>>, index) -> (!fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1,0>>)
    hlfir.assign %8#0 to %2#0 : !fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1>>
    %9 = fir.address_of(@_QQclXFF) : !fir.ref<!fir.char<1>>
    %c1_1 = arith.constant 1 : index
    %10:2 = hlfir.declare %9 typeparams %c1_1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclXFF"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    hlfir.assign %10#0 to %4#0 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
    %11 = hlfir.cmpchar sgt %2#0 %4#0 : (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>) -> i1
// CHECK:           %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %c1 to %c1 step %c1 iter_args(%[[VAL_20:.*]] = %c0_i8) -> (i8) {
// CHECK:             %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %c0_i8 : i8
// CHECK:             %[[VAL_22:.*]] = fir.if %[[VAL_21]] -> (i8) {
// CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_9]]#0  substr %[[VAL_19]], %[[VAL_19]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_25:.*]] = fir.extract_value %[[VAL_24]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_11]]#0  substr %[[VAL_19]], %[[VAL_19]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_28:.*]] = fir.extract_value %[[VAL_27]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_25]], %[[VAL_28]] : i8
// CHECK:               %[[VAL_30:.*]] = arith.select %[[VAL_29]], %c-1_i8, %[[VAL_20]] : i8
// CHECK:               %[[VAL_31:.*]] = arith.cmpi ugt, %[[VAL_25]], %[[VAL_28]] : i8
// CHECK:               %[[VAL_32:.*]] = arith.select %[[VAL_31]], %c1_i8, %[[VAL_30]] : i8
// CHECK:               fir.result %[[VAL_32]] : i8
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_20]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_22]] : i8
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = fir.if %false -> (i8) {
// CHECK:             %[[VAL_34:.*]] = fir.do_loop %[[VAL_35:.*]] = %c1 to %c0 step %c1 iter_args(%[[VAL_36:.*]] = %[[VAL_18]]) -> (i8) {
// CHECK:               %[[VAL_37:.*]] = arith.cmpi eq, %[[VAL_36]], %c0_i8 : i8
// CHECK:               %[[VAL_38:.*]] = fir.if %[[VAL_37]] -> (i8) {
// CHECK:                 %[[VAL_39:.*]] = arith.addi %[[VAL_35]], %c1 : index
// CHECK:                 %[[VAL_40:.*]] = hlfir.designate %[[VAL_9]]#0  substr %[[VAL_39]], %[[VAL_39]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_42:.*]] = fir.extract_value %[[VAL_41]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_43:.*]] = arith.cmpi ult, %[[VAL_42]], %c32_i8 : i8
// CHECK:                 %[[VAL_44:.*]] = arith.select %[[VAL_43]], %c-1_i8, %[[VAL_36]] : i8
// CHECK:                 %[[VAL_45:.*]] = arith.cmpi ugt, %[[VAL_42]], %c32_i8 : i8
// CHECK:                 %[[VAL_46:.*]] = arith.select %[[VAL_45]], %c1_i8, %[[VAL_44]] : i8
// CHECK:                 fir.result %[[VAL_46]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_36]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_38]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_34]] : i8
// CHECK:           } else {
// CHECK:             %[[VAL_47:.*]] = fir.do_loop %[[VAL_48:.*]] = %c1 to %c0 step %c1 iter_args(%[[VAL_49:.*]] = %[[VAL_18]]) -> (i8) {
// CHECK:               %[[VAL_50:.*]] = arith.cmpi eq, %[[VAL_49]], %c0_i8 : i8
// CHECK:               %[[VAL_51:.*]] = fir.if %[[VAL_50]] -> (i8) {
// CHECK:                 %[[VAL_52:.*]] = arith.addi %[[VAL_48]], %c1 : index
// CHECK:                 %[[VAL_53:.*]] = hlfir.designate %[[VAL_11]]#0  substr %[[VAL_52]], %[[VAL_52]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_54:.*]] = fir.load %[[VAL_53]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_55:.*]] = fir.extract_value %[[VAL_54]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_56:.*]] = arith.cmpi ugt, %[[VAL_55]], %c32_i8 : i8
// CHECK:                 %[[VAL_57:.*]] = arith.select %[[VAL_56]], %c-1_i8, %[[VAL_49]] : i8
// CHECK:                 %[[VAL_58:.*]] = arith.cmpi ult, %[[VAL_55]], %c32_i8 : i8
// CHECK:                 %[[VAL_59:.*]] = arith.select %[[VAL_58]], %c1_i8, %[[VAL_57]] : i8
// CHECK:                 fir.result %[[VAL_59]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_49]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_51]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_47]] : i8
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = arith.cmpi sgt, %[[VAL_33]], %c0_i8 : i8
// CHECK:           %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i1) -> !fir.logical<4>
    %12 = fir.convert %11 : (i1) -> !fir.logical<4>
    hlfir.assign %12 to %6#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    %13 = fir.load %6#0 : !fir.ref<!fir.logical<4>>
    return %13 : !fir.logical<4>
  }

//  function test2
//    logical :: test2
//    character*1 :: c1
//    c1 = ' '
//    test2  = c1 .lt. ' ' // char(255)
//  end function test2
  func.func @_QPtest2() -> !fir.logical<4> {
// CHECK-LABEL:   func.func @_QPtest2() -> !fir.logical<4> {
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.char<1> {bindc_name = "c1", uniq_name = "_QFtest2Ec1"}
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %c1 {uniq_name = "_QFtest2Ec1"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test2", uniq_name = "_QFtest2Etest2"}
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QFtest2Etest2"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QQclX20) : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX20"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           hlfir.assign %[[VAL_14]]#0 to %[[VAL_10]]#0 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_15:.*]] = fir.address_of(@_QQclX20FF) : !fir.ref<!fir.char<1,2>>
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] typeparams %c2 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX20FF"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
    %0 = fir.dummy_scope : !fir.dscope
    %c1 = arith.constant 1 : index
    %1 = fir.alloca !fir.char<1> {bindc_name = "c1", uniq_name = "_QFtest2Ec1"}
    %2:2 = hlfir.declare %1 typeparams %c1 {uniq_name = "_QFtest2Ec1"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %3 = fir.alloca !fir.logical<4> {bindc_name = "test2", uniq_name = "_QFtest2Etest2"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFtest2Etest2"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %5 = fir.address_of(@_QQclX20) : !fir.ref<!fir.char<1>>
    %c1_0 = arith.constant 1 : index
    %6:2 = hlfir.declare %5 typeparams %c1_0 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX20"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    hlfir.assign %6#0 to %2#0 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
    %7 = fir.address_of(@_QQclX20FF) : !fir.ref<!fir.char<1,2>>
    %c2 = arith.constant 2 : index
    %8:2 = hlfir.declare %7 typeparams %c2 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX20FF"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
    %9 = hlfir.cmpchar slt %2#0 %8#0 : (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1,2>>) -> i1
// CHECK:           %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %c1 to %c1 step %c1 iter_args(%[[VAL_19:.*]] = %c0_i8) -> (i8) {
// CHECK:             %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_19]], %c0_i8 : i8
// CHECK:             %[[VAL_21:.*]] = fir.if %[[VAL_20]] -> (i8) {
// CHECK:               %[[VAL_22:.*]] = hlfir.designate %[[VAL_10]]#0  substr %[[VAL_18]], %[[VAL_18]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_24:.*]] = fir.extract_value %[[VAL_23]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_25:.*]] = hlfir.designate %[[VAL_16]]#0  substr %[[VAL_18]], %[[VAL_18]]  typeparams %c1 : (!fir.ref<!fir.char<1,2>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_27:.*]] = fir.extract_value %[[VAL_26]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_28:.*]] = arith.cmpi ult, %[[VAL_24]], %[[VAL_27]] : i8
// CHECK:               %[[VAL_29:.*]] = arith.select %[[VAL_28]], %c-1_i8, %[[VAL_19]] : i8
// CHECK:               %[[VAL_30:.*]] = arith.cmpi ugt, %[[VAL_24]], %[[VAL_27]] : i8
// CHECK:               %[[VAL_31:.*]] = arith.select %[[VAL_30]], %c1_i8, %[[VAL_29]] : i8
// CHECK:               fir.result %[[VAL_31]] : i8
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_19]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_21]] : i8
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = fir.if %false -> (i8) {
// CHECK:             %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %c1 to %c-1 step %c1 iter_args(%[[VAL_35:.*]] = %[[VAL_17]]) -> (i8) {
// CHECK:               %[[VAL_36:.*]] = arith.cmpi eq, %[[VAL_35]], %c0_i8 : i8
// CHECK:               %[[VAL_37:.*]] = fir.if %[[VAL_36]] -> (i8) {
// CHECK:                 %[[VAL_38:.*]] = arith.addi %[[VAL_34]], %c2 : index
// CHECK:                 %[[VAL_39:.*]] = hlfir.designate %[[VAL_10]]#0  substr %[[VAL_38]], %[[VAL_38]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_40:.*]] = fir.load %[[VAL_39]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_41:.*]] = fir.extract_value %[[VAL_40]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_42:.*]] = arith.cmpi ult, %[[VAL_41]], %c32_i8 : i8
// CHECK:                 %[[VAL_43:.*]] = arith.select %[[VAL_42]], %c-1_i8, %[[VAL_35]] : i8
// CHECK:                 %[[VAL_44:.*]] = arith.cmpi ugt, %[[VAL_41]], %c32_i8 : i8
// CHECK:                 %[[VAL_45:.*]] = arith.select %[[VAL_44]], %c1_i8, %[[VAL_43]] : i8
// CHECK:                 fir.result %[[VAL_45]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_35]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_37]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_33]] : i8
// CHECK:           } else {
// CHECK:             %[[VAL_46:.*]] = fir.do_loop %[[VAL_47:.*]] = %c1 to %c1 step %c1 iter_args(%[[VAL_48:.*]] = %[[VAL_17]]) -> (i8) {
// CHECK:               %[[VAL_49:.*]] = arith.cmpi eq, %[[VAL_48]], %c0_i8 : i8
// CHECK:               %[[VAL_50:.*]] = fir.if %[[VAL_49]] -> (i8) {
// CHECK:                 %[[VAL_51:.*]] = arith.addi %[[VAL_47]], %c1 : index
// CHECK:                 %[[VAL_52:.*]] = hlfir.designate %[[VAL_16]]#0  substr %[[VAL_51]], %[[VAL_51]]  typeparams %c1 : (!fir.ref<!fir.char<1,2>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_53:.*]] = fir.load %[[VAL_52]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_54:.*]] = fir.extract_value %[[VAL_53]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_55:.*]] = arith.cmpi ugt, %[[VAL_54]], %c32_i8 : i8
// CHECK:                 %[[VAL_56:.*]] = arith.select %[[VAL_55]], %c-1_i8, %[[VAL_48]] : i8
// CHECK:                 %[[VAL_57:.*]] = arith.cmpi ult, %[[VAL_54]], %c32_i8 : i8
// CHECK:                 %[[VAL_58:.*]] = arith.select %[[VAL_57]], %c1_i8, %[[VAL_56]] : i8
// CHECK:                 fir.result %[[VAL_58]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_48]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_50]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_46]] : i8
// CHECK:           }
// CHECK:           %[[VAL_59:.*]] = arith.cmpi slt, %[[VAL_32]], %c0_i8 : i8
// CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i1) -> !fir.logical<4>
    %10 = fir.convert %9 : (i1) -> !fir.logical<4>
    hlfir.assign %10 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    %11 = fir.load %4#0 : !fir.ref<!fir.logical<4>>
    return %11 : !fir.logical<4>
  }

//  function test3
//    logical :: test3
//    character*1 :: c1, c2
//    c2 = 'a' // 'b'
//    c1 = 'a'
//    test3  = c2 .gt. c1
//  end function test3
  func.func @_QPtest3() -> !fir.logical<4> {
// CHECK-LABEL:   func.func @_QPtest3() -> !fir.logical<4> {
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.char<1> {bindc_name = "c1", uniq_name = "_QFtest3Ec1"}
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %c1 {uniq_name = "_QFtest3Ec1"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.char<1> {bindc_name = "c2", uniq_name = "_QFtest3Ec2"}
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] typeparams %c1 {uniq_name = "_QFtest3Ec2"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_13:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test3", uniq_name = "_QFtest3Etest3"}
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QFtest3Etest3"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_15:.*]] = fir.address_of(@_QQclX6162) : !fir.ref<!fir.char<1,2>>
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] typeparams %c2 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX6162"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
// CHECK:           hlfir.assign %[[VAL_16]]#0 to %[[VAL_12]]#0 : !fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_17:.*]] = fir.address_of(@_QQclX61) : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX61"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           hlfir.assign %[[VAL_18]]#0 to %[[VAL_10]]#0 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
    %0 = fir.dummy_scope : !fir.dscope
    %c1 = arith.constant 1 : index
    %1 = fir.alloca !fir.char<1> {bindc_name = "c1", uniq_name = "_QFtest3Ec1"}
    %2:2 = hlfir.declare %1 typeparams %c1 {uniq_name = "_QFtest3Ec1"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %c1_0 = arith.constant 1 : index
    %3 = fir.alloca !fir.char<1> {bindc_name = "c2", uniq_name = "_QFtest3Ec2"}
    %4:2 = hlfir.declare %3 typeparams %c1_0 {uniq_name = "_QFtest3Ec2"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %5 = fir.alloca !fir.logical<4> {bindc_name = "test3", uniq_name = "_QFtest3Etest3"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFtest3Etest3"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %7 = fir.address_of(@_QQclX6162) : !fir.ref<!fir.char<1,2>>
    %c2 = arith.constant 2 : index
    %8:2 = hlfir.declare %7 typeparams %c2 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX6162"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
    hlfir.assign %8#0 to %4#0 : !fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1>>
    %9 = fir.address_of(@_QQclX61) : !fir.ref<!fir.char<1>>
    %c1_1 = arith.constant 1 : index
    %10:2 = hlfir.declare %9 typeparams %c1_1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX61"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    hlfir.assign %10#0 to %2#0 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
    %11 = hlfir.cmpchar sgt %4#0 %2#0 : (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>) -> i1
// CHECK:           %[[VAL_19:.*]] = fir.do_loop %[[VAL_20:.*]] = %c1 to %c1 step %c1 iter_args(%[[VAL_21:.*]] = %c0_i8) -> (i8) {
// CHECK:             %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_21]], %c0_i8 : i8
// CHECK:             %[[VAL_23:.*]] = fir.if %[[VAL_22]] -> (i8) {
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_12]]#0  substr %[[VAL_20]], %[[VAL_20]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_26:.*]] = fir.extract_value %[[VAL_25]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_27:.*]] = hlfir.designate %[[VAL_10]]#0  substr %[[VAL_20]], %[[VAL_20]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_29:.*]] = fir.extract_value %[[VAL_28]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_26]], %[[VAL_29]] : i8
// CHECK:               %[[VAL_31:.*]] = arith.select %[[VAL_30]], %c-1_i8, %[[VAL_21]] : i8
// CHECK:               %[[VAL_32:.*]] = arith.cmpi ugt, %[[VAL_26]], %[[VAL_29]] : i8
// CHECK:               %[[VAL_33:.*]] = arith.select %[[VAL_32]], %c1_i8, %[[VAL_31]] : i8
// CHECK:               fir.result %[[VAL_33]] : i8
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_21]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_23]] : i8
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = fir.if %false -> (i8) {
// CHECK:             %[[VAL_35:.*]] = fir.do_loop %[[VAL_36:.*]] = %c1 to %c0 step %c1 iter_args(%[[VAL_37:.*]] = %[[VAL_19]]) -> (i8) {
// CHECK:               %[[VAL_38:.*]] = arith.cmpi eq, %[[VAL_37]], %c0_i8 : i8
// CHECK:               %[[VAL_39:.*]] = fir.if %[[VAL_38]] -> (i8) {
// CHECK:                 %[[VAL_40:.*]] = arith.addi %[[VAL_36]], %c1 : index
// CHECK:                 %[[VAL_41:.*]] = hlfir.designate %[[VAL_12]]#0  substr %[[VAL_40]], %[[VAL_40]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_42:.*]] = fir.load %[[VAL_41]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_43:.*]] = fir.extract_value %[[VAL_42]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_44:.*]] = arith.cmpi ult, %[[VAL_43]], %c32_i8 : i8
// CHECK:                 %[[VAL_45:.*]] = arith.select %[[VAL_44]], %c-1_i8, %[[VAL_37]] : i8
// CHECK:                 %[[VAL_46:.*]] = arith.cmpi ugt, %[[VAL_43]], %c32_i8 : i8
// CHECK:                 %[[VAL_47:.*]] = arith.select %[[VAL_46]], %c1_i8, %[[VAL_45]] : i8
// CHECK:                 fir.result %[[VAL_47]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_37]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_39]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_35]] : i8
// CHECK:           } else {
// CHECK:             %[[VAL_48:.*]] = fir.do_loop %[[VAL_49:.*]] = %c1 to %c0 step %c1 iter_args(%[[VAL_50:.*]] = %[[VAL_19]]) -> (i8) {
// CHECK:               %[[VAL_51:.*]] = arith.cmpi eq, %[[VAL_50]], %c0_i8 : i8
// CHECK:               %[[VAL_52:.*]] = fir.if %[[VAL_51]] -> (i8) {
// CHECK:                 %[[VAL_53:.*]] = arith.addi %[[VAL_49]], %c1 : index
// CHECK:                 %[[VAL_54:.*]] = hlfir.designate %[[VAL_10]]#0  substr %[[VAL_53]], %[[VAL_53]]  typeparams %c1 : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_55:.*]] = fir.load %[[VAL_54]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_56:.*]] = fir.extract_value %[[VAL_55]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_57:.*]] = arith.cmpi ugt, %[[VAL_56]], %c32_i8 : i8
// CHECK:                 %[[VAL_58:.*]] = arith.select %[[VAL_57]], %c-1_i8, %[[VAL_50]] : i8
// CHECK:                 %[[VAL_59:.*]] = arith.cmpi ult, %[[VAL_56]], %c32_i8 : i8
// CHECK:                 %[[VAL_60:.*]] = arith.select %[[VAL_59]], %c1_i8, %[[VAL_58]] : i8
// CHECK:                 fir.result %[[VAL_60]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_50]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_52]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_48]] : i8
// CHECK:           }
// CHECK:           %[[VAL_61:.*]] = arith.cmpi sgt, %[[VAL_34]], %c0_i8 : i8
// CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_61]] : (i1) -> !fir.logical<4>
    %12 = fir.convert %11 : (i1) -> !fir.logical<4>
    hlfir.assign %12 to %6#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    %13 = fir.load %6#0 : !fir.ref<!fir.logical<4>>
    return %13 : !fir.logical<4>
  }

//  function test4(c1,c2,c3)
//    implicit none
//    logical :: test4
//    character(len=*,kind=1) :: c1, c2, c3
//    test4  = c1 // c2 .gt. c3
//  end function test4
  func.func @_QPtest4(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) -> !fir.logical<4> {
// CHECK-LABEL:   func.func @_QPtest4(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) -> !fir.logical<4> {
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_5]] {uniq_name = "_QFtest4Ec1"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]]#0 typeparams %[[VAL_8]]#1 dummy_scope %[[VAL_5]] {uniq_name = "_QFtest4Ec2"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_10:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]]#0 typeparams %[[VAL_10]]#1 dummy_scope %[[VAL_5]] {uniq_name = "_QFtest4Ec3"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_12:.*]] = fir.alloca !fir.logical<4> {bindc_name = "test4", uniq_name = "_QFtest4Etest4"}
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFtest4Etest4"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 {uniq_name = "_QFtest4Ec1"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %3:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFtest4Ec2"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %5:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %6:2 = hlfir.declare %5#0 typeparams %5#1 dummy_scope %0 {uniq_name = "_QFtest4Ec3"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %7 = fir.alloca !fir.logical<4> {bindc_name = "test4", uniq_name = "_QFtest4Etest4"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFtest4Etest4"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %9 = arith.addi %1#1, %3#1 : index
    %10 = hlfir.concat %2#0, %4#0 len %9 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
    %11 = hlfir.cmpchar sgt %10 %6#0 : (!hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>) -> i1
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_8]]#1 : index
// CHECK:           %[[VAL_15:.*]] = hlfir.concat %[[VAL_7]]#0, %[[VAL_9]]#0 len %[[VAL_14]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:           %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_15]] typeparams %[[VAL_14]] {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
// CHECK:           %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_14]], %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_19:.*]] = fir.do_loop %[[VAL_20:.*]] = %c1 to %[[VAL_18]] step %c1 iter_args(%[[VAL_21:.*]] = %c0_i8) -> (i8) {
// CHECK:             %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_21]], %c0_i8 : i8
// CHECK:             %[[VAL_23:.*]] = fir.if %[[VAL_22]] -> (i8) {
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_16]]#0  substr %[[VAL_20]], %[[VAL_20]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_26:.*]] = fir.extract_value %[[VAL_25]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_27:.*]] = hlfir.designate %[[VAL_11]]#0  substr %[[VAL_20]], %[[VAL_20]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_29:.*]] = fir.extract_value %[[VAL_28]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_26]], %[[VAL_29]] : i8
// CHECK:               %[[VAL_31:.*]] = arith.select %[[VAL_30]], %c-1_i8, %[[VAL_21]] : i8
// CHECK:               %[[VAL_32:.*]] = arith.cmpi ugt, %[[VAL_26]], %[[VAL_29]] : i8
// CHECK:               %[[VAL_33:.*]] = arith.select %[[VAL_32]], %c1_i8, %[[VAL_31]] : i8
// CHECK:               fir.result %[[VAL_33]] : i8
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_21]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_23]] : i8
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_35:.*]] = fir.if %[[VAL_34]] -> (i8) {
// CHECK:             %[[VAL_36:.*]] = arith.subi %[[VAL_14]], %[[VAL_10]]#1 : index
// CHECK:             %[[VAL_37:.*]] = fir.do_loop %[[VAL_38:.*]] = %c1 to %[[VAL_36]] step %c1 iter_args(%[[VAL_39:.*]] = %[[VAL_19]]) -> (i8) {
// CHECK:               %[[VAL_40:.*]] = arith.cmpi eq, %[[VAL_39]], %c0_i8 : i8
// CHECK:               %[[VAL_41:.*]] = fir.if %[[VAL_40]] -> (i8) {
// CHECK:                 %[[VAL_42:.*]] = arith.addi %[[VAL_10]]#1, %[[VAL_38]] : index
// CHECK:                 %[[VAL_43:.*]] = hlfir.designate %[[VAL_16]]#0  substr %[[VAL_42]], %[[VAL_42]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_45:.*]] = fir.extract_value %[[VAL_44]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_46:.*]] = arith.cmpi ult, %[[VAL_45]], %c32_i8 : i8
// CHECK:                 %[[VAL_47:.*]] = arith.select %[[VAL_46]], %c-1_i8, %[[VAL_39]] : i8
// CHECK:                 %[[VAL_48:.*]] = arith.cmpi ugt, %[[VAL_45]], %c32_i8 : i8
// CHECK:                 %[[VAL_49:.*]] = arith.select %[[VAL_48]], %c1_i8, %[[VAL_47]] : i8
// CHECK:                 fir.result %[[VAL_49]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_39]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_41]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_37]] : i8
// CHECK:           } else {
// CHECK:             %[[VAL_50:.*]] = arith.subi %[[VAL_10]]#1, %[[VAL_14]] : index
// CHECK:             %[[VAL_51:.*]] = fir.do_loop %[[VAL_52:.*]] = %c1 to %[[VAL_50]] step %c1 iter_args(%[[VAL_53:.*]] = %[[VAL_19]]) -> (i8) {
// CHECK:               %[[VAL_54:.*]] = arith.cmpi eq, %[[VAL_53]], %c0_i8 : i8
// CHECK:               %[[VAL_55:.*]] = fir.if %[[VAL_54]] -> (i8) {
// CHECK:                 %[[VAL_56:.*]] = arith.addi %[[VAL_14]], %[[VAL_52]] : index
// CHECK:                 %[[VAL_57:.*]] = hlfir.designate %[[VAL_11]]#0  substr %[[VAL_56]], %[[VAL_56]]  typeparams %c1 : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_58:.*]] = fir.load %[[VAL_57]] : !fir.ref<!fir.char<1>>
// CHECK:                 %[[VAL_59:.*]] = fir.extract_value %[[VAL_58]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:                 %[[VAL_60:.*]] = arith.cmpi ugt, %[[VAL_59]], %c32_i8 : i8
// CHECK:                 %[[VAL_61:.*]] = arith.select %[[VAL_60]], %c-1_i8, %[[VAL_53]] : i8
// CHECK:                 %[[VAL_62:.*]] = arith.cmpi ult, %[[VAL_59]], %c32_i8 : i8
// CHECK:                 %[[VAL_63:.*]] = arith.select %[[VAL_62]], %c1_i8, %[[VAL_61]] : i8
// CHECK:                 fir.result %[[VAL_63]] : i8
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_53]] : i8
// CHECK:               }
// CHECK:               fir.result %[[VAL_55]] : i8
// CHECK:             }
// CHECK:             fir.result %[[VAL_51]] : i8
// CHECK:           }
// CHECK:           hlfir.end_associate %[[VAL_16]]#1, %[[VAL_16]]#2 : !fir.ref<!fir.char<1,?>>, i1
// CHECK:           %[[VAL_64:.*]] = arith.cmpi sgt, %[[VAL_35]], %c0_i8 : i8
// CHECK:           %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (i1) -> !fir.logical<4>
    %12 = fir.convert %11 : (i1) -> !fir.logical<4>
    hlfir.assign %12 to %8#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    %13 = fir.load %8#0 : !fir.ref<!fir.logical<4>>
    return %13 : !fir.logical<4>
  }
