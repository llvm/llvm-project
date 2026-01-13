// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

// Simplify should reduce hlfir.index to constant (5)
func.func @_QPt1() {
// CHECK-LABEL:   func.func @_QPt1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt1En"}
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFt1En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.char<1,4> {bindc_name = "s", uniq_name = "_QFt1Es"}
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_3]] {uniq_name = "_QFt1Es"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
// CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QQclX616263) : !fir.ref<!fir.char<1,3>>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %[[VAL_2]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX616263"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
// CHECK:           hlfir.assign %[[VAL_10]]#0 to %[[VAL_8]]#0 : !fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,4>>
// CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QQclX) : !fir.ref<!fir.char<1,0>>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX"} : (!fir.ref<!fir.char<1,0>>, index) -> (!fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1,0>>)
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_0]] : (index) -> i32
// CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_6]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt1En"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFt1En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c4 = arith.constant 4 : index
    %3 = fir.alloca !fir.char<1,4> {bindc_name = "s", uniq_name = "_QFt1Es"}
    %4:2 = hlfir.declare %3 typeparams %c4 {uniq_name = "_QFt1Es"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
    %5 = fir.address_of(@_QQclX616263) : !fir.ref<!fir.char<1,3>>
    %c3 = arith.constant 3 : index
    %6:2 = hlfir.declare %5 typeparams %c3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX616263"} : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
    hlfir.assign %6#0 to %4#0 : !fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,4>>
    %7 = fir.address_of(@_QQclX) : !fir.ref<!fir.char<1,0>>
    %c0 = arith.constant 0 : index
    %8:2 = hlfir.declare %7 typeparams %c0 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX"} : (!fir.ref<!fir.char<1,0>>, index) -> (!fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1,0>>)
    %true = arith.constant true
    %9 = hlfir.index %8#0 in %4#0 back %true : (!fir.ref<!fir.char<1,0>>, !fir.ref<!fir.char<1,4>>, i1) -> i32
    hlfir.assign %9 to %2#0 : i32, !fir.ref<i32>
    return
}

// ! 'back' is unknown at compile time, substring is zero length - generate select (back ? strlen+1 : 1)
func.func @_QPt2(%arg0: !fir.boxchar<2> {fir.bindc_name = "s"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) {
// CHECK-LABEL:   func.func @_QPt2(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<2> {fir.bindc_name = "s"},
// CHECK-SAME:                     %[[ARG1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFt2Eb"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt2En"}
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFt2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt2Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QQcl2X) : !fir.ref<!fir.char<2,0>>
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X"} : (!fir.ref<!fir.char<2,0>>, index) -> (!fir.ref<!fir.char<2,0>>, !fir.ref<!fir.char<2,0>>)
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.logical<4>>
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_0]] : index
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_10]] : (!fir.logical<4>) -> i1
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_0]] : index
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
// CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFt2Eb"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %2 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt2En"}
    %3:2 = hlfir.declare %2 {uniq_name = "_QFt2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4:2 = fir.unboxchar %arg0 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %5:2 = hlfir.declare %4#0 typeparams %4#1 dummy_scope %0 {uniq_name = "_QFt2Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %6 = fir.address_of(@_QQcl2X) : !fir.ref<!fir.char<2,0>>
    %c0 = arith.constant 0 : index
    %7:2 = hlfir.declare %6 typeparams %c0 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X"} : (!fir.ref<!fir.char<2,0>>, index) -> (!fir.ref<!fir.char<2,0>>, !fir.ref<!fir.char<2,0>>)
    %8 = fir.load %1#0 : !fir.ref<!fir.logical<4>>
    %9 = hlfir.index %7#0 in %5#0 back %8 : (!fir.ref<!fir.char<2,0>>, !fir.boxchar<2>, !fir.logical<4>) -> i32
    hlfir.assign %9 to %3#0 : i32, !fir.ref<i32>
    return
}

// inline as search loop (backward)
func.func @_QPt3(%arg0: !fir.boxchar<2> {fir.bindc_name = "s"}) {
// CHECK-LABEL:   func.func @_QPt3(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<2> {fir.bindc_name = "s"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt3En"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt3En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt3Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQcl2X6500) : !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X6500"} : (!fir.ref<!fir.char<2>>, index) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
// CHECK:           %[[VAL_9:.*]] = hlfir.designate %[[VAL_8]]#0  substr %[[VAL_1]], %[[VAL_1]]  typeparams %[[VAL_1]] : (!fir.ref<!fir.char<2>>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_11:.*]] = fir.extract_value %[[VAL_10]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_1]] to %[[VAL_5]]#1 step %[[VAL_1]] iter_args(%[[VAL_15:.*]] = %[[VAL_0]]) -> (index) {
// CHECK:             %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_0]] : index
// CHECK:             %[[VAL_17:.*]] = fir.if %[[VAL_16]] -> (index) {
// CHECK:               %[[VAL_18:.*]] = arith.subi %[[VAL_12]], %[[VAL_14]] : index
// CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_6]]#0  substr %[[VAL_18]], %[[VAL_18]]  typeparams %[[VAL_1]] : (!fir.boxchar<2>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_21:.*]] = fir.extract_value %[[VAL_20]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:               %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_21]], %[[VAL_11]] : i16
// CHECK:               %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_18]], %[[VAL_15]] : index
// CHECK:               fir.result %[[VAL_23]] : index
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_15]] : index
// CHECK:             }
// CHECK:             fir.result %[[VAL_17]] : index
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
// CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt3En"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFt3En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFt3Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %5 = fir.address_of(@_QQcl2X6500) : !fir.ref<!fir.char<2>>
    %c1 = arith.constant 1 : index
    %6:2 = hlfir.declare %5 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X6500"} : (!fir.ref<!fir.char<2>>, index) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
    %true = arith.constant true
    %7 = hlfir.index %6#0 in %4#0 back %true : (!fir.ref<!fir.char<2>>, !fir.boxchar<2>, i1) -> i32
    hlfir.assign %7 to %2#0 : i32, !fir.ref<i32>
    return
}

//inline as search loop (forward)
func.func @_QPt4(%arg0: !fir.boxchar<2> {fir.bindc_name = "s"}) {
// CHECK-LABEL:   func.func @_QPt4(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<2> {fir.bindc_name = "s"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt4En"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt4En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt4Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQcl2X6500) : !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X6500"} : (!fir.ref<!fir.char<2>>, index) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
// CHECK:           %[[VAL_9:.*]] = hlfir.designate %[[VAL_8]]#0  substr %[[VAL_1]], %[[VAL_1]]  typeparams %[[VAL_1]] : (!fir.ref<!fir.char<2>>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_11:.*]] = fir.extract_value %[[VAL_10]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:           %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_1]] to %[[VAL_5]]#1 step %[[VAL_1]] iter_args(%[[VAL_14:.*]] = %[[VAL_0]]) -> (index) {
// CHECK:             %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_16:.*]] = fir.if %[[VAL_15]] -> (index) {
// CHECK:               %[[VAL_17:.*]] = hlfir.designate %[[VAL_6]]#0  substr %[[VAL_13]], %[[VAL_13]]  typeparams %[[VAL_1]] : (!fir.boxchar<2>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_19:.*]] = fir.extract_value %[[VAL_18]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:               %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_19]], %[[VAL_11]] : i16
// CHECK:               %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_13]], %[[VAL_14]] : index
// CHECK:               fir.result %[[VAL_21]] : index
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_14]] : index
// CHECK:             }
// CHECK:             fir.result %[[VAL_16]] : index
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
// CHECK:           hlfir.assign %[[VAL_22]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt4En"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFt4En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFt4Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %5 = fir.address_of(@_QQcl2X6500) : !fir.ref<!fir.char<2>>
    %c1 = arith.constant 1 : index
    %6:2 = hlfir.declare %5 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X6500"} : (!fir.ref<!fir.char<2>>, index) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
    %false = arith.constant false
    %7 = hlfir.index %6#0 in %4#0 back %false : (!fir.ref<!fir.char<2>>, !fir.boxchar<2>, i1) -> i32
    hlfir.assign %7 to %2#0 : i32, !fir.ref<i32>
    return
}

// Same as t4 above but result kind=1
func.func @_QPt5(%arg0: !fir.boxchar<2> {fir.bindc_name = "s"}) {
// CHECK-LABEL:   func.func @_QPt5(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<2> {fir.bindc_name = "s"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt5En"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt5En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt5Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQcl2X6500) : !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X6500"} : (!fir.ref<!fir.char<2>>, index) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
// CHECK:           %[[VAL_9:.*]] = hlfir.designate %[[VAL_8]]#0  substr %[[VAL_1]], %[[VAL_1]]  typeparams %[[VAL_1]] : (!fir.ref<!fir.char<2>>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.char<2>>
// CHECK:           %[[VAL_11:.*]] = fir.extract_value %[[VAL_10]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:           %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_1]] to %[[VAL_5]]#1 step %[[VAL_1]] iter_args(%[[VAL_14:.*]] = %[[VAL_0]]) -> (index) {
// CHECK:             %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_14]], %[[VAL_0]] : index
// CHECK:             %[[VAL_16:.*]] = fir.if %[[VAL_15]] -> (index) {
// CHECK:               %[[VAL_17:.*]] = hlfir.designate %[[VAL_6]]#0  substr %[[VAL_13]], %[[VAL_13]]  typeparams %[[VAL_1]] : (!fir.boxchar<2>, index, index, index) -> !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<!fir.char<2>>
// CHECK:               %[[VAL_19:.*]] = fir.extract_value %[[VAL_18]], [0 : index] : (!fir.char<2>) -> i16
// CHECK:               %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_19]], %[[VAL_11]] : i16
// CHECK:               %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_13]], %[[VAL_14]] : index
// CHECK:               fir.result %[[VAL_21]] : index
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_14]] : index
// CHECK:             }
// CHECK:             fir.result %[[VAL_16]] : index
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_12]] : (index) -> i8
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i8) -> i32
// CHECK:           hlfir.assign %[[VAL_23]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt5En"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFt5En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFt5Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %5 = fir.address_of(@_QQcl2X6500) : !fir.ref<!fir.char<2>>
    %c1 = arith.constant 1 : index
    %6:2 = hlfir.declare %5 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl2X6500"} : (!fir.ref<!fir.char<2>>, index) -> (!fir.ref<!fir.char<2>>, !fir.ref<!fir.char<2>>)
    %false = arith.constant false
    %7 = hlfir.index %6#0 in %4#0 back %false : (!fir.ref<!fir.char<2>>, !fir.boxchar<2>, i1) -> i8
    %8 = fir.convert %7 : (i8) -> i32
    hlfir.assign %8 to %2#0 : i32, !fir.ref<i32>
    return
  }

// Do no simplify - runtime call for forward search with character kind=1 is faster
func.func @_QPt6(%arg0: !fir.boxchar<1> {fir.bindc_name = "s"}) {
// CHECK-LABEL:   func.func @_QPt6(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "s"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt6En"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt6En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt6Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX65) : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX65"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_9:.*]] = hlfir.index %[[VAL_8]]#0 in %[[VAL_6]]#0 back %[[VAL_0]] : (!fir.ref<!fir.char<1>>, !fir.boxchar<1>, i1) -> i32
// CHECK:           hlfir.assign %[[VAL_9]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt6En"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFt6En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFt6Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %5 = fir.address_of(@_QQclX65) : !fir.ref<!fir.char<1>>
    %c1 = arith.constant 1 : index
    %6:2 = hlfir.declare %5 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX65"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %false = arith.constant false
    %7 = hlfir.index %6#0 in %4#0 back %false : (!fir.ref<!fir.char<1>>, !fir.boxchar<1>, i1) -> i32
    hlfir.assign %7 to %2#0 : i32, !fir.ref<i32>
    return
}

// Do not simplify - runtime call for forward search with character kind=1 is faster
// Lookup direction is unknown at compile time, hence forward is pessimistically assumed
func.func @_QPt7(%arg0: !fir.boxchar<1> {fir.bindc_name = "s"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) {
// CHECK-LABEL:   func.func @_QPt7(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "s"},
// CHECK-SAME:                     %[[ARG1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_1]] {uniq_name = "_QFt7Eb"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt7En"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt7En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFt7Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX65) : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_0]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX65"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
// CHECK:           %[[VAL_10:.*]] = hlfir.index %[[VAL_8]]#0 in %[[VAL_6]]#0 back %[[VAL_9]] : (!fir.ref<!fir.char<1>>, !fir.boxchar<1>, !fir.logical<4>) -> i32
// CHECK:           hlfir.assign %[[VAL_10]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFt7Eb"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %2 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt7En"}
    %3:2 = hlfir.declare %2 {uniq_name = "_QFt7En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %5:2 = hlfir.declare %4#0 typeparams %4#1 dummy_scope %0 {uniq_name = "_QFt7Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %6 = fir.address_of(@_QQclX65) : !fir.ref<!fir.char<1>>
    %c1 = arith.constant 1 : index
    %7:2 = hlfir.declare %6 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX65"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %8 = fir.load %1#0 : !fir.ref<!fir.logical<4>>
    %9 = hlfir.index %7#0 in %5#0 back %8 : (!fir.ref<!fir.char<1>>, !fir.boxchar<1>, !fir.logical<4>) -> i32
    hlfir.assign %9 to %3#0 : i32, !fir.ref<i32>
    return
}

// Inline as backward search loop for character kind=1.
// The case similar to t7 but direction is known, so it is faster than runtime call.
func.func @_QPt8(%arg0: !fir.boxchar<1> {fir.bindc_name = "s"}) {
// CHECK-LABEL:   func.func @_QPt8(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "s"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt8En"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt8En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt8Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX65) : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX65"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_9:.*]] = hlfir.designate %[[VAL_8]]#0  substr %[[VAL_1]], %[[VAL_1]]  typeparams %[[VAL_1]] : (!fir.ref<!fir.char<1>>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_11:.*]] = fir.extract_value %[[VAL_10]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_1]] to %[[VAL_5]]#1 step %[[VAL_1]] iter_args(%[[VAL_15:.*]] = %[[VAL_0]]) -> (index) {
// CHECK:             %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_0]] : index
// CHECK:             %[[VAL_17:.*]] = fir.if %[[VAL_16]] -> (index) {
// CHECK:               %[[VAL_18:.*]] = arith.subi %[[VAL_12]], %[[VAL_14]] : index
// CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_6]]#0  substr %[[VAL_18]], %[[VAL_18]]  typeparams %[[VAL_1]] : (!fir.boxchar<1>, index, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.char<1>>
// CHECK:               %[[VAL_21:.*]] = fir.extract_value %[[VAL_20]], [0 : index] : (!fir.char<1>) -> i8
// CHECK:               %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_21]], %[[VAL_11]] : i8
// CHECK:               %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_18]], %[[VAL_15]] : index
// CHECK:               fir.result %[[VAL_23]] : index
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_15]] : index
// CHECK:             }
// CHECK:             fir.result %[[VAL_17]] : index
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
// CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
// CHECK:           return
// CHECK:         }
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt8En"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFt8En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFt8Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %5 = fir.address_of(@_QQclX65) : !fir.ref<!fir.char<1>>
    %c1 = arith.constant 1 : index
    %6:2 = hlfir.declare %5 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX65"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    %true = arith.constant true
    %7 = hlfir.index %6#0 in %4#0 back %true : (!fir.ref<!fir.char<1>>, !fir.boxchar<1>, i1) -> i32
    hlfir.assign %7 to %2#0 : i32, !fir.ref<i32>
    return
}

