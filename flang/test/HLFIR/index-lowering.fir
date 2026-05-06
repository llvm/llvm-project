// Test hlfir.index operation lowering to a fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

func.func @_QPt(%arg0: !fir.boxchar<1> {fir.bindc_name = "s"}) {
// CHECK-LABEL:   func.func @_QPt(
// CHECK-SAME:                    %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "s"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFtEn"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFtEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFtEs"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX74686973) : !fir.ref<!fir.char<1,4>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX74686973"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFtEn"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFtEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %4:2 = hlfir.declare %3#0 typeparams %3#1 dummy_scope %0 {uniq_name = "_QFtEs"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %5 = fir.address_of(@_QQclX74686973) : !fir.ref<!fir.char<1,4>>
    %c4 = arith.constant 4 : index
    %6:2 = hlfir.declare %5 typeparams %c4 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX74686973"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
    %7 = hlfir.index %6#0 in %4#0 : (!fir.ref<!fir.char<1,4>>, !fir.boxchar<1>) -> i32
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
// CHECK:           %[[VAL_13:.*]] = fir.call @_FortranAIndex1(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_0]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
// CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
    hlfir.assign %7 to %2#0 : i32, !fir.ref<i32>
    return
}

func.func @_QPt1(%arg0: !fir.boxchar<1> {fir.bindc_name = "s"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) {
// CHECK-LABEL:   func.func @_QPt1(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "s"},
// CHECK-SAME:                     %[[ARG1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_1]] {uniq_name = "_QFt1Eb"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt1En"}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFt1En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFt1Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QQclX74686973) : !fir.ref<!fir.char<1,4>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_0]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX74686973"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFt1Eb"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %2 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt1En"}
    %3:2 = hlfir.declare %2 {uniq_name = "_QFt1En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %5:2 = hlfir.declare %4#0 typeparams %4#1 dummy_scope %0 {uniq_name = "_QFt1Es"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %6 = fir.address_of(@_QQclX74686973) : !fir.ref<!fir.char<1,4>>
    %c4 = arith.constant 4 : index
    %7:2 = hlfir.declare %6 typeparams %c4 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX74686973"} : (!fir.ref<!fir.char<1,4>>, index) -> (!fir.ref<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>)
    %8 = fir.load %1#0 : !fir.ref<!fir.logical<4>>
    %9 = hlfir.index %7#0 in %5#0 back %8 : (!fir.ref<!fir.char<1,4>>, !fir.boxchar<1>, !fir.logical<4>) -> i32
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_0]] : (index) -> i64
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_9]] : (!fir.logical<4>) -> i1
// CHECK:           %[[VAL_15:.*]] = fir.call @_FortranAIndex1(%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> i32
// CHECK:           hlfir.assign %[[VAL_16]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
    hlfir.assign %9 to %3#0 : i32, !fir.ref<i32>
    return
}

func.func @_QPt2(%arg0: !fir.boxchar<2> {fir.bindc_name = "s"}, %arg1: !fir.boxchar<2> {fir.bindc_name = "c"}) {
// CHECK-LABEL:   func.func @_QPt2(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<2> {fir.bindc_name = "s"},
// CHECK-SAME:                     %[[ARG1:.*]]: !fir.boxchar<2> {fir.bindc_name = "c"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFt2Ec"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt2En"}
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFt2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFt2Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = fir.unboxchar %arg1 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 {uniq_name = "_QFt2Ec"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %3 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt2En"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFt2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = fir.unboxchar %arg0 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %6:2 = hlfir.declare %5#0 typeparams %5#1 dummy_scope %0 {uniq_name = "_QFt2Es"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %false = arith.constant false
    %7 = hlfir.index %2#0 in %6#0 back %false : (!fir.boxchar<2>, !fir.boxchar<2>, i1) -> i32
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<i16>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<i16>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
// CHECK:           %[[VAL_12:.*]] = fir.call @_FortranAIndex2(%[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_0]]) : (!fir.ref<i16>, i64, !fir.ref<i16>, i64, i1) -> i64
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> i32
// CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
    hlfir.assign %7 to %4#0 : i32, !fir.ref<i32>
    return
}

func.func @_QPt3(%arg0: !fir.boxchar<4> {fir.bindc_name = "s"}, %arg1: !fir.boxchar<4> {fir.bindc_name = "c"}) {
// CHECK-LABEL:   func.func @_QPt3(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<4> {fir.bindc_name = "s"},
// CHECK-SAME:                     %[[ARG1:.*]]: !fir.boxchar<4> {fir.bindc_name = "c"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFt3Ec"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
// CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt3En"}
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFt3En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFt3Es"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = fir.unboxchar %arg1 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
    %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 {uniq_name = "_QFt3Ec"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
    %3 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFt3En"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFt3En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = fir.unboxchar %arg0 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
    %6:2 = hlfir.declare %5#0 typeparams %5#1 dummy_scope %0 {uniq_name = "_QFt3Es"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
    %true = arith.constant true
    %7 = hlfir.index %2#0 in %6#0 back %true : (!fir.boxchar<4>, !fir.boxchar<4>, i1) -> i8
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
// CHECK:           %[[VAL_12:.*]] = fir.call @_FortranAIndex4(%[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_0]]) : (!fir.ref<i32>, i64, !fir.ref<i32>, i64, i1) -> i64
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> i8
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i8) -> i32
// CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
    %8 = fir.convert %7 : (i8) -> i32
    hlfir.assign %8 to %4#0 : i32, !fir.ref<i32>
    return
}

func.func @_QPt4(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}) {
// CHECK-LABEL:   func.func @_QPt4(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"},
// CHECK-SAME:                     %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt4Ec1"} : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,?>>>)
// CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) typeparams %[[VAL_7]]#1 dummy_scope %[[VAL_2]] {uniq_name = "_QFt4Ec2"} : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,?>>>)
// CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.array<3xi8> {bindc_name = "n", uniq_name = "_QFt4En"}
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_12]]) {uniq_name = "_QFt4En"} : (!fir.ref<!fir.array<3xi8>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi8>>, !fir.ref<!fir.array<3xi8>>)
// CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xi8> {
// CHECK:           ^bb0(%[[VAL_15:.*]]: index):
// CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_15]])  typeparams %[[VAL_3]]#1 : (!fir.box<!fir.array<3x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_15]])  typeparams %[[VAL_7]]#1 : (!fir.box<!fir.array<3x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:             %[[VAL_18:.*]]:2 = fir.unboxchar %[[VAL_17]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:             %[[VAL_19:.*]]:2 = fir.unboxchar %[[VAL_16]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %2 = fir.convert %1#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
    %c3 = arith.constant 3 : index
    %3 = fir.shape %c3 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %2(%3) typeparams %1#1 dummy_scope %0 {uniq_name = "_QFt4Ec1"} : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,?>>>)
    %5:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %6 = fir.convert %5#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
    %c3_0 = arith.constant 3 : index
    %7 = fir.shape %c3_0 : (index) -> !fir.shape<1>
    %8:2 = hlfir.declare %6(%7) typeparams %5#1 dummy_scope %0 {uniq_name = "_QFt4Ec2"} : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<3x!fir.char<1,?>>>, !fir.ref<!fir.array<3x!fir.char<1,?>>>)
    %c3_1 = arith.constant 3 : index
    %9 = fir.alloca !fir.array<3xi8> {bindc_name = "n", uniq_name = "_QFt4En"}
    %10 = fir.shape %c3_1 : (index) -> !fir.shape<1>
    %11:2 = hlfir.declare %9(%10) {uniq_name = "_QFt4En"} : (!fir.ref<!fir.array<3xi8>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi8>>, !fir.ref<!fir.array<3xi8>>)
    %12 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<3xi8> {
    ^bb0(%arg2: index):
      %13 = hlfir.designate %4#0 (%arg2)  typeparams %1#1 : (!fir.box<!fir.array<3x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
      %14 = hlfir.designate %8#0 (%arg2)  typeparams %5#1 : (!fir.box<!fir.array<3x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
      %15 = hlfir.index %14 in %13 : (!fir.boxchar<1>, !fir.boxchar<1>) -> i8
// CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_18]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_7]]#1 : (index) -> i64
// CHECK:             %[[VAL_24:.*]] = fir.call @_FortranAIndex1(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_0]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> i8
// CHECK:             hlfir.yield_element %[[VAL_25]] : i8
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_13]]#0 : !hlfir.expr<3xi8>, !fir.ref<!fir.array<3xi8>>
// CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<3xi8>
      hlfir.yield_element %15 : i8
    }
    hlfir.assign %12 to %11#0 : !hlfir.expr<3xi8>, !fir.ref<!fir.array<3xi8>>
    hlfir.destroy %12 : !hlfir.expr<3xi8>
    return
}
