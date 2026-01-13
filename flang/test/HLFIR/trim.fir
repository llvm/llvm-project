// RUN: fir-opt --lower-hlfir-intrinsics %s | FileCheck %s

// CHECK-LABEL:   func.func @_QPtrim_test(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_5]] {uniq_name = "_QFtrim_testEc"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.char<1,8> {bindc_name = "tc", uniq_name = "_QFtrim_testEtc"}
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_3]] {uniq_name = "_QFtrim_testEtc"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
// CHECK:           %[[VAL_10:.*]] = fir.embox %[[VAL_7]]#1 typeparams %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
// CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
// CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_11]] typeparams %[[VAL_2]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
// CHECK:           fir.store %[[VAL_12]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranATrim(%[[VAL_14]], %[[VAL_15]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
// CHECK:           %[[VAL_18:.*]] = fir.box_elesize %[[VAL_17]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
// CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]] typeparams %[[VAL_18]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           %[[VAL_21:.*]] = hlfir.as_expr %[[VAL_20]]#0 move %[[VAL_0]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_9]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.char<1,8>>
// CHECK:           hlfir.destroy %[[VAL_21]] : !hlfir.expr<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

func.func @_QPtrim_test(%arg0: !fir.boxchar<1> {fir.bindc_name = "c"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 {uniq_name = "_QFtrim_testEc"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %c8 = arith.constant 8 : index
  %3 = fir.alloca !fir.char<1,8> {bindc_name = "tc", uniq_name = "_QFtrim_testEtc"}
  %4:2 = hlfir.declare %3 typeparams %c8 {uniq_name = "_QFtrim_testEtc"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
  %5 = hlfir.char_trim %2#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %5 to %4#0 : !hlfir.expr<!fir.char<1,?>>, !fir.ref<!fir.char<1,8>>
  hlfir.destroy %5 : !hlfir.expr<!fir.char<1,?>>
  return
}
