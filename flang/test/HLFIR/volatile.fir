// RUN: fir-opt --strict-fir-volatile-verifier --convert-hlfir-to-fir %s -o - | FileCheck %s

func.func @foo() {
  %true = arith.constant true
  %false = arith.constant false
  %0 = fir.alloca !fir.logical<4> {bindc_name = "a", uniq_name = "_QFEa"}
  %1 = fir.volatile_cast %0 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>, volatile>
  %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEa"} : (!fir.ref<!fir.logical<4>, volatile>) -> (!fir.ref<!fir.logical<4>, volatile>, !fir.ref<!fir.logical<4>, volatile>)
  %3 = fir.alloca !fir.logical<4> {bindc_name = "b", uniq_name = "_QFEb"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFEb"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %5 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.convert %false : (i1) -> !fir.logical<4>
  hlfir.assign %7 to %2#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
  %8 = fir.load %2#0 : !fir.ref<!fir.logical<4>, volatile>
  hlfir.assign %8 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  %9 = fir.convert %true : (i1) -> !fir.logical<4>
  hlfir.assign %9 to %2#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
  return
}
// CHECK-LABEL:   func.func @foo() {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.logical<4> {bindc_name = "a", uniq_name = "_QFEa"}
// CHECK:           %[[VAL_3:.*]] = fir.volatile_cast %[[VAL_2]] : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>, volatile>
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEa"} : (!fir.ref<!fir.logical<4>, volatile>) -> !fir.ref<!fir.logical<4>, volatile>
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.logical<4> {bindc_name = "b", uniq_name = "_QFEb"}
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_5]] {uniq_name = "_QFEb"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
// CHECK:           %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_7]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (i1) -> !fir.logical<4>
// CHECK:           fir.store %[[VAL_9]] to %[[VAL_4]] : !fir.ref<!fir.logical<4>, volatile>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.logical<4>, volatile>
// CHECK:           fir.store %[[VAL_10]] to %[[VAL_6]] : !fir.ref<!fir.logical<4>>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (i1) -> !fir.logical<4>
// CHECK:           fir.store %[[VAL_11]] to %[[VAL_4]] : !fir.ref<!fir.logical<4>, volatile>
// CHECK:           return
// CHECK:         }

// -----

func.func private @_QFPassign_different_length(%arg0: !fir.boxchar<1> {fir.bindc_name = "string"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %2 = fir.convert %1#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,3>>
    %3 = fir.volatile_cast %2 : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
    %4:2 = hlfir.declare %3 typeparams %c3 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_inout, volatile>, uniq_name = "_QFFassign_different_lengthEstring"} : (!fir.ref<!fir.char<1,3>, volatile>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,3>, volatile>, !fir.ref<!fir.char<1,3>, volatile>)
    %5 = fir.address_of(@_QQclX626F) : !fir.ref<!fir.char<1,2>>
    %6:2 = hlfir.declare %5 typeparams %c2 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX626F"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
    hlfir.assign %6#0 to %4#0 : !fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,3>, volatile>
    return
}
// CHECK-LABEL:   func.func private @_QFPassign_different_length(
// CHECK-SAME:                                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1> {fir.bindc_name = "string"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,3>>
// CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_5]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.char<1,3>, volatile>
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_6]] typeparams %[[VAL_2]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<intent_inout, volatile>, uniq_name = "_QFFassign_different_lengthEstring"} : (!fir.ref<!fir.char<1,3>, volatile>, index, !fir.dscope) -> !fir.ref<!fir.char<1,3>, volatile>
// CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QQclX626F) : !fir.ref<!fir.char<1,2>>
// CHECK:           %[[VAL_9:.*]] = fir.declare %[[VAL_8]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX626F"} : (!fir.ref<!fir.char<1,2>>, index) -> !fir.ref<!fir.char<1,2>>
// CHECK:           %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_1]] : index
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_2]], %[[VAL_1]] : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
// CHECK:           %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,3>, volatile>) -> !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.char<1,2>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]) <{isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_2]], %[[VAL_17]] : index
// CHECK:           %[[VAL_19:.*]] = arith.constant 32 : i8
// CHECK:           %[[VAL_20:.*]] = fir.undefined !fir.char<1>
// CHECK:           %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_19]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:           %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_23:.*]] = %[[VAL_11]] to %[[VAL_18]] step %[[VAL_22]] {
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,3>, volatile>) -> !fir.ref<!fir.array<3x!fir.char<1>>, volatile>
// CHECK:             %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_23]] : (!fir.ref<!fir.array<3x!fir.char<1>>, volatile>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[VAL_21]] to %[[VAL_25]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
