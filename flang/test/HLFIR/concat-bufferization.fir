// Test hlfir.concat operation lowering to operations operating on memory.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s


func.func @concat(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = arith.addi %2#1, %4#1 : index
  %7 = hlfir.concat %3#0, %5#0 len %6 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %7 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @concat(
// CHECK-SAME:                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                      %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                      %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>) {
// CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]]#0 typeparams %[[VAL_7]]#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_7]]#1 : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_7]]#1 : index
// CHECK:           %[[VAL_11:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_10]] : index) {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_10]], %[[VAL_17]] : index
// CHECK:           fir.do_loop %[[VAL_19:.*]] = %[[VAL_5]]#1 to %[[VAL_18]] step %[[VAL_17]] {
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_5]]#1 : index
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_21]], %[[VAL_20]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_19]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[VAL_23]] to %[[VAL_25]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_11]] typeparams %[[VAL_10]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_27:.*]] = arith.constant false
// CHECK:           %[[VAL_28:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_29:.*]] = fir.insert_value %[[VAL_28]], %[[VAL_27]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_30:.*]] = fir.insert_value %[[VAL_29]], %[[VAL_26]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[VAL_26]]#0 to %[[VAL_4]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           return
// CHECK:         }


func.func @concat_chained(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>, %arg3: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 {uniq_name = "c4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = arith.addi %2#1, %4#1 : index
  %9 = hlfir.concat %3#0, %5#0 len %8 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %10 = arith.addi %8, %6#1 : index
  %11 = hlfir.concat %9, %7#0 len %10 : (!hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %11 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  return
}

// CHECK-LABEL:   func.func @concat_chained(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                              %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                              %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>) {
// CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]]#0 typeparams %[[VAL_8]]#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_10:.*]]:2 = fir.unboxchar %[[VAL_3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]]#0 typeparams %[[VAL_10]]#1 {uniq_name = "c4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_8]]#1 : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_8]]#1 : index
// CHECK:           %[[VAL_14:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_13]] : index) {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_18]], %[[VAL_19]], %[[VAL_17]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_13]], %[[VAL_20]] : index
// CHECK:           fir.do_loop %[[VAL_22:.*]] = %[[VAL_6]]#1 to %[[VAL_21]] step %[[VAL_20]] {
// CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_6]]#1 : index
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_23]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[VAL_28:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_22]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[VAL_26]] to %[[VAL_28]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_14]] typeparams %[[VAL_13]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_30:.*]] = arith.constant false
// CHECK:           %[[VAL_31:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_32:.*]] = fir.insert_value %[[VAL_31]], %[[VAL_30]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_33:.*]] = fir.insert_value %[[VAL_32]], %[[VAL_29]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_34:.*]] = arith.addi %[[VAL_12]], %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_35:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_36:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_35]] : index) {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_37:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
// CHECK:           %[[VAL_39:.*]] = arith.muli %[[VAL_37]], %[[VAL_38]] : i64
// CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_29]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_40]], %[[VAL_41]], %[[VAL_39]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_42:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_43:.*]] = arith.subi %[[VAL_35]], %[[VAL_42]] : index
// CHECK:           fir.do_loop %[[VAL_44:.*]] = %[[VAL_13]] to %[[VAL_43]] step %[[VAL_42]] {
// CHECK:             %[[VAL_45:.*]] = arith.subi %[[VAL_44]], %[[VAL_13]] : index
// CHECK:             %[[VAL_46:.*]] = fir.convert %[[VAL_11]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[VAL_47:.*]] = fir.coordinate_of %[[VAL_46]], %[[VAL_45]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_48:.*]] = fir.load %[[VAL_47]] : !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[VAL_50:.*]] = fir.coordinate_of %[[VAL_49]], %[[VAL_44]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[VAL_48]] to %[[VAL_50]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[VAL_51:.*]]:2 = hlfir.declare %[[VAL_36]] typeparams %[[VAL_35]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_52:.*]] = arith.constant false
// CHECK:           %[[VAL_53:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_54:.*]] = fir.insert_value %[[VAL_53]], %[[VAL_52]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_55:.*]] = fir.insert_value %[[VAL_54]], %[[VAL_51]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[VAL_51]]#0 to %[[VAL_5]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           return
// CHECK:         }
