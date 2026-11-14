// Test hlfir.concat operation lowering to operations operating on memory.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s


func.func @concat(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = arith.addi %2#1, %4#1 : index
  %7 = hlfir.concat %3#0, %5#0 len %6 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %7 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %7 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @concat(
// CHECK-SAME:                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                      %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                      %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>) {
// CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]]#0 typeparams %[[VAL_7]]#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_7]]#1 : index
// CHECK:           %[[VAL_11:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_9]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[VAL_58:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_59:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_60:.*]] = arith.addi %[[VAL_58]], %[[VAL_59]] : index
// CHECK:           %[[VAL_61:.*]] = arith.subi %[[VAL_60]], %[[VAL_59]] : index
// CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_11]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[VAL_63:.*]] = fir.coordinate_of %[[VAL_62]], %[[VAL_61]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_64]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_65:.*]] = arith.addi %[[VAL_58]], %[[VAL_5]]#1 : index
// CHECK:           %[[VAL_66:.*]] = arith.addi %[[VAL_65]], %[[VAL_59]] : index
// CHECK:           %[[VAL_67:.*]] = arith.subi %[[VAL_66]], %[[VAL_59]] : index
// CHECK:           %[[VAL_68:.*]] = fir.convert %[[VAL_11]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[VAL_69:.*]] = fir.coordinate_of %[[VAL_68]], %[[VAL_67]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_70:.*]] = fir.convert %[[VAL_69]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[VAL_71:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_72:.*]] = fir.convert %[[VAL_7]]#1 : (index) -> i64
// CHECK:           %[[VAL_73:.*]] = arith.muli %[[VAL_71]], %[[VAL_72]] : i64
// CHECK:           %[[VAL_74:.*]] = fir.convert %[[VAL_70]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_75:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_74]], %[[VAL_75]], %[[VAL_73]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_11_REF:.*]] = fir.convert %[[VAL_11]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_11_REF]] typeparams %[[VAL_9]] uniq_name("tmp") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_27:.*]] = arith.constant true
// CHECK:           %[[VAL_28:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_29:.*]] = fir.insert_value %[[VAL_28]], %[[VAL_27]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_30:.*]] = fir.insert_value %[[VAL_29]], %[[VAL_26]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[VAL_26]]#0 to %[[VAL_4]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[VAL_31:.*]] = fir.box_addr %[[VAL_26]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[VAL_31]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }


func.func @concat_chained(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>, %arg3: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 uniq_name("c4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = arith.addi %2#1, %4#1 : index
  %10 = arith.addi %8, %6#1 : index
  %11 = hlfir.concat %3#0, %5#0, %7#0 len %10 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %11 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %11 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL:   func.func @concat_chained(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                              %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>,
// CHECK-SAME:                              %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.boxchar<1>) {
// CHECK:           %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 uniq_name("c1") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 uniq_name("c2") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]]#0 typeparams %[[VAL_8]]#1 uniq_name("c3") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_10:.*]]:2 = fir.unboxchar %[[VAL_3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]]#0 typeparams %[[VAL_10]]#1 uniq_name("c4") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_8]]#1 : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_10]]#1 : index
// CHECK:           %[[VAL_14:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_13]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[VAL_58:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_59:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_60:.*]] = arith.addi %[[VAL_58]], %[[VAL_59]] : index
// CHECK:           %[[VAL_61:.*]] = arith.subi %[[VAL_60]], %[[VAL_59]] : index
// CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_14]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[VAL_63:.*]] = fir.coordinate_of %[[VAL_62]], %[[VAL_61]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_64]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_18]], %[[VAL_19]], %[[VAL_17]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_65:.*]] = arith.addi %[[VAL_58]], %[[VAL_6]]#1 : index
// CHECK:           %[[VAL_66:.*]] = arith.addi %[[VAL_65]], %[[VAL_59]] : index
// CHECK:           %[[VAL_67:.*]] = arith.subi %[[VAL_66]], %[[VAL_59]] : index
// CHECK:           %[[VAL_68:.*]] = fir.convert %[[VAL_14]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[VAL_69:.*]] = fir.coordinate_of %[[VAL_68]], %[[VAL_67]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_70:.*]] = fir.convert %[[VAL_69]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[VAL_71:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_72:.*]] = fir.convert %[[VAL_8]]#1 : (index) -> i64
// CHECK:           %[[VAL_73:.*]] = arith.muli %[[VAL_71]], %[[VAL_72]] : i64
// CHECK:           %[[VAL_74:.*]] = fir.convert %[[VAL_70]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_75:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_74]], %[[VAL_75]], %[[VAL_73]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_76:.*]] = arith.addi %[[VAL_65]], %[[VAL_8]]#1 : index
// CHECK:           %[[VAL_77:.*]] = arith.addi %[[VAL_76]], %[[VAL_59]] : index
// CHECK:           %[[VAL_78:.*]] = arith.subi %[[VAL_77]], %[[VAL_59]] : index
// CHECK:           %[[VAL_79:.*]] = fir.convert %[[VAL_14]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:           %[[VAL_80:.*]] = fir.coordinate_of %[[VAL_79]], %[[VAL_78]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_81:.*]] = fir.convert %[[VAL_80]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[VAL_82:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_83:.*]] = fir.convert %[[VAL_10]]#1 : (index) -> i64
// CHECK:           %[[VAL_84:.*]] = arith.muli %[[VAL_82]], %[[VAL_83]] : i64
// CHECK:           %[[VAL_85:.*]] = fir.convert %[[VAL_81]] : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           %[[VAL_86:.*]] = fir.convert %[[VAL_11]]#1 : (!fir.ref<!fir.char<1,?>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%[[VAL_85]], %[[VAL_86]], %[[VAL_84]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_14_REF:.*]] = fir.convert %[[VAL_14]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_14_REF]] typeparams %[[VAL_13]] uniq_name("tmp") : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_30:.*]] = arith.constant true
// CHECK:           %[[VAL_31:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_32:.*]] = fir.insert_value %[[VAL_31]], %[[VAL_30]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_33:.*]] = fir.insert_value %[[VAL_32]], %[[VAL_29]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           hlfir.assign %[[VAL_29]]#0 to %[[VAL_5]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:           %[[VAL_57:.*]] = fir.box_addr %[[VAL_29]]#0 : (!fir.boxchar<1>) -> !fir.heap<!fir.char<1,?>>
// CHECK:           fir.freemem %[[VAL_57]] : !fir.heap<!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

// When every operand has a compile-time length, the buffer is on the stack.
func.func @concat_constant_len(%arg0: !fir.ref<!fir.char<1,3>>, %arg1: !fir.ref<!fir.char<1,2>>, %arg2: !fir.ref<!fir.char<1,5>>) {
  %c5 = arith.constant 5 : index
  %0 = hlfir.concat %arg0, %arg1 len %c5 : (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,2>>, index) -> !hlfir.expr<!fir.char<1,5>>
  hlfir.assign %0 to %arg2 : !hlfir.expr<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,5>>
  return
}
// CHECK-LABEL:   func.func @concat_constant_len(
// CHECK-SAME:                                   %[[ARG0:.*]]: !fir.ref<!fir.char<1,3>>, %[[ARG1:.*]]: !fir.ref<!fir.char<1,2>>, %[[ARG2:.*]]: !fir.ref<!fir.char<1,5>>) {
// CHECK-NOT:       fir.allocmem
// CHECK:           %[[TEMP:.*]] = fir.alloca !fir.char<1,5> <{bindc_name = ".chrtmp"}>
// CHECK:           %[[C3:.*]] = arith.constant 3 : index
// CHECK:           %[[LEN0:.*]] = fir.convert %[[C3]] : (index) -> i64
// CHECK:           %[[SIZE0:.*]] = arith.muli %{{.*}}, %[[LEN0]] : i64
// CHECK:           %[[SRC0:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.char<1,3>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%{{.*}}, %[[SRC0]], %[[SIZE0]])
// CHECK:           %[[OFFSET:.*]] = arith.addi %{{.*}}, %[[C3]] : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[LOWER:.*]] = arith.addi %[[OFFSET]], %{{.*}} : index
// CHECK:           %[[INDEX:.*]] = arith.subi %[[LOWER]], %{{.*}} : index
// CHECK:           %[[ARRAY:.*]] = fir.convert %[[TEMP]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.array<5x!fir.char<1>>>
// CHECK:           fir.coordinate_of %[[ARRAY]], %[[INDEX]]
// CHECK:           %[[LEN1:.*]] = fir.convert %[[C2]] : (index) -> i64
// CHECK:           %[[SIZE1:.*]] = arith.muli %{{.*}}, %[[LEN1]] : i64
// CHECK:           %[[SRC1:.*]] = fir.convert %[[ARG1]] : (!fir.ref<!fir.char<1,2>>) -> !llvm.ptr
// CHECK:           "llvm.intr.memmove"(%{{.*}}, %[[SRC1]], %[[SIZE1]])
// CHECK:           %[[RES:.*]]:2 = hlfir.declare %[[TEMP]] typeparams %{{.*}} uniq_name("tmp") : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           fir.insert_value %{{.*}}, %[[FALSE]], [1 : index]
// CHECK:           hlfir.assign %[[RES]]#0 to %[[ARG2]] : !fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>
// CHECK-NOT:       fir.freemem
// CHECK:           return

// Offsets are in characters, copy sizes are in bytes.
func.func @concat_kind4(%arg0: !fir.boxchar<4>, %arg1: !fir.boxchar<4>, %arg2: !fir.boxchar<4>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  %1:2 = fir.unboxchar %arg1 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  %2 = arith.addi %0#1, %1#1 : index
  %3 = hlfir.concat %arg0, %arg1 len %2 : (!fir.boxchar<4>, !fir.boxchar<4>, index) -> !hlfir.expr<!fir.char<4,?>>
  hlfir.assign %3 to %arg2 : !hlfir.expr<!fir.char<4,?>>, !fir.boxchar<4>
  hlfir.destroy %3 : !hlfir.expr<!fir.char<4,?>>
  return
}
// CHECK-LABEL:   func.func @concat_kind4(
// CHECK:           %[[LEN:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:           %[[TEMP:.*]] = fir.allocmem !fir.char<4,?>(%[[LEN]] : index) <{bindc_name = ".chrtmp", uniq_name = ""}>
// CHECK:           %[[STR0:.*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[ARRAY0:.*]] = fir.convert %[[TEMP]] : (!fir.heap<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
// CHECK:           fir.coordinate_of %[[ARRAY0]], %{{.*}} : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
// CHECK:           %[[KIND0:.*]] = arith.constant 4 : i64
// CHECK:           %[[LEN0:.*]] = fir.convert %[[STR0]]#1 : (index) -> i64
// CHECK:           %[[SIZE0:.*]] = arith.muli %[[KIND0]], %[[LEN0]] : i64
// CHECK:           "llvm.intr.memmove"(%{{.*}}, %{{.*}}, %[[SIZE0]])
// CHECK:           %[[OFFSET:.*]] = arith.addi %{{.*}}, %[[STR0]]#1 : index
// CHECK:           %[[STR1:.*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[LOWER:.*]] = arith.addi %[[OFFSET]], %{{.*}} : index
// CHECK:           %[[INDEX:.*]] = arith.subi %[[LOWER]], %{{.*}} : index
// CHECK:           %[[ARRAY1:.*]] = fir.convert %[[TEMP]] : (!fir.heap<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
// CHECK:           fir.coordinate_of %[[ARRAY1]], %[[INDEX]] : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
// CHECK:           %[[KIND1:.*]] = arith.constant 4 : i64
// CHECK:           %[[LEN1:.*]] = fir.convert %[[STR1]]#1 : (index) -> i64
// CHECK:           %[[SIZE1:.*]] = arith.muli %[[KIND1]], %[[LEN1]] : i64
// CHECK:           "llvm.intr.memmove"(%{{.*}}, %{{.*}}, %[[SIZE1]])
// CHECK:           hlfir.declare %{{.*}} typeparams %[[LEN]] uniq_name("tmp") : (!fir.ref<!fir.char<4,?>>, index) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
// CHECK:           fir.freemem
// CHECK:           return
