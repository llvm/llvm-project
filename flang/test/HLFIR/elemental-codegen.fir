// Test hlfir.elemental code generation
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

func.func @numeric_type(%arg0: !fir.ref<!fir.array<10x20xi32>>, %arg1: !fir.ref<!fir.array<10x20xi32>>) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %1 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
  ^bb0(%arg2: index, %arg3: index):
    %2 = hlfir.designate %arg0 (%arg2, %arg3)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
    %3 = hlfir.designate %arg1 (%arg2, %arg3)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
    %4 = fir.load %2 : !fir.ref<i32>
    %5 = fir.load %3 : !fir.ref<i32>
    %6 = arith.addi %4, %5 : i32
    hlfir.yield_element %6 : i32
  }
  return
}
// CHECK-LABEL:   func.func @numeric_type(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.ref<!fir.array<10x20xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.array<10x20xi32>>) {
// CHECK:    %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:    %[[VAL_3:.*]] = arith.constant 20 : index
// CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:    %[[VAL_5:.*]] = fir.allocmem !fir.array<10x20xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:    %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_4]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<10x20xi32>>, !fir.heap<!fir.array<10x20xi32>>)
// CHECK:    %[[VAL_7:.*]] = arith.constant true
// CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:    fir.do_loop %[[VAL_9:.*]] = %[[VAL_8]] to %[[VAL_3]] step %[[VAL_8]] {
// CHECK:      fir.do_loop %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_2]] step %[[VAL_8]] {
// CHECK:        %[[VAL_11:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_10]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:        %[[VAL_12:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_10]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:        %[[VAL_13:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
// CHECK:        %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
// CHECK:        %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:        %[[VAL_16:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_10]], %[[VAL_9]])  : (!fir.heap<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:        hlfir.assign %[[VAL_15]] to %[[VAL_16]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:      }
// CHECK:    }
// CHECK:    %[[VAL_17:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_18:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_7]], [1 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_19:.*]] = fir.insert_value %[[VAL_18]], %[[VAL_6]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, !fir.heap<!fir.array<10x20xi32>>) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>


func.func @char_type(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>, %arg1: index, %arg2: index, %arg3: index) {
  %0 = fir.shape %arg1 : (index) -> !fir.shape<1>
  %1 = hlfir.elemental %0 typeparams %arg2 : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
  ^bb0(%arg4: index):
    %2 = hlfir.designate %arg0 (%arg4)  typeparams %arg3 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
    %3 = hlfir.concat %2, %2 len %arg2 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
    hlfir.yield_element %3 : !hlfir.expr<!fir.char<1,?>>
  }
  return
}
// CHECK-LABEL:   func.func @char_type(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_2:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_3:[^:]*]]: index) {
// CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:    %[[VAL_5:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_2]] : index), %[[VAL_1]] {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:    %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_4]]) typeparams %[[VAL_2]] {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>)
// CHECK:    %[[VAL_7:.*]] = arith.constant true
// CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:    fir.do_loop %[[VAL_9:.*]] = %[[VAL_8]] to %[[VAL_1]] step %[[VAL_8]] {
// CHECK:      %[[VAL_10:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_9]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
               // concatenation
// CHECK:      %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_14:.*]] typeparams %[[VAL_13:.*]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:      %[[VAL_31:.*]] = arith.constant false
// CHECK:      %[[VAL_32:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:      %[[VAL_33:.*]] = fir.insert_value %[[VAL_32]], %[[VAL_31]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:      %[[VAL_34:.*]] = fir.insert_value %[[VAL_33]], %[[VAL_30]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:      %[[VAL_35:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_9]])  typeparams %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:      hlfir.assign %[[VAL_30]]#0 to %[[VAL_35]] temporary_lhs : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:    }
// CHECK:    %[[VAL_36:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>
// CHECK:    %[[VAL_37:.*]] = fir.insert_value %[[VAL_36]], %[[VAL_7]], [1 : index] : (tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>, i1) -> tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>
// CHECK:    %[[VAL_38:.*]] = fir.insert_value %[[VAL_37]], %[[VAL_6]]#0, [0 : index] : (tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>, !fir.box<!fir.array<?x!fir.char<1,?>>>) -> tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>


func.func @derived_transpose(%arg0: !fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, %arg1: index, %arg2: index) {
  %0 = fir.shape %arg2, %arg1 : (index, index) -> !fir.shape<2>
  %1 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.type<t{field:f32}>> {
  ^bb0(%arg4: index, %arg5: index):
    %2 = hlfir.designate %arg0 (%arg4, %arg5) : (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, index, index) -> !fir.box<!fir.type<t{field:f32}>>
    %3 = hlfir.as_expr %2 : (!fir.box<!fir.type<t{field:f32}>>) -> !hlfir.expr<!fir.type<t{field:f32}>>
    hlfir.yield_element %3 : !hlfir.expr<!fir.type<t{field:f32}>>
  }
  return
}
// CHECK-LABEL:   func.func @derived_transpose(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_2:.*]]: index) {
// CHECK:    %[[VAL_3:.*]] = fir.shape %[[VAL_2]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:    %[[VAL_4:.*]] = fir.allocmem !fir.array<?x?x!fir.type<t{field:f32}>>, %[[VAL_2]], %[[VAL_1]] {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_3]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?x?x!fir.type<t{field:f32}>>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, !fir.heap<!fir.array<?x?x!fir.type<t{field:f32}>>>)
// CHECK:    %[[VAL_6:.*]] = arith.constant true
// CHECK:    %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:    fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_1]] step %[[VAL_7]] {
// CHECK:      fir.do_loop %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_2]] step %[[VAL_7]] {
// CHECK:        %[[VAL_10:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_9]], %[[VAL_8]])  : (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, index, index) -> !fir.box<!fir.type<t{field:f32}>>
// CHECK:        %[[VAL_11:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]], %[[VAL_8]])  : (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, index, index) -> !fir.ref<!fir.type<t{field:f32}>>
// CHECK:        hlfir.assign %[[VAL_10]] to %[[VAL_11]] temporary_lhs : !fir.box<!fir.type<t{field:f32}>>, !fir.ref<!fir.type<t{field:f32}>>
// CHECK:      }
// CHECK:    }
// CHECK:    %[[VAL_12:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>
// CHECK:    %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_6]], [1 : index] : (tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>, i1) -> tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>
// CHECK:    %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_5]]#0, [0 : index] : (tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>, !fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>) -> tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>

func.func @unordered() {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %3 = hlfir.elemental %0 unordered : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
  ^bb0(%i: index, %j: index):
    %c0 = arith.constant 0 : i32
    hlfir.yield_element %c0 : i32
  }
  return
}
// CHECK-LABEL:   func.func @unordered() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_3:.*]] = fir.allocmem !fir.array<10x20xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_2]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<10x20xi32>>, !fir.heap<!fir.array<10x20xi32>>)
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_6]] to %[[VAL_1]] step %[[VAL_6]] unordered {
// CHECK:             fir.do_loop %[[VAL_8:.*]] = %[[VAL_6]] to %[[VAL_0]] step %[[VAL_6]] unordered {
// CHECK:               %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_8]], %[[VAL_7]])  : (!fir.heap<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_9]] to %[[VAL_10]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:           %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_5]], [1 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:           %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, !fir.heap<!fir.array<10x20xi32>>) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:           return
// CHECK:         }

func.func @test_polymorphic(%arg0: !fir.class<!fir.type<_QMtypesTt>> {fir.bindc_name = "x"}, %arg1: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "y"}, %ex0 : index, %ex1 : index) {
  %1:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEx"} : (!fir.class<!fir.type<_QMtypesTt>>) -> (!fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>)
  %2:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEy"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
  %4 = fir.shape %ex0, %ex1 : (index, index) -> !fir.shape<2>
  %5 = hlfir.elemental %4 mold %1#0 unordered : (!fir.shape<2>, !fir.class<!fir.type<_QMtypesTt>>) -> !hlfir.expr<?x?x!fir.type<_QMtypesTt>?> {
  ^bb0(%arg3: index, %arg4: index):
    %6 = hlfir.designate %2#0 (%arg3, %arg4)  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
    %7 = hlfir.as_expr %6 : (!fir.class<!fir.type<_QMtypesTt>>) -> !hlfir.expr<!fir.type<_QMtypesTt>?>
    hlfir.yield_element %7 : !hlfir.expr<!fir.type<_QMtypesTt>?>
  }
  return
}
// CHECK-LABEL:   func.func @test_polymorphic(
// CHECK-SAME:        %[[VAL_0:.*]]: !fir.class<!fir.type<_QMtypesTt>> {fir.bindc_name = "x"},
// CHECK-SAME:        %[[VAL_1:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "y"},
// CHECK-SAME:        %[[EX0:.*]]: index,
// CHECK-SAME:        %[[EX1:.*]]: index) {
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEx"} : (!fir.class<!fir.type<_QMtypesTt>>) -> (!fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEy"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[EX0]], %[[EX1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_8:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_8]](%[[VAL_10]]) : (!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_11]] to %[[VAL_4]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_12:.*]] = arith.constant true
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = ".tmp.array"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>)
// CHECK:           %[[RANK:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.class<!fir.type<_QMtypesTt>>) -> !fir.box<none>
// CHECK:           %[[VAL_17:.*]] = fir.call @_FortranAAllocatableApplyMold(%[[VAL_15]], %[[VAL_16]], %[[RANK]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_18]] : (index) -> i64
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[EX0]] : (index) -> i64
// CHECK:           %[[VAL_24:.*]] = fir.call @_FortranAAllocatableSetBounds(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
// CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_18]] : (index) -> i64
// CHECK:           %[[VAL_29:.*]] = fir.convert %[[EX1]] : (index) -> i64
// CHECK:           %[[VAL_30:.*]] = fir.call @_FortranAAllocatableSetBounds(%[[VAL_26]], %[[VAL_27]], %[[VAL_28]], %[[VAL_29]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
// CHECK:           %[[VAL_31:.*]] = fir.address_of(@_QQcl.
// CHECK:           %[[VAL_32:.*]] = arith.constant {{.*}} : index
// CHECK:           %[[VAL_33:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_34:.*]] = arith.constant false
// CHECK:           %[[VAL_35:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_38:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_36]], %[[VAL_34]], %[[VAL_35]], %[[VAL_37]], %[[VAL_33]]) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_40:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_41:.*]] = %[[VAL_40]] to %[[EX1]] step %[[VAL_40]] unordered {
// CHECK:             fir.do_loop %[[VAL_42:.*]] = %[[VAL_40]] to %[[EX0]] step %[[VAL_40]] unordered {
// CHECK:               %[[VAL_43:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_42]], %[[VAL_41]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_44:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_45:.*]]:3 = fir.box_dims %[[VAL_39]], %[[VAL_44]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_46:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_47:.*]]:3 = fir.box_dims %[[VAL_39]], %[[VAL_46]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_48:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_49:.*]] = arith.subi %[[VAL_45]]#0, %[[VAL_48]] : index
// CHECK:               %[[VAL_50:.*]] = arith.addi %[[VAL_42]], %[[VAL_49]] : index
// CHECK:               %[[VAL_51:.*]] = arith.subi %[[VAL_47]]#0, %[[VAL_48]] : index
// CHECK:               %[[VAL_52:.*]] = arith.addi %[[VAL_41]], %[[VAL_51]] : index
// CHECK:               %[[VAL_53:.*]] = hlfir.designate %[[VAL_39]] (%[[VAL_50]], %[[VAL_52]])  : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               hlfir.assign %[[VAL_43]] to %[[VAL_53]] temporary_lhs : !fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = fir.undefined tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           %[[VAL_55:.*]] = fir.insert_value %[[VAL_54]], %[[VAL_12]], [1 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>, i1) -> tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           %[[VAL_56:.*]] = fir.insert_value %[[VAL_55]], %[[VAL_39]], [0 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>, !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>) -> tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           return
// CHECK:         }

// Test that hlfir.expr mold is properly applied for the second hlfir.elemental.
func.func @test_polymorphic_expr(%arg0: !fir.class<!fir.type<_QMtypesTt>> {fir.bindc_name = "x"}, %arg1: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "y"}, %ex0 : index, %ex1 : index) {
  %1:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEx"} : (!fir.class<!fir.type<_QMtypesTt>>) -> (!fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>)
  %2:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEy"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
  %4 = fir.shape %ex0, %ex1 : (index, index) -> !fir.shape<2>
  %5 = hlfir.elemental %4 mold %1#0 unordered : (!fir.shape<2>, !fir.class<!fir.type<_QMtypesTt>>) -> !hlfir.expr<?x?x!fir.type<_QMtypesTt>?> {
  ^bb0(%arg3: index, %arg4: index):
    %6 = hlfir.designate %2#0 (%arg3, %arg4)  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
    %7 = hlfir.as_expr %6 : (!fir.class<!fir.type<_QMtypesTt>>) -> !hlfir.expr<!fir.type<_QMtypesTt>?>
    hlfir.yield_element %7 : !hlfir.expr<!fir.type<_QMtypesTt>?>
  }
  %8 = hlfir.elemental %4 mold %5 unordered : (!fir.shape<2>, !hlfir.expr<?x?x!fir.type<_QMtypesTt>?>) -> !hlfir.expr<?x?x!fir.type<_QMtypesTt>?> {
  ^bb0(%arg3: index, %arg4: index):
    %9 = hlfir.apply %5, %arg3, %arg4 : (!hlfir.expr<?x?x!fir.type<_QMtypesTt>?>, index, index) -> !hlfir.expr<!fir.type<_QMtypesTt>?>
    hlfir.yield_element %9 : !hlfir.expr<!fir.type<_QMtypesTt>?>
  }
  return
}
// CHECK-LABEL:   func.func @test_polymorphic_expr(
// CHECK-SAME:        %[[VAL_0:.*]]: !fir.class<!fir.type<_QMtypesTt>> {fir.bindc_name = "x"},
// CHECK-SAME:        %[[VAL_1:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "y"},
// CHECK-SAME:        %[[VAL_2:.*]]: index,
// CHECK-SAME:        %[[VAL_3:.*]]: index) {
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEx"} : (!fir.class<!fir.type<_QMtypesTt>>) -> (!fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEy"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_9]](%[[VAL_11]]) : (!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_12]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_13:.*]] = arith.constant true
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_5]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = ".tmp.array"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>)
// CHECK:           %[[VAL_15:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.class<!fir.type<_QMtypesTt>>) -> !fir.box<none>
// CHECK:           %[[VAL_18:.*]] = fir.call @_FortranAAllocatableApplyMold(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (index) -> i32
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
// CHECK:           %[[VAL_25:.*]] = fir.call @_FortranAAllocatableSetBounds(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
// CHECK:           %[[VAL_26:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_26]] : (index) -> i32
// CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
// CHECK:           %[[VAL_31:.*]] = fir.call @_FortranAAllocatableSetBounds(%[[VAL_27]], %[[VAL_28]], %[[VAL_29]], %[[VAL_30]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
// CHECK:           %[[VAL_32:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_33:.*]] = arith.constant {{.*}} : index
// CHECK:           %[[VAL_34:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_35:.*]] = arith.constant false
// CHECK:           %[[VAL_36:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_32]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_39:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_37]], %[[VAL_35]], %[[VAL_36]], %[[VAL_38]], %[[VAL_34]]) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_41:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_42:.*]] = %[[VAL_41]] to %[[VAL_3]] step %[[VAL_41]] unordered {
// CHECK:             fir.do_loop %[[VAL_43:.*]] = %[[VAL_41]] to %[[VAL_2]] step %[[VAL_41]] unordered {
// CHECK:               %[[VAL_44:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_43]], %[[VAL_42]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_45:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_46:.*]]:3 = fir.box_dims %[[VAL_40]], %[[VAL_45]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_47:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_48:.*]]:3 = fir.box_dims %[[VAL_40]], %[[VAL_47]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_49:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_50:.*]] = arith.subi %[[VAL_46]]#0, %[[VAL_49]] : index
// CHECK:               %[[VAL_51:.*]] = arith.addi %[[VAL_43]], %[[VAL_50]] : index
// CHECK:               %[[VAL_52:.*]] = arith.subi %[[VAL_48]]#0, %[[VAL_49]] : index
// CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_42]], %[[VAL_52]] : index
// CHECK:               %[[VAL_54:.*]] = hlfir.designate %[[VAL_40]] (%[[VAL_51]], %[[VAL_53]])  : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               hlfir.assign %[[VAL_44]] to %[[VAL_54]] temporary_lhs : !fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = fir.undefined tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           %[[VAL_56:.*]] = fir.insert_value %[[VAL_55]], %[[VAL_13]], [1 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>, i1) -> tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           %[[VAL_57:.*]] = fir.insert_value %[[VAL_56]], %[[VAL_40]], [0 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>, !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>) -> tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           %[[VAL_58:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_59:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_60:.*]] = fir.shape %[[VAL_59]], %[[VAL_59]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_61:.*]] = fir.embox %[[VAL_58]](%[[VAL_60]]) : (!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.shape<2>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_61]] to %[[VAL_4]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_62:.*]] = arith.constant true
// CHECK:           %[[VAL_63:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = ".tmp.array"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>)
// CHECK:           %[[VAL_64:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_65:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_66:.*]] = fir.convert %[[VAL_40]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>) -> !fir.box<none>
// CHECK:           %[[VAL_67:.*]] = fir.call @_FortranAAllocatableApplyMold(%[[VAL_65]], %[[VAL_66]], %[[VAL_64]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
// CHECK:           %[[VAL_68:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_69:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_70:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_71:.*]] = fir.convert %[[VAL_69]] : (index) -> i32
// CHECK:           %[[VAL_72:.*]] = fir.convert %[[VAL_68]] : (index) -> i64
// CHECK:           %[[VAL_73:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
// CHECK:           %[[VAL_74:.*]] = fir.call @_FortranAAllocatableSetBounds(%[[VAL_70]], %[[VAL_71]], %[[VAL_72]], %[[VAL_73]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
// CHECK:           %[[VAL_75:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_76:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_77:.*]] = fir.convert %[[VAL_75]] : (index) -> i32
// CHECK:           %[[VAL_78:.*]] = fir.convert %[[VAL_68]] : (index) -> i64
// CHECK:           %[[VAL_79:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
// CHECK:           %[[VAL_80:.*]] = fir.call @_FortranAAllocatableSetBounds(%[[VAL_76]], %[[VAL_77]], %[[VAL_78]], %[[VAL_79]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
// CHECK:           %[[VAL_81:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_82:.*]] = arith.constant {{.*}} : index
// CHECK:           %[[VAL_83:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_84:.*]] = arith.constant false
// CHECK:           %[[VAL_85:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_86:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_87:.*]] = fir.convert %[[VAL_81]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_88:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_86]], %[[VAL_84]], %[[VAL_85]], %[[VAL_87]], %[[VAL_83]]) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK:           %[[VAL_89:.*]] = fir.load %[[VAL_63]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_90:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_91:.*]] = %[[VAL_90]] to %[[VAL_3]] step %[[VAL_90]] unordered {
// CHECK:             fir.do_loop %[[VAL_92:.*]] = %[[VAL_90]] to %[[VAL_2]] step %[[VAL_90]] unordered {
// CHECK:               %[[VAL_93:.*]] = hlfir.designate %[[VAL_40]] (%[[VAL_92]], %[[VAL_91]])  : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_94:.*]] = arith.constant false
// CHECK:               %[[VAL_95:.*]] = fir.undefined tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>
// CHECK:               %[[VAL_96:.*]] = fir.insert_value %[[VAL_95]], %[[VAL_94]], [1 : index] : (tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>, i1) -> tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>
// CHECK:               %[[VAL_97:.*]] = fir.insert_value %[[VAL_96]], %[[VAL_93]], [0 : index] : (tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>, !fir.class<!fir.type<_QMtypesTt>>) -> tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>
// CHECK:               %[[VAL_98:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_99:.*]]:3 = fir.box_dims %[[VAL_89]], %[[VAL_98]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_100:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_101:.*]]:3 = fir.box_dims %[[VAL_89]], %[[VAL_100]] : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_102:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_103:.*]] = arith.subi %[[VAL_99]]#0, %[[VAL_102]] : index
// CHECK:               %[[VAL_104:.*]] = arith.addi %[[VAL_92]], %[[VAL_103]] : index
// CHECK:               %[[VAL_105:.*]] = arith.subi %[[VAL_101]]#0, %[[VAL_102]] : index
// CHECK:               %[[VAL_106:.*]] = arith.addi %[[VAL_91]], %[[VAL_105]] : index
// CHECK:               %[[VAL_107:.*]] = hlfir.designate %[[VAL_89]] (%[[VAL_104]], %[[VAL_106]])  : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               hlfir.assign %[[VAL_93]] to %[[VAL_107]] temporary_lhs : !fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_108:.*]] = fir.undefined tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           %[[VAL_109:.*]] = fir.insert_value %[[VAL_108]], %[[VAL_62]], [1 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>, i1) -> tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           %[[VAL_110:.*]] = fir.insert_value %[[VAL_109]], %[[VAL_89]], [0 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>, !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>) -> tuple<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>, i1>
// CHECK:           return
// CHECK:         }
