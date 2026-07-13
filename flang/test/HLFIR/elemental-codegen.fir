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
// CHECK-SAME:      %[[ARG0:.*]]: !fir.class<!fir.type<_QMtypesTt>> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "y"},
// CHECK-SAME:      %[[ARG2:.*]]: index,
// CHECK-SAME:      %[[ARG3:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEx"} : (!fir.class<!fir.type<_QMtypesTt>>) -> (!fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>)
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEy"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[ARG2]], %[[ARG3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_4]](%[[VAL_3]]) source_box %[[VAL_1]]#1 : (!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.shape<2>, !fir.class<!fir.type<_QMtypesTt>>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_6:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_10:.*]] = arith.constant false
// CHECK:           %[[VAL_11:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:           %[[VAL_15:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_12]], %[[VAL_13]], %[[VAL_10]], %[[VAL_11]],
// CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] {uniq_name = ".tmp.array"} : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_18:.*]] = arith.constant true
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_20:.*]] = %[[VAL_19]] to %[[ARG3]] step %[[VAL_19]] unordered {
// CHECK:             fir.do_loop %[[VAL_21:.*]] = %[[VAL_19]] to %[[ARG2]] step %[[VAL_19]] unordered {
// CHECK:               %[[VAL_22:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_21]], %[[VAL_20]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_17]]#0 (%[[VAL_21]], %[[VAL_20]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               hlfir.assign %[[VAL_22]] to %[[VAL_23]] temporary_lhs : !fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = fir.undefined tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           %[[VAL_25:.*]] = fir.insert_value %[[VAL_24]], %[[VAL_18]], [1 : index] : (tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>, i1) -> tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           %[[VAL_26:.*]] = fir.insert_value %[[VAL_25]], %[[VAL_17]]#0, [0 : index] : (tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
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
// CHECK-SAME:      %[[ARG0:.*]]: !fir.class<!fir.type<_QMtypesTt>> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "y"},
// CHECK-SAME:      %[[ARG2:.*]]: index,
// CHECK-SAME:      %[[ARG3:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEx"} : (!fir.class<!fir.type<_QMtypesTt>>) -> (!fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>)
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG1]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtestEy"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[ARG2]], %[[ARG3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]](%[[VAL_4]]) source_box %[[VAL_2]]#1 : (!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.shape<2>, !fir.class<!fir.type<_QMtypesTt>>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_7:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_11:.*]] = arith.constant false
// CHECK:           %[[VAL_12:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:           %[[VAL_16:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_13]], %[[VAL_14]], %[[VAL_11]], %[[VAL_12]],
// CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] {uniq_name = ".tmp.array"} : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_19:.*]] = arith.constant true
// CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_21:.*]] = %[[VAL_20]] to %[[ARG3]] step %[[VAL_20]] unordered {
// CHECK:             fir.do_loop %[[VAL_22:.*]] = %[[VAL_20]] to %[[ARG2]] step %[[VAL_20]] unordered {
// CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_22]], %[[VAL_21]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_22]], %[[VAL_21]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               hlfir.assign %[[VAL_23]] to %[[VAL_24]] temporary_lhs : !fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = fir.undefined tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           %[[VAL_26:.*]] = fir.insert_value %[[VAL_25]], %[[VAL_19]], [1 : index] : (tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>, i1) -> tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           %[[VAL_27:.*]] = fir.insert_value %[[VAL_26]], %[[VAL_18]]#0, [0 : index] : (tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           %[[VAL_28:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_29:.*]] = fir.embox %[[VAL_28]](%[[VAL_4]]) source_box %[[VAL_18]]#1 : (!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.shape<2>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_29]] to %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_30:.*]] = fir.zero_bits !fir.ref<none>
// CHECK:           %[[VAL_34:.*]] = arith.constant false
// CHECK:           %[[VAL_35:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<none>) -> !fir.ref<i64>
// CHECK:           %[[VAL_39:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_36]], %[[VAL_37]], %[[VAL_34]], %[[VAL_35]],
// CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_41:.*]]:2 = hlfir.declare %[[VAL_40]] {uniq_name = ".tmp.array"} : (!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMtypesTt>>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_42:.*]] = arith.constant true
// CHECK:           %[[VAL_43:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_44:.*]] = %[[VAL_43]] to %[[ARG3]] step %[[VAL_43]] unordered {
// CHECK:             fir.do_loop %[[VAL_45:.*]] = %[[VAL_43]] to %[[ARG2]] step %[[VAL_43]] unordered {
// CHECK:               %[[VAL_46:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_45]], %[[VAL_44]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_47:.*]] = arith.constant false
// CHECK:               %[[VAL_48:.*]] = fir.undefined tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>
// CHECK:               %[[VAL_49:.*]] = fir.insert_value %[[VAL_48]], %[[VAL_47]], [1 : index] : (tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>, i1) -> tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>
// CHECK:               %[[VAL_50:.*]] = fir.insert_value %[[VAL_49]], %[[VAL_46]], [0 : index] : (tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>, !fir.class<!fir.type<_QMtypesTt>>) -> tuple<!fir.class<!fir.type<_QMtypesTt>>, i1>
// CHECK:               %[[VAL_51:.*]] = hlfir.designate %[[VAL_41]]#0 (%[[VAL_45]], %[[VAL_44]])  : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, index, index) -> !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               hlfir.assign %[[VAL_46]] to %[[VAL_51]] temporary_lhs : !fir.class<!fir.type<_QMtypesTt>>, !fir.class<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_52:.*]] = fir.box_addr %[[VAL_46]] : (!fir.class<!fir.type<_QMtypesTt>>) -> !fir.heap<!fir.type<_QMtypesTt>>
// CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_46]] : (!fir.class<!fir.type<_QMtypesTt>>) -> !fir.box<none>
// CHECK:               fir.call @_FortranADestroyWithoutFinalization(%[[VAL_53]]) : (!fir.box<none>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = fir.undefined tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           %[[VAL_55:.*]] = fir.insert_value %[[VAL_54]], %[[VAL_42]], [1 : index] : (tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>, i1) -> tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           %[[VAL_56:.*]] = fir.insert_value %[[VAL_55]], %[[VAL_41]]#0, [0 : index] : (tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>) -> tuple<!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt>>>, i1>
// CHECK:           return
// CHECK:         }
