// Test hlfir.as_expr code generation

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @char_expr(%addr: !fir.ref<!fir.char<1,?>>, %len: index) {
  %0:2 = hlfir.declare %addr typeparams %len {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %1 = hlfir.as_expr %0#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @char_expr(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,?>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_1]] : index) {bindc_name = ".tmp"}
// CHECK:  %[[VAL_5:.*]] = arith.constant false
// CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[VAL_1]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  hlfir.assign %[[VAL_2]]#0 to %[[VAL_4]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:  %[[VAL_6:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_5]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>

func.func @char_expr_2(%addr: !fir.ref<!fir.char<1,10>>, %len: index) {
  %0:2 = hlfir.declare %addr typeparams %len {uniq_name = "c"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  %1 = hlfir.as_expr %0#0 : (!fir.ref<!fir.char<1,10>>) -> !hlfir.expr<!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @char_expr_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.char<1,10> {bindc_name = ".tmp"}
// CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "c"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
// CHECK:  %[[VAL_5:.*]] = arith.constant false
// CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_1]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
// CHECK:  hlfir.assign %[[VAL_3]]#0 to %[[VAL_4]]#0 : !fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>
// CHECK:  %[[VAL_6:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,10>>, i1>
// CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_5]], [1 : index] : (tuple<!fir.ref<!fir.char<1,10>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,10>>, i1>
// CHECK:  %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,10>>, i1>, !fir.ref<!fir.char<1,10>>) -> tuple<!fir.ref<!fir.char<1,10>>, i1>

func.func @shape_from_type(%arg0 : !fir.ref<!fir.array<10x20xi32>>) {
  %expr = hlfir.as_expr %arg0 : (!fir.ref<!fir.array<10x20xi32>>) -> !hlfir.expr<10x20xi32>
  return
}
// CHECK-LABEL:   func.func @shape_from_type(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x20xi32>>) {
// CHECK:    %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:    %[[VAL_2:.*]] = arith.constant 20 : index
// CHECK:    %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:    %[[VAL_4:.*]] = fir.allocmem !fir.array<10x20xi32> {bindc_name = ".tmp", uniq_name = ""}
// CHECK:    %[[VAL_5:.*]] = arith.constant true
// CHECK:    %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_3]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<10x20xi32>>, !fir.heap<!fir.array<10x20xi32>>)
// CHECK:    hlfir.assign %[[VAL_0]] to %[[VAL_6]]#0 : !fir.ref<!fir.array<10x20xi32>>, !fir.heap<!fir.array<10x20xi32>>
// CHECK:    %[[VAL_7:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_5]], [1 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_6]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, !fir.heap<!fir.array<10x20xi32>>) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>


func.func @shape_from_box(%arg0 : !fir.class<!fir.array<10x?xi32>>) {
  %expr = hlfir.as_expr %arg0 : (!fir.class<!fir.array<10x?xi32>>) -> !hlfir.expr<10x?xi32>
  return
}
// CHECK-LABEL:   func.func @shape_from_box(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.class<!fir.array<10x?xi32>>) {
// CHECK:    %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:    %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:    %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.class<!fir.array<10x?xi32>>, index) -> (index, index, index)
// CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_1]], %[[VAL_3]]#1 : (index, index) -> !fir.shape<2>
// CHECK:    %[[VAL_5:.*]] = fir.allocmem !fir.array<10x?xi32>, %[[VAL_3]]#1 {bindc_name = ".tmp", uniq_name = ""}
// CHECK:    %[[VAL_6:.*]] = arith.constant true
// CHECK:    %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_4]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<10x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<10x?xi32>>, !fir.heap<!fir.array<10x?xi32>>)
// CHECK:    hlfir.assign %[[VAL_0]] to %[[VAL_7]]#0 : !fir.class<!fir.array<10x?xi32>>, !fir.box<!fir.array<10x?xi32>>
// CHECK:    %[[VAL_8:.*]] = fir.undefined tuple<!fir.box<!fir.array<10x?xi32>>, i1>
// CHECK:    %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_6]], [1 : index] : (tuple<!fir.box<!fir.array<10x?xi32>>, i1>, i1) -> tuple<!fir.box<!fir.array<10x?xi32>>, i1>
// CHECK:    %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_7]]#0, [0 : index] : (tuple<!fir.box<!fir.array<10x?xi32>>, i1>, !fir.box<!fir.array<10x?xi32>>) -> tuple<!fir.box<!fir.array<10x?xi32>>, i1>

func.func @test_move(%arg0 : !fir.ref<!fir.array<10x20xi32>>, %must_free: i1) {
  %expr = hlfir.as_expr %arg0 move %must_free: (!fir.ref<!fir.array<10x20xi32>>, i1) -> !hlfir.expr<10x20xi32>
  return
}
// CHECK-LABEL:   func.func @test_move(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x20xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i1) {
// CHECK:    %[[VAL_2:.*]] = fir.undefined tuple<!fir.ref<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_3:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_1]], [1 : index] : (tuple<!fir.ref<!fir.array<10x20xi32>>, i1>, i1) -> tuple<!fir.ref<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_0]], [0 : index] : (tuple<!fir.ref<!fir.array<10x20xi32>>, i1>, !fir.ref<!fir.array<10x20xi32>>) -> tuple<!fir.ref<!fir.array<10x20xi32>>, i1>
