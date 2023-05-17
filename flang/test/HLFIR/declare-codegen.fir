// Test hlfir.declare operation code generation to FIR

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @numeric_declare(%arg0: !fir.ref<f32>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  return
}
// CHECK-LABEL:   func.func @numeric_declare(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<f32>) {
// CHECK:         %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.ref<f32>) -> !fir.ref<f32>


func.func @char_declare(%arg0: !fir.boxchar<1> ) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  return
}
// CHECK-LABEL:   func.func @char_declare(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:         %[[VAL_2:.*]] = fir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
// CHECK:         %[[VAL_3:.*]] = fir.emboxchar %[[VAL_2]], %[[VAL_1]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>


func.func @derived_declare(%arg0: !fir.ref<!fir.type<t{field:i32}>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<!fir.type<t{field:i32}>>) -> (!fir.ref<!fir.type<t{field:i32}>>, !fir.ref<!fir.type<t{field:i32}>>)
  return
}
// CHECK-LABEL:   func.func @derived_declare(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<t{field:i32}>>) {
// CHECK:         %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.ref<!fir.type<t{field:i32}>>) -> !fir.ref<!fir.type<t{field:i32}>>


func.func @pdt_declare(%arg0: !fir.ref<!fir.type<pdt(param:i32){field:i32}>>) {
  %c1 = arith.constant 1 : index
  %0:2 = hlfir.declare %arg0 typeparams %c1 {uniq_name = "x"} : (!fir.ref<!fir.type<pdt(param:i32){field:i32}>>, index) -> (!fir.box<!fir.type<pdt(param:i32){field:i32}>>, !fir.ref<!fir.type<pdt(param:i32){field:i32}>>)
  return
}
// CHECK-LABEL:   func.func @pdt_declare(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<pdt(param:i32){field:i32}>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:         %[[VAL_2:.*]] = fir.declare %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "x"} : (!fir.ref<!fir.type<pdt(param:i32){field:i32}>>, index) -> !fir.ref<!fir.type<pdt(param:i32){field:i32}>>
// CHECK:         %[[VAL_3:.*]] = fir.embox %[[VAL_2]] typeparams %[[VAL_1]] : (!fir.ref<!fir.type<pdt(param:i32){field:i32}>>, index) -> !fir.box<!fir.type<pdt(param:i32){field:i32}>>


func.func @array_declare(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %shape = fir.shape %c1, %c2 : (index, index) -> !fir.shape<2>
  %0:2 = hlfir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
  return
}
// CHECK-LABEL:   func.func @array_declare(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<?x?xf32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:         %[[VAL_4:.*]] = fir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_4]](%[[VAL_3]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>


func.func @array_declare_2(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape_shift %c1, %c2, %c3, %c4 : (index, index, index, index) -> !fir.shapeshift<2>
  %0:2 = hlfir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
  return
}
// CHECK-LABEL:   func.func @array_declare_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<?x?xf32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_5:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:         %[[VAL_6:.*]] = fir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_6]](%[[VAL_5]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<?x?xf32>>

func.func @array_declare_constant_extents_with_lower_bounds(%arg0: !fir.ref<!fir.array<2x4xf32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape_shift %c1, %c2, %c3, %c4 : (index, index, index, index) -> !fir.shapeshift<2>
  %0:2 = hlfir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.array<2x4xf32>>, !fir.shapeshift<2>) -> (!fir.box<!fir.array<2x4xf32>>, !fir.ref<!fir.array<2x4xf32>>)
  return
}
// CHECK-LABEL:   func.func @array_declare_constant_extents_with_lower_bounds(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<2x4xf32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_5:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:         %[[VAL_6:.*]] = fir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "x"} : (!fir.ref<!fir.array<2x4xf32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<2x4xf32>>
// CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_6]](%[[VAL_5]]) : (!fir.ref<!fir.array<2x4xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<2x4xf32>>


func.func @array_declare_box(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %shape = fir.shift %c1, %c2 : (index, index) -> !fir.shift<2>
  %0:2 = hlfir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  return
}
// CHECK-LABEL:   func.func @array_declare_box(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_3:.*]] = fir.shift %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shift<2>
// CHECK:         %[[VAL_4:.*]] = fir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:         %[[VAL_5:.*]] = fir.rebox %[[VAL_4]](%[[VAL_3]]) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>


func.func @array_declare_box_2(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  return
}
// CHECK-LABEL:   func.func @array_declare_box_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>) {
// CHECK:         %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:         %[[VAL_2:.*]] = fir.rebox %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>

func.func @array_declare_char_box(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>)
  return
}
// CHECK-LABEL:   func.func @array_declare_char_box(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:         %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:         %[[VAL_2:.*]] = fir.rebox %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>


func.func @array_declare_char_box_2(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %shape = fir.shift %c1, %c2 : (index, index) -> !fir.shift<2>
  %0:2 = hlfir.declare %arg0(%shape) typeparams %c3 {uniq_name = "x"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.shift<2>, index) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>)
  return
}
// CHECK-LABEL:   func.func @array_declare_char_box_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_4:.*]] = fir.shift %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shift<2>
// CHECK:         %[[VAL_5:.*]] = fir.declare %[[VAL_0]](%[[VAL_4]]) typeparams %[[VAL_3]] {uniq_name = "x"} : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.shift<2>, index) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:         %[[VAL_6:.*]] = fir.rebox %[[VAL_5]](%[[VAL_4]]) : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?x!fir.char<1,?>>>


func.func @array_declare_char_boxaddr(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>)
  return
}
// CHECK-LABEL: func.func @array_declare_char_boxaddr(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>) {
// CHECK:         %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>


func.func @array_declare_char_boxaddr_2(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>) {
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 typeparams %c3 {uniq_name = "x"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>, index) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>)
  return
}
// CHECK-LABEL: func.func @array_declare_char_boxaddr_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK:         %[[VAL_2:.*]] = fir.declare %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "x"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>, index) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>

func.func @array_declare_unlimited_polymorphic_boxaddr(%arg0: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>) -> (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>, !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>)
  return
}
// CHECK-LABEL: func.func @array_declare_unlimited_polymorphic_boxaddr(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>) {
// CHECK:         %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>


func.func @test_optional_declare(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c42 = arith.constant 42 : index
  %0 = fir.shift %c42 : (index) -> !fir.shift<1>
  %1:2 = hlfir.declare %arg0(%0) {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  return
}

// CHECK-LABEL:   func.func @test_optional_declare(
// CHECK-SAME:      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:  %[[VAL_2:.*]] = fir.shift %[[VAL_1]] : (index) -> !fir.shift<1>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]](%[[VAL_2]]) {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_4:.*]] = fir.is_present %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>) -> i1
// CHECK:  %[[VAL_5:.*]] = fir.if %[[VAL_4]] -> (!fir.box<!fir.array<?xi32>>) {
// CHECK:    %[[VAL_6:.*]] = fir.rebox %[[VAL_3]](%[[VAL_2]]) : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:    fir.result %[[VAL_6]] : !fir.box<!fir.array<?xi32>>
// CHECK:  } else {
// CHECK:    %[[VAL_7:.*]] = fir.absent !fir.box<!fir.array<?xi32>>
// CHECK:    fir.result %[[VAL_7]] : !fir.box<!fir.array<?xi32>>
// CHECK:  }
