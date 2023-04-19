// Test code generation to FIR of hlfir.designate operations
// with component reference.
// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test_scalar(%arg0: !fir.ref<!fir.type<t1{scalar_i:i32,scalar_x:f32}>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.ref<!fir.type<t1{scalar_i:i32,scalar_x:f32}>>) -> (!fir.ref<!fir.type<t1{scalar_i:i32,scalar_x:f32}>>, !fir.ref<!fir.type<t1{scalar_i:i32,scalar_x:f32}>>)
  %1 = hlfir.designate %0#0{"scalar_x"}   : (!fir.ref<!fir.type<t1{scalar_i:i32,scalar_x:f32}>>) -> !fir.ref<f32>
  return
}
// CHECK-LABEL:   func.func @test_scalar(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<t1{scalar_i:i32,scalar_x:f32}>>) {
// CHECK:  %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "a"}
// CHECK:  %[[VAL_2:.*]] = fir.field_index scalar_x, !fir.type<t1{scalar_i:i32,scalar_x:f32}>
// CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<t1{scalar_i:i32,scalar_x:f32}>>, !fir.field) -> !fir.ref<f32>

func.func @test_array_char_comp_1(%arg0: !fir.ref<!fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.ref<!fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>) -> (!fir.ref<!fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>, !fir.ref<!fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>)
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %1 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %c5 = arith.constant 5 : index
  %2 = hlfir.designate %0#0{"array_char_comp"}   shape %1 typeparams %c5 : (!fir.ref<!fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>, !fir.shape<2>, index) -> !fir.ref<!fir.array<10x20x!fir.char<1,5>>>
  return
}
// CHECK-LABEL:   func.func @test_array_char_comp_1(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>) {
// CHECK:  %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "a"}
// CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
// CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:  %[[VAL_5:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_6:.*]] = fir.field_index array_char_comp, !fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>
// CHECK:  %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.type<t_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>, !fir.field) -> !fir.ref<!fir.array<10x20x!fir.char<1,5>>>

func.func @test_array(%arg0: !fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>) -> (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>, !fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>)
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.designate %0#0{"scalar_x"}   shape %5 : (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}
// CHECK-LABEL:   func.func @test_array(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>) {
// CHECK:  %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "a"}
// CHECK:  %[[VAL_2:.*]] = fir.rebox %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>) -> !fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>
// CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>, index) -> (index, index, index)
// CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_6:.*]] = fir.field_index scalar_x, !fir.type<t1{scalar_i:i32,scalar_x:f32}>
// CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>, index) -> (index, index, index)
// CHECK:  %[[VAL_11:.*]] = fir.slice %[[VAL_8]], %[[VAL_10]]#1, %[[VAL_7]] path %[[VAL_6]] : (index, index, index, !fir.field) -> !fir.slice<1>
// CHECK:  %[[VAL_12:.*]] = fir.rebox %[[VAL_1]] {{\[}}%[[VAL_11]]] : (!fir.box<!fir.array<?x!fir.type<t1{scalar_i:i32,scalar_x:f32}>>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>

func.func @test_array_char(%arg0: !fir.ref<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, %n : index) {
  %5 = fir.shape %n : (index) -> !fir.shape<1>
  %6:2 = hlfir.declare %arg0(%5) {uniq_name = "a"} : (!fir.ref<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.ref<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>)
  %c5 = arith.constant 5 : index
  %10 = hlfir.designate %6#0{"scalar_char"}   shape %5 typeparams %c5 : (!fir.box<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
  return
}
// CHECK-LABEL:   func.func @test_array_char(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_9:.*]] = fir.declare %[[VAL_0]](%[[VAL_8]]) {uniq_name = "a"}
// CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_9]](%[[VAL_8]]) : (!fir.ref<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>
// CHECK:  %[[VAL_11:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_12:.*]] = fir.field_index scalar_char, !fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>
// CHECK:  %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_15:.*]] = fir.slice %[[VAL_14]], %[[VAL_1]], %[[VAL_13]] path %[[VAL_12]] : (index, index, index, !fir.field) -> !fir.slice<1>
// CHECK:  %[[VAL_16:.*]] = fir.embox %[[VAL_9]](%[[VAL_8]]) {{\[}}%[[VAL_15]]] : (!fir.ref<!fir.array<?x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,5>>>

func.func @test_array_char_substring(%arg0: !fir.ref<!fir.array<100x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>) {
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "a"} : (!fir.ref<!fir.array<100x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.ref<!fir.array<100x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>)
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c3_0 = arith.constant 3 : index
  %5 = hlfir.designate %1#0{"scalar_char"}  substr %c3, %c5  shape %0 typeparams %c3_0 : (!fir.ref<!fir.array<100x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<100x!fir.char<1,3>>>
  return
}
// CHECK-LABEL:   func.func @test_array_char_substring(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "a"}
// CHECK:  %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_5:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_7:.*]] = fir.field_index scalar_char, !fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>
// CHECK:  %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_11:.*]] = arith.subi %[[VAL_4]], %[[VAL_10]] : index
// CHECK:  %[[VAL_12:.*]] = fir.slice %[[VAL_9]], %[[VAL_1]], %[[VAL_8]] path %[[VAL_7]] substr %[[VAL_11]], %[[VAL_6]] : (index, index, index, !fir.field, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_13:.*]] = fir.embox %[[VAL_3]](%[[VAL_2]]) {{\[}}%[[VAL_12]]] : (!fir.ref<!fir.array<100x!fir.type<t_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<100x!fir.char<1,3>>>

func.func @test_array_array_comp_1(%arg0: !fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>) {
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "a"} : (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>)
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %5 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %6 = hlfir.designate %1#0{"array_comp"} <%5> (%c4, %c5)  shape %0 : (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<2>, index, index, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  return
}
// CHECK-LABEL:   func.func @test_array_array_comp_1(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "a"}
// CHECK:  %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_5:.*]] = arith.constant 20 : index
// CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
// CHECK:  %[[VAL_7:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_8:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_9:.*]] = fir.field_index array_comp, !fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>
// CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:  %[[ANOTHER_ONE:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_7B:.*]] = arith.subi %[[VAL_7]], %[[ANOTHER_ONE]] : index
// CHECK:  %[[VAL_8B:.*]] = arith.subi %[[VAL_8]], %[[ANOTHER_ONE]] : index
// CHECK:  %[[VAL_12:.*]] = fir.slice %[[VAL_11]], %[[VAL_1]], %[[VAL_10]] path %[[VAL_9]], %[[VAL_7B]], %[[VAL_8B]] : (index, index, index, !fir.field, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_13:.*]] = fir.embox %[[VAL_3]](%[[VAL_2]]) {{\[}}%[[VAL_12]]] : (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<100xf32>>


func.func @test_array_array_comp_1_with_lbs(%arg0: !fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>) {
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "a"} : (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>)
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c42 = arith.constant 42 : index
  %c43 = arith.constant 43 : index
  %5 = fir.shape_shift %c42, %c10, %c43, %c20 : (index, index, index, index) -> !fir.shapeshift<2>
  %c45 = arith.constant 45 : index
  %c47 = arith.constant 47 : index
  %6 = hlfir.designate %1#0{"array_comp"} <%5> (%c45, %c47)  shape %0 : (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shapeshift<2>, index, index, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  return
}
// CHECK-LABEL:   func.func @test_array_array_comp_1_with_lbs(
// CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]] = fir.declare %{{.*}}(%[[VAL_2]]) {uniq_name = "a"} : (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 43 : index
// CHECK:           %[[VAL_8:.*]] = fir.shape_shift %[[VAL_6]], %[[VAL_4]], %[[VAL_7]], %[[VAL_5]] : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:           %[[VAL_9:.*]] = arith.constant 45 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 47 : index
// CHECK:           %[[VAL_11:.*]] = fir.field_index array_comp, !fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]] = arith.subi %[[VAL_9]], %[[VAL_6]] : index
// CHECK:           %[[VAL_15:.*]] = arith.subi %[[VAL_10]], %[[VAL_7]] : index
// CHECK:           %[[VAL_16:.*]] = fir.slice %[[VAL_13]], %[[VAL_1]], %[[VAL_12]] path %[[VAL_11]], %[[VAL_14]], %[[VAL_15]] : (index, index, index, !fir.field, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_3]](%[[VAL_2]]) {{\[}}%[[VAL_16]]] : (!fir.ref<!fir.array<100x!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<100xf32>>


func.func @test_array_comp_slice(%arg0: !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "ga"} : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) -> (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>)
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %1 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c17 = arith.constant 17 : index
  %2 = fir.shape %c10, %c17 : (index, index) -> !fir.shape<2>
  %3 = hlfir.designate %0#0{"array_comp"} <%1> (%c1:%c10:%c1, %c4:%c20:%c1)  shape %2 : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.shape<2>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<10x17xf32>>
  return
}
// CHECK-LABEL:   func.func @test_array_comp_slice(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) {
// CHECK:  %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "ga"} : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) -> !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>
// CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
// CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_7:.*]] = arith.constant 17 : index
// CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_2]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
// CHECK:  %[[VAL_9:.*]] = fir.field_index array_comp, !fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>
// CHECK:  %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_9]] : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10x20xf32>>
// CHECK:  %[[VAL_11:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_4]]) %[[VAL_5]], %[[VAL_6]] : (!fir.ref<!fir.array<10x20xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
// CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<f32>) -> !fir.ref<!fir.array<10x17xf32>>


func.func @test_array_comp_non_contiguous_slice(%arg0: !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "ga"} : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) -> (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>)
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %4 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c4 = arith.constant 4 : index
  %c17 = arith.constant 17 : index
  %5 = fir.shape %c6, %c17 : (index, index) -> !fir.shape<2>
  %6 = hlfir.designate %0#0{"array_comp"} <%4> (%c1:%c6:%c1, %c4:%c20:%c1)  shape %5 : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.shape<2>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<6x17xf32>>
  return
}
// CHECK-LABEL:   func.func @test_array_comp_non_contiguous_slice(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) {
// CHECK:  %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "ga"} : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>) -> !fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>
// CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
// CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_6:.*]] = arith.constant 6 : index
// CHECK:  %[[VAL_7:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_8:.*]] = arith.constant 17 : index
// CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_6]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
// CHECK:  %[[VAL_10:.*]] = fir.field_index array_comp, !fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>
// CHECK:  %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_10]] : (!fir.ref<!fir.type<t_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10x20xf32>>
// CHECK:  %[[VAL_12:.*]] = fir.undefined index
// CHECK:  %[[VAL_13:.*]] = fir.slice %[[VAL_5]], %[[VAL_6]], %[[VAL_5]], %[[VAL_7]], %[[VAL_3]], %[[VAL_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
// CHECK:  %[[VAL_14:.*]] = fir.embox %[[VAL_11]](%[[VAL_4]]) {{\[}}%[[VAL_13]]] : (!fir.ref<!fir.array<10x20xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<6x17xf32>>
