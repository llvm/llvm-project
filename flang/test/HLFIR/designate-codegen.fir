// Test code generation to FIR of hlfir.designate operations
// without component reference.
// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @array_ref(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.ref<i64>) {
  %0:2 = hlfir.declare %arg1 {uniq_name = "n"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2 = fir.load %0#0 : !fir.ref<i64>
  %3 = hlfir.designate %1#0 (%2)  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  return
}
// CHECK-LABEL:   func.func @array_ref(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i64>) {
// CHECK:  %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i64>) -> !fir.ref<i64>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:  %[[VAL_6:.*]] = fir.array_coor %[[VAL_3]] %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>


func.func @char_array_ref(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>, %arg1: !fir.ref<i32>) {
  %0:2 = hlfir.declare %arg1 {uniq_name = "n"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
  %2 = fir.box_elesize %1#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  %c10 = arith.constant 10 : index
  %3 = hlfir.designate %1#0 (%c10)  typeparams %2 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @char_array_ref(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:  %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:  %[[VAL_5:.*]] = fir.box_elesize %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:  %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_7:.*]] = fir.array_coor %[[VAL_3]] %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> !fir.ref<!fir.char<1,?>>
// CHECK:  %[[VAL_8:.*]] = fir.emboxchar %[[VAL_7]], %[[VAL_5]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>


func.func @char_array_ref_cst_len(%arg0: !fir.box<!fir.array<?x!fir.char<1,5>>>, %arg1: !fir.ref<i32>) {
  %0:2 = hlfir.declare %arg1 {uniq_name = "n"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c5 = arith.constant 5 : index
  %1:2 = hlfir.declare %arg0 typeparams %c5 {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> (!fir.box<!fir.array<?x!fir.char<1,5>>>, !fir.box<!fir.array<?x!fir.char<1,5>>>)
  %c10 = arith.constant 10 : index
  %2 = hlfir.designate %1#0 (%c10)  typeparams %c5 : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index, index) -> !fir.ref<!fir.char<1,5>>
  return
}
// CHECK-LABEL:   func.func @char_array_ref_cst_len(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,5>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:  %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:  %[[VAL_3:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_4:.*]] = fir.declare %[[VAL_0]] typeparams %[[VAL_3]] {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
// CHECK:  %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_7:.*]] = fir.array_coor %[[VAL_4]] %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> !fir.ref<!fir.char<1,5>>


func.func @char_array_ref_3(%arg0: !fir.ref<!fir.array<100x!fir.char<1,?>>>, %arg1: !fir.ref<i32>) {
  %0:2 = hlfir.declare %arg1 {uniq_name = "n"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c5 = arith.constant 5 : index
  %c100 = arith.constant 100 : index
  %sh = fir.shape %c100 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%sh) typeparams %c5 {uniq_name = "x"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
  %c10 = arith.constant 10 : index
  %3 = hlfir.designate %1#0 (%c10)  typeparams %c5 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @char_array_ref_3(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.char<1,?>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:  %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:  %[[VAL_3:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_6:.*]] = fir.declare %[[VAL_0]](%[[VAL_5]]) typeparams %[[VAL_3]] {uniq_name = "x"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
// CHECK:  %[[VAL_8:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_9:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_5]]) %[[VAL_8]] typeparams %[[VAL_3]] : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
// CHECK:  %[[VAL_10:.*]] = fir.emboxchar %[[VAL_9]], %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>


func.func @array_section(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c3 = arith.constant 3 : index
  %2 = fir.shape %c3 : (index) -> !fir.shape<1>
  %3 = hlfir.designate %1#0 (%c2:%c8:%c3)  shape %2 : (!fir.ref<!fir.array<10xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<3xf32>>
  return
}
// CHECK-LABEL:   func.func @array_section(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xf32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "x"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
// CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_8:.*]] = fir.undefined index
// CHECK:  %[[VAL_9:.*]] = fir.slice %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_3]](%[[VAL_2]]) {{\[}}%[[VAL_9]]] : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<3xf32>>


func.func @array_section_2(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.ref<i64>) {
  %0:2 = hlfir.declare %arg1 {uniq_name = "n"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2 = fir.load %0#0 : !fir.ref<i64>
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %1#1, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %4 = fir.convert %2 : (i64) -> index
  %c3 = arith.constant 3 : index
  %c42 = arith.constant 42 : index
  %5 = fir.shape %c42 : (index) -> !fir.shape<1>
  %6 = hlfir.designate %1#0 (%4:%3#1:%c3)  shape %5 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}
// CHECK-LABEL:   func.func @array_section_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i64>) {
// CHECK:  %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i64>) -> !fir.ref<i64>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:  %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 42 : index
// CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_12:.*]] = fir.undefined index
// CHECK:  %[[VAL_13:.*]] = fir.slice %[[VAL_8]], %[[VAL_7]]#1, %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_14:.*]] = fir.rebox %[[VAL_3]] {{\[}}%[[VAL_13]]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>


func.func @char_array_section(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>, %arg1: !fir.ref<i32>) {
  %0:2 = hlfir.declare %arg1 {uniq_name = "n"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
  %2 = fir.box_elesize %1#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %1#1, %c0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
  %c3 = arith.constant 3 : index
  %c42 = arith.constant 42 : index
  %4 = fir.shape %c42 : (index) -> !fir.shape<1>
  %5 = hlfir.designate %1#0 (%c1:%3#1:%c3)  shape %4 typeparams %2 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  return
}
// CHECK-LABEL:   func.func @char_array_section(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:  %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:  %[[VAL_5:.*]] = fir.box_elesize %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 42 : index
// CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_12:.*]] = fir.undefined index
// CHECK:  %[[VAL_13:.*]] = fir.slice %[[VAL_6]], %[[VAL_8]]#1, %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_14:.*]] = fir.rebox %[[VAL_3]] {{\[}}%[[VAL_13]]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>


func.func @char_array_section_cst_len(%arg0: !fir.box<!fir.array<?x!fir.char<1,5>>>, %arg1: !fir.ref<i32>) {
  %0:2 = hlfir.declare %arg1 {uniq_name = "n"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c5 = arith.constant 5 : index
  %1:2 = hlfir.declare %arg0 typeparams %c5 {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> (!fir.box<!fir.array<?x!fir.char<1,5>>>, !fir.box<!fir.array<?x!fir.char<1,5>>>)
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %2:3 = fir.box_dims %1#1, %c0 : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> (index, index, index)
  %c3 = arith.constant 3 : index
  %c42 = arith.constant 42 : index
  %3 = fir.shape %c42 : (index) -> !fir.shape<1>
  %4 = hlfir.designate %1#0 (%c1:%2#1:%c3)  shape %3 typeparams %c5 : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
  return
}
// CHECK-LABEL:   func.func @char_array_section_cst_len(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,5>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:  %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:  %[[VAL_3:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_4:.*]] = fir.declare %[[VAL_0]] typeparams %[[VAL_3]] {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
// CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> (index, index, index)
// CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 42 : index
// CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_12:.*]] = fir.undefined index
// CHECK:  %[[VAL_13:.*]] = fir.slice %[[VAL_6]], %[[VAL_8]]#1, %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_14:.*]] = fir.rebox %[[VAL_4]] {{\[}}%[[VAL_13]]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,5>>>

func.func @test_polymorphic_array_elt(%arg0: !fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>> {fir.bindc_name = "x"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest1Ex"} : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>) -> (!fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>)
  %c7 = arith.constant 7 : index
  %1 = hlfir.designate %0#0 (%c7)  : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>, index) -> !fir.class<!fir.type<_QMtypesTt1>>
  return
}
// CHECK-LABEL:   func.func @test_polymorphic_array_elt(
// CHECK-SAME:        %[[VAL_0:.*]]: !fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFtest1Ex"} : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>) -> !fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>
// CHECK:           %[[VAL_2:.*]] = fir.rebox %[[VAL_1]] : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>) -> !fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>
// CHECK:           %[[VAL_3:.*]] = arith.constant 7 : index
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_1]] %[[VAL_3]] : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>, index) -> !fir.ref<!fir.type<_QMtypesTt1>>
// CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_4]] source_box %[[VAL_1]] : (!fir.ref<!fir.type<_QMtypesTt1>>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>) -> !fir.class<!fir.type<_QMtypesTt1>>
// CHECK:           return
// CHECK:         }
