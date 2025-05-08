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
// CHECK:  %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:  %[[VAL_6:.*]] = fir.array_coor %[[VAL_4]] %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>


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
// CHECK:  %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:  %[[VAL_5:.*]] = fir.box_elesize %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:  %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_7:.*]] = fir.array_coor %[[VAL_4]] %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> !fir.ref<!fir.char<1,?>>
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
// CHECK:  %[[VAL_5:.*]] = fir.rebox %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
// CHECK:  %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_7:.*]] = fir.array_coor %[[VAL_5]] %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> !fir.ref<!fir.char<1,5>>


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
// CHECK:  %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:  %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
// CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 42 : index
// CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_12:.*]] = fir.undefined index
// CHECK:  %[[VAL_13:.*]] = fir.slice %[[VAL_8]], %[[VAL_7]]#1, %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_14:.*]] = fir.rebox %[[VAL_4]] {{\[}}%[[VAL_13]]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>


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
// CHECK:  %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:  %[[VAL_5:.*]] = fir.box_elesize %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_4]], %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 42 : index
// CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_12:.*]] = fir.undefined index
// CHECK:  %[[VAL_13:.*]] = fir.slice %[[VAL_6]], %[[VAL_8]]#1, %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_14:.*]] = fir.rebox %[[VAL_4]] {{\[}}%[[VAL_13]]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>


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
// CHECK:  %[[VAL_5:.*]] = fir.rebox %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
// CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, index) -> (index, index, index)
// CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 42 : index
// CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_12:.*]] = fir.undefined index
// CHECK:  %[[VAL_13:.*]] = fir.slice %[[VAL_6]], %[[VAL_8]]#1, %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
// CHECK:  %[[VAL_14:.*]] = fir.rebox %[[VAL_5]] {{\[}}%[[VAL_13]]] : (!fir.box<!fir.array<?x!fir.char<1,5>>>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,5>>>

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
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_2]] %[[VAL_3]] : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>, index) -> !fir.ref<!fir.type<_QMtypesTt1>>
// CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_4]] source_box %[[VAL_2]] : (!fir.ref<!fir.type<_QMtypesTt1>>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt1>>>) -> !fir.class<!fir.type<_QMtypesTt1>>
// CHECK:           return
// CHECK:         }

// Test proper generation of fir.array_coor for contiguous box with default lbounds.
func.func @_QPtest_contiguous_derived_default(%arg0: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>> {fir.bindc_name = "d1", fir.contiguous, fir.optional}) {
  %c1 = arith.constant 1 : index
  %c16_i32 = arith.constant 16 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<contiguous, optional>, uniq_name = "_QFtest_contiguous_derived_defaultEd1"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>, !fir.dscope) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>)
  fir.select_type %1#1 : !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>> [#fir.type_is<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>, ^bb1, unit, ^bb2]
^bb1:  // pred: ^bb0
  %2 = fir.convert %1#1 : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
  %3:2 = hlfir.declare %2 {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_contiguous_derived_defaultEd1"} : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>) -> (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>)
  %4 = hlfir.designate %3#0 (%c1, %c1)  : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, index, index) -> !fir.ref<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>
  %5 = hlfir.designate %4{"i"}   : (!fir.ref<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>) -> !fir.ref<i32>
  hlfir.assign %c16_i32 to %5 : i32, !fir.ref<i32>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %6:2 = hlfir.declare %1#1 {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_contiguous_derived_defaultEd1"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>)
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  return
}
// CHECK-LABEL:   func.func @_QPtest_contiguous_derived_default(
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = fir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_contiguous_derived_defaultEd1"} : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
// CHECK:           %[[VAL_10:.*]] = fir.rebox %[[VAL_9]] : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
// CHECK:           %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>) -> !fir.ref<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_12]] : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_14]] : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_13]]#1, %[[VAL_15]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_17:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_16]]) %[[VAL_0]], %[[VAL_0]] : (!fir.ref<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, !fir.shape<2>, index, index) -> !fir.ref<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>

// Test proper generation of fir.array_coor for contiguous box with non-default lbounds.
func.func @_QPtest_contiguous_derived_lbounds(%arg0: !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>> {fir.bindc_name = "d1", fir.contiguous}) {
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c16_i32 = arith.constant 16 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shift %c1, %c3 : (index, index) -> !fir.shift<2>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_contiguous_derived_lboundsEd1"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>, !fir.shift<2>, !fir.dscope) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>)
  fir.select_type %2#1 : !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>> [#fir.type_is<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>, ^bb1, unit, ^bb2]
^bb1:  // pred: ^bb0
  %3 = fir.convert %2#1 : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>) -> !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
  %4:2 = hlfir.declare %3(%1) {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_contiguous_derived_lboundsEd1"} : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, !fir.shift<2>) -> (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>)
  %5 = hlfir.designate %4#0 (%c1, %c3)  : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, index, index) -> !fir.ref<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>
  %6 = hlfir.designate %5{"i"}   : (!fir.ref<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>) -> !fir.ref<i32>
  hlfir.assign %c16_i32 to %6 : i32, !fir.ref<i32>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %7:2 = hlfir.declare %2#1(%1) {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_contiguous_derived_lboundsEd1"} : (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>, !fir.shift<2>) -> (!fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>, !fir.class<!fir.array<?x?x!fir.type<_QMtypesTt1>>>)
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  return
}
// CHECK-LABEL:   func.func @_QPtest_contiguous_derived_lbounds(
// CHECK:           %[[VAL_0:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = fir.declare %{{.*}}(%[[VAL_4:.*]]) {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_contiguous_derived_lboundsEd1"} : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
// CHECK:           %[[VAL_9:.*]] = fir.rebox %[[VAL_8]](%[[VAL_4]]) : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
// CHECK:           %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>) -> !fir.ref<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_11]] : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_13]] : (!fir.box<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_16:.*]] = arith.subi %[[VAL_1]], %[[VAL_1]] : index
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : index
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_19:.*]] = arith.subi %[[VAL_0]], %[[VAL_0]] : index
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : index
// CHECK:           %[[VAL_21:.*]] = fir.array_coor %[[VAL_10]] %[[VAL_17]], %[[VAL_20]] : (!fir.ref<!fir.array<?x?x!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>>, index, index) -> !fir.ref<!fir.type<_QMtypesTt2{t1:!fir.type<_QMtypesTt1>,i:i32}>>
