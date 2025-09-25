// Test hlfir.designate operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

// array(1)
func.func @array_ref(%arg0 : !fir.ref<!fir.array<10xi32>>) {
  %c1 = arith.constant 1 : index
  %0 = hlfir.designate %arg0 (%c1) :  (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @array_ref(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_1]]) :  (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>

// array(1, 2:3:4, 5)
func.func @array_section(%arg0 : !fir.box<!fir.array<10x?x30xi32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %shape = fir.undefined !fir.shape<1>
  %0 = hlfir.designate %arg0 (%c1, %c2:%c3:%c4, %c5) shape %shape:  (!fir.box<!fir.array<10x?x30xi32>>, index, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL: func.func @array_section(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<10x?x30xi32>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_5:.*]] = arith.constant 5 : index
// CHECK:  %[[VAL_6:.*]] = fir.undefined !fir.shape<1>
// CHECK:  %[[VAL_7:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_1]], %[[VAL_2]]:%[[VAL_3]]:%[[VAL_4]], %[[VAL_5]]) shape %[[VAL_6]] : (!fir.box<!fir.array<10x?x30xi32>>, index, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>

// array%comp
func.func @component_ref(%arg0 : !fir.box<!fir.array<10x!fir.type<t{i:i32}>>>) {
  %shape = fir.undefined !fir.shape<1>
  %0 = hlfir.designate %arg0 {"i"} shape %shape:  (!fir.box<!fir.array<10x!fir.type<t{i:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  return
}
// CHECK-LABEL: func.func @component_ref(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<10x!fir.type<t{i:i32}>>>) {
// CHECK:  %[[VAL_1:.*]] = fir.undefined !fir.shape<1>
// CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_0]]{"i"} shape %[[VAL_1]] : (!fir.box<!fir.array<10x!fir.type<t{i:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>

// array%array_comp(1)
func.func @component_array_ref(%arg0 : !fir.box<!fir.array<10x!fir.type<t2{array_comp:!fir.array<100xi32>}>>>) {
  %c1 = arith.constant 1 : index
  %component_shape = fir.undefined !fir.shapeshift<1>
  %result_shape = fir.undefined !fir.shape<1>
  %0 = hlfir.designate %arg0 {"array_comp"}<%component_shape>(%c1) shape %result_shape : (!fir.box<!fir.array<10x!fir.type<t2{array_comp:!fir.array<100xi32>}>>>, !fir.shapeshift<1>, index, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  return
}
// CHECK-LABEL: func.func @component_array_ref(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<10x!fir.type<t2{array_comp:!fir.array<100xi32>}>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = fir.undefined !fir.shapeshift<1>
// CHECK:  %[[VAL_3:.*]] = fir.undefined !fir.shape<1>
// CHECK:  %[[VAL_4:.*]] = hlfir.designate %[[VAL_0]]{"array_comp"} <%[[VAL_2]]> (%[[VAL_1]]) shape %[[VAL_3]] : (!fir.box<!fir.array<10x!fir.type<t2{array_comp:!fir.array<100xi32>}>>>, !fir.shapeshift<1>, index, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>

// char(3:4)
func.func @substring(%arg0 : !fir.boxchar<1>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = hlfir.designate %arg0 substr %c3, %c4 typeparams %c2:  (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>
  return
}
// CHECK-LABEL: func.func @substring(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_4:.*]] = hlfir.designate %[[VAL_0]] substr %[[VAL_2]], %[[VAL_3]] typeparams %[[VAL_1]] : (!fir.boxchar<1>, index, index, index) -> !fir.boxchar<1>

// char_array(3:4)
func.func @array_substring(%arg0 : !fir.box<!fir.array<10x!fir.char<1,?>>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shape = fir.undefined !fir.shape<1>
  %0 = hlfir.designate %arg0 substr %c3, %c4 shape %shape typeparams %c2 : (!fir.box<!fir.array<10x!fir.char<1,?>>>, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<10x!fir.char<1,?>>>
  return
}
// CHECK-LABEL: func.func @array_substring(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<10x!fir.char<1,?>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:  %[[VAL_4:.*]] = fir.undefined !fir.shape<1>
// CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_0]] substr %[[VAL_2]], %[[VAL_3]] shape %[[VAL_4]] typeparams %[[VAL_1]] : (!fir.box<!fir.array<10x!fir.char<1,?>>>, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<10x!fir.char<1,?>>>

func.func @real_part(%arg0 : !fir.ref<complex<f32>>) {
  %0 = hlfir.designate %arg0 real :  (!fir.ref<complex<f32>>) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: func.func @real_part(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<complex<f32>>) {
// CHECK:  %[[VAL_1:.*]] = hlfir.designate %[[VAL_0]] real :  (!fir.ref<complex<f32>>) -> !fir.ref<f32>

func.func @imag_part_mlir_complex(%arg0 : !fir.ref<complex<f32>>) {
  %0 = hlfir.designate %arg0 imag :  (!fir.ref<complex<f32>>) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: func.func @imag_part_mlir_complex(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<complex<f32>>) {
// CHECK:  %[[VAL_1:.*]] = hlfir.designate %[[VAL_0]] imag :  (!fir.ref<complex<f32>>) -> !fir.ref<f32>

// array%array_complex_comp(1)%im
func.func @component_array_ref_cplx_part(%arg0 : !fir.box<!fir.array<10x!fir.type<t3{array_comp:!fir.array<100xcomplex<f64>>}>>>) {
  %c1 = arith.constant 1 : index
  %component_shape = fir.undefined !fir.shapeshift<1>
  %result_shape = fir.undefined !fir.shape<1>
  %0 = hlfir.designate %arg0 {"array_comp"}<%component_shape>(%c1) imag shape %result_shape : (!fir.box<!fir.array<10x!fir.type<t3{array_comp:!fir.array<100xcomplex<f64>>}>>>, !fir.shapeshift<1>, index, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
  return
}
// CHECK-LABEL: func.func @component_array_ref_cplx_part(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<10x!fir.type<t3{array_comp:!fir.array<100xcomplex<f64>>}>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = fir.undefined !fir.shapeshift<1>
// CHECK:  %[[VAL_3:.*]] = fir.undefined !fir.shape<1>
// CHECK:  %[[VAL_4:.*]] = hlfir.designate %[[VAL_0]]{"array_comp"} <%[[VAL_2]]> (%[[VAL_1]]) imag shape %[[VAL_3]] : (!fir.box<!fir.array<10x!fir.type<t3{array_comp:!fir.array<100xcomplex<f64>>}>>>, !fir.shapeshift<1>, index, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>

// pdt_array(1)
func.func @pdt(%arg0 : !fir.box<!fir.array<?x!fir.type<pdt(param:i32){field:i32}>>>) {
  %c1 = arith.constant 1 : index
  %0 = hlfir.designate %arg0(%c1) typeparams %c1 :  (!fir.box<!fir.array<?x!fir.type<pdt(param:i32){field:i32}>>>, index, index) -> !fir.box<!fir.type<pdt(param:i32){field:i32}>>
  return
}
// CHECK-LABEL: func.func @pdt(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<pdt(param:i32){field:i32}>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_1]]) typeparams %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.type<pdt(param:i32){field:i32}>>>, index, index) -> !fir.box<!fir.type<pdt(param:i32){field:i32}>>
