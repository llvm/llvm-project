// HLFIR ops diagnotic tests

// RUN: fir-opt -strict-intrinsic-verifier -split-input-file -verify-diagnostics %s

func.func @bad_declare(%arg0: !fir.ref<f32>) {
  // expected-error@+1 {{'hlfir.declare' op first result type is inconsistent with variable properties: expected '!fir.ref<f32>'}}
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<f32>) -> (!fir.box<f32>, !fir.ref<f32>)
  return
}

// -----
func.func @bad_declare_lower_bounds(%arg0: !fir.ref<!fir.array<2x4xf32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape_shift %c1, %c2, %c3, %c4 : (index, index, index, index) -> !fir.shapeshift<2>
  // expected-error@+1 {{'hlfir.declare' op first result type is inconsistent with variable properties: expected '!fir.box<!fir.array<2x4xf32>>'}}
  %0:2 = hlfir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.array<2x4xf32>>, !fir.shapeshift<2>) -> (!fir.ref<!fir.array<2x4xf32>>, !fir.ref<!fir.array<2x4xf32>>)
  return
}

// -----
func.func @bad_declare(%arg0: !fir.ref<f32>) {
  // expected-error@+1 {{'hlfir.declare' op second result type must match input memref type}}
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<f32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  return
}

// -----

// Test that FortranVariableInterface verifier is kicking in. This verifier itself is already tested with fir.declare.
func.func @bad_array_declare(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  // expected-error@+1 {{'hlfir.declare' op of array entity with a raw address base must have a shape operand that is a shape or shapeshift}}
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
  return
}

// -----
func.func @bad_assign_scalar_character(%arg0: !fir.boxchar<1>, %arg1: !fir.char<1,?>) {
  // expected-error@+1 {{'hlfir.assign' op operand #0 must be any Fortran value or variable type, but got '!fir.char<1,?>'}}
  hlfir.assign %arg1 to %arg0 : !fir.char<1,?>, !fir.boxchar<1>
  return
}

// -----
func.func @bad_assign_scalar_character_1(%arg0: !fir.boxchar<1>, %arg1: !hlfir.expr<!fir.char<1,?>>) {
  // expected-error@+1 {{'hlfir.assign' op operand #1 must be any HLFIR variable type, but got '!hlfir.expr<!fir.char<1,?>>'}}
  hlfir.assign %arg0 to %arg1 : !fir.boxchar<1>, !hlfir.expr<!fir.char<1,?>>
  return
}

// -----
func.func @bad_assign_scalar_integer(%arg0: !fir.ref<i32>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.assign' op operand #1 must be any HLFIR variable type, but got 'i32'}}
  hlfir.assign %arg0 to %arg1 : !fir.ref<i32>, i32
  return
}

// -----
func.func @bad_assign_array(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !hlfir.expr<?xi32>) {
  // expected-error@+1 {{'hlfir.assign' op operand #1 must be any HLFIR variable type, but got '!fir.ref<!fir.array<?xi32>>'}}
  hlfir.assign %arg1 to %arg0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<?xi32>>
  return
}

// -----
func.func @bad_assign_array_2(%arg0: !fir.ref<!fir.array<10xi32>>, %arg1: !hlfir.expr<?xi32>) {
  // expected-error@+1 {{'hlfir.assign' op operand #1 must be any HLFIR variable type, but got '!hlfir.expr<?xi32>'}}
  hlfir.assign %arg0 to %arg1 : !fir.ref<!fir.array<10xi32>>, !hlfir.expr<?xi32>
  return
}

// -----
func.func @bad_designate_component(%arg0 : !fir.ref<i32>) {
  // expected-error@+1 {{'hlfir.designate' op component must be provided only when the memref is a derived type}}
  %0 = hlfir.designate %arg0 {"some_component"} :  (!fir.ref<i32>) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_component_2(%arg0 : !fir.ref<!fir.type<t{i:i32}>>) {
  // expected-error@+1 {{'hlfir.designate' op component "bad_comp" is not a component of memref element type '!fir.type<t{i:i32}>'}}
  %0 = hlfir.designate %arg0 {"bad_comp"} :  (!fir.ref<!fir.type<t{i:i32}>>) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_component_3(%arg0 : !fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>) {
  // expected-error@+1 {{'hlfir.designate' op indices must be provided and must not contain triplets when both memref and component are arrays}}
  %0 = hlfir.designate %arg0 {"i"} :  (!fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_component_4(%arg0 : !fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>) {
  %component_shape = fir.undefined !fir.shape<1>
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op indices must be provided and must not contain triplets when both memref and component are arrays}}
  %0 = hlfir.designate %arg0 {"i"}<%component_shape>(%c1:%c1:%c1):  (!fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>, !fir.shape<1>, index, index, index) -> !fir.ref<!fir.array<20xi32>>
  return
}

// -----
func.func @bad_designate_component_5(%arg0 : !fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>) {
  %component_shape = fir.undefined !fir.shape<2>
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op component_shape must be a fir.shape or fir.shapeshift with the rank of the component}}
  %0 = hlfir.designate %arg0 {"i"}<%component_shape>(%c1):  (!fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>, !fir.shape<2>, index) -> !fir.ref<!fir.array<20xi32>>
  return
}

// -----
func.func @bad_designate_component_6(%arg0 : !fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>) {
  %component_shape = fir.undefined !fir.shift<1>
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op component_shape must be a fir.shape or fir.shapeshift with the rank of the component}}
  %0 = hlfir.designate %arg0 {"i"}<%component_shape>(%c1):  (!fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>, !fir.shift<1>, index) -> !fir.ref<!fir.array<20xi32>>
  return
}

// -----
func.func @bad_designate_component_7(%arg0 : !fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>) {
  %component_shape = fir.undefined !fir.shapeshift<2>
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op component_shape must be a fir.shape or fir.shapeshift with the rank of the component}}
  %0 = hlfir.designate %arg0 {"i"}<%component_shape>(%c1):  (!fir.ref<!fir.array<20x!fir.type<t{i:!fir.array<100xi32>}>>>, !fir.shapeshift<2>, index) -> !fir.ref<!fir.array<20xi32>>
  return
}

// -----
func.func @bad_designate_component_8(%arg0 : !fir.ref<!fir.type<t{i:!fir.array<100xi32>}>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op component_shape must be provided when indexing a component}}
  %0 = hlfir.designate %arg0 {"i"}(%c1):  (!fir.ref<!fir.type<t{i:!fir.array<100xi32>}>>, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_component_9(%arg0 : !fir.ref<!fir.array<20x!fir.type<t{i:i32}>>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op indices must not be provided if component appears and is not an array component}}
  %0 = hlfir.designate %arg0 {"i"}(%c1):  (!fir.ref<!fir.array<20x!fir.type<t{i:i32}>>>, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_component_10(%arg0 : !fir.ref<!fir.type<t{i:!fir.array<100xi32>}>>) {
  %component_shape = fir.undefined !fir.shapeshift<1>
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op indices number must match array component rank}}
  %0 = hlfir.designate %arg0 {"i"}<%component_shape>(%c1, %c1):  (!fir.ref<!fir.type<t{i:!fir.array<100xi32>}>>, !fir.shapeshift<1>, index, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_substring_1(%arg0 : !fir.ref<!fir.char<1,20>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op substring must contain 2 indices when provided}}
  %0 = hlfir.designate %arg0 substr %c1, %c1, %c1:  (!fir.ref<!fir.char<1,20>>, index, index, index) -> !fir.boxchar<1>
  return
}

// -----
func.func @bad_designate_indices_1(%arg0 : !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op indices number must match memref rank}}
  %0 = hlfir.designate %arg0 (%c1, %c1):  (!fir.ref<i32>, index, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_indices_2(%arg0 : !fir.ref<!fir.array<10xi32>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op indices number must match memref rank}}
  %0 = hlfir.designate %arg0 (%c1, %c1):  (!fir.ref<!fir.array<10xi32>>, index, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_substring_2(%arg0 : !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op memref or component must have character type if substring indices are provided}}
  %0 = hlfir.designate %arg0 substr %c1, %c1:  (!fir.ref<i32>, index, index) -> !fir.boxchar<1>
  return
}

// -----
func.func @bad_designate_cmplx_part(%arg0 : !fir.ref<!fir.array<10xi32>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op memref or component must have complex type if complex_part is provided}}
  %0 = hlfir.designate %arg0 (%c1) imag:  (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_out_rank(%arg0 : !fir.ref<!fir.array<10xi32>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op result type rank is not consistent with operands, expected rank 1}}
  %0 = hlfir.designate %arg0 (%c1:%c1:%c1):  (!fir.ref<!fir.array<10xi32>>, index, index, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_out_type(%arg0 : !fir.ref<complex<f32>>) {
  // expected-error@+1 {{'hlfir.designate' op result element type is not consistent with operands, expected 'f32'}}
  %0 = hlfir.designate %arg0 imag:  (!fir.ref<complex<f32>>) -> !fir.ref<complex<f64>>
  return
}

// -----
func.func @bad_designate_out_type(%arg0 : !fir.ref<!fir.box<complex<f32>>>) {
  // expected-error@+1 {{'hlfir.designate' op result type must only be a box address type if it designates a component that is a fir.box or fir.class and if there are no indices, substrings, and complex part}}
  %0 = hlfir.designate %arg0 imag:  (!fir.ref<!fir.box<complex<f32>>>) -> !fir.ref<!fir.box<f32>>
  return
}

// -----
func.func @bad_designate_shape(%arg0 : !fir.ref<!fir.array<10xi32>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op shape must be provided if and only if the result is an array that is not a box address}}
  %0 = hlfir.designate %arg0 (%c1:%c1:%c1):  (!fir.ref<!fir.array<10xi32>>, index, index, index) -> !fir.box<!fir.array<?xi32>>
  return
}

// -----
func.func @bad_designate_shape_2(%arg0 : !fir.ref<!fir.array<10xi32>>) {
  %c1 = arith.constant 1 : index
  %shape = fir.undefined !fir.shape<1>
  // expected-error@+1 {{'hlfir.designate' op shape must be provided if and only if the result is an array that is not a box address}}
  %0 = hlfir.designate %arg0 (%c1) shape %shape:  (!fir.ref<!fir.array<10xi32>>, index, !fir.shape<1>) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_designate_len_params(%arg0 : !fir.ref<!fir.char<1,10>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op must be provided one length parameter when the result is a character}}
  %0 = hlfir.designate %arg0 substr %c1, %c1:  (!fir.ref<!fir.char<1,10>>, index, index) -> !fir.boxchar<1>
  return
}

// -----
func.func @bad_designate_len_params_2(%arg0 : !fir.box<!fir.array<?x!fir.type<pdt(param:i32){field:i32}>>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op must be provided the same number of length parameters as in the result derived type}}
  %0 = hlfir.designate %arg0(%c1) typeparams %c1, %c1 :  (!fir.box<!fir.array<?x!fir.type<pdt(param:i32){field:i32}>>>, index, index, index) -> !fir.box<!fir.type<pdt(param:i32){field:i32}>>
  return
}

// -----
func.func @bad_designate_len_params_3(%arg0 : !fir.box<!fir.array<?xi32>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.designate' op must not be provided length parameters if the result type does not have length parameters}}
  %0 = hlfir.designate %arg0(%c1) typeparams %c1 :  (!fir.box<!fir.array<?xi32>>, index, index) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_concat(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,20>>) {
  %c30 = arith.constant 30 : index
  // expected-error@+1 {{'hlfir.concat' op result #0 must be any character scalar expression type, but got '!fir.ref<!fir.char<1,30>>'}}
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, index) -> (!fir.ref<!fir.char<1,30>>)
  return
}

// -----
func.func @bad_concat_2(%arg0: !fir.ref<!fir.array<100x!fir.char<1,10>>>, %arg1: !fir.ref<!fir.array<100x!fir.char<1,20>>>) {
  %c30 = arith.constant 30 : index
  // expected-error@+1 {{'hlfir.concat' op operand #0 must be variadic of any character scalar type, but got '!fir.ref<!fir.array<100x!fir.char<1,10>>>'}}
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>, index) -> (!hlfir.expr<100x!fir.char<1,30>>)
  return
}

// -----
func.func @bad_concat_3(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<i32>) {
  %c30 = arith.constant 30 : index
  // expected-error@+1 {{'hlfir.concat' op operand #1 must be variadic of any character scalar type, but got '!fir.ref<i32>'}}
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.char<1,10>>, !fir.ref<i32>, index) -> (!hlfir.expr<!fir.char<1,30>>)
  return
}

// -----
func.func @bad_concat_4(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<2,20>>) {
  %c30 = arith.constant 30 : index
  // expected-error@+1 {{'hlfir.concat' op strings must have the same KIND as the result type}}
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<2,20>>, index) -> (!hlfir.expr<!fir.char<1,30>>)
  return
}

// -----
func.func @bad_concat_4(%arg0: !fir.ref<!fir.char<1,30>>) {
  %c30 = arith.constant 30 : index
  // expected-error@+1 {{'hlfir.concat' op must be provided at least two string operands}}
  %0 = hlfir.concat %arg0 len %c30 : (!fir.ref<!fir.char<1,30>>, index) -> (!hlfir.expr<!fir.char<1,30>>)
  return
}

// -----
func.func @bad_cmpchar_1(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<2,10>>) {
  // expected-error@+1 {{'hlfir.cmpchar' op character arguments must have the same KIND}}
  %0 = hlfir.cmpchar ne %arg0  %arg1 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<2,10>>) -> i1
}

func.func @bad_cmpchar_2(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,10>>) {
  // expected-error@+1 {{'hlfir.cmpchar' op expected signed predicate}}
  %0 = hlfir.cmpchar ugt %arg0  %arg1 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> i1
}

// -----
func.func @bad_index_1(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<2,10>>) {
  // expected-error@+1 {{'hlfir.index' op character arguments must have the same KIND}}
  %0 = hlfir.index %arg0 in %arg1 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<2,10>>) -> i32
}

// -----
func.func @bad_any1(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.any' op result must have the same element type as MASK argument}}
  %0 = hlfir.any %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<8>
}

// -----
func.func @bad_any2(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.any' op result must have the same element type as MASK argument}}
  %0 = hlfir.any %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<?x!fir.logical<8>>
}

// -----
func.func @bad_any3(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32){
  // expected-error@+1 {{'hlfir.any' op result rank must be one less than MASK}}
  %0 = hlfir.any %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<?x?x!fir.logical<4>>
}

// -----
func.func @bad_any4(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.any' op result must be an array}}
  %0 = hlfir.any %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<!fir.logical<4>>
}

// -----
func.func @bad_any5(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.any' op result must be of logical type}}
  %0 = hlfir.any %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> i32
}

// -----
func.func @bad_any6(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.any' op result must be of logical type}}
  %0 = hlfir.any %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<!fir.logical<4>>
}

// -----
func.func @bad_all1(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.all' op result must have the same element type as MASK argument}}
  %0 = hlfir.all %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<8>
}

// -----
func.func @bad_all2(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.all' op result must have the same element type as MASK argument}}
  %0 = hlfir.all %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<?x!fir.logical<8>>
}

// -----
func.func @bad_all3(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32){
  // expected-error@+1 {{'hlfir.all' op result rank must be one less than MASK}}
  %0 = hlfir.all %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<?x?x!fir.logical<4>>
}

// -----
func.func @bad_all4(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.all' op result must be an array}}
  %0 = hlfir.all %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<!fir.logical<4>>
}

// -----
func.func @bad_all5(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.all' op result must be of logical type}}
  %0 = hlfir.all %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> i32
}

// -----
func.func @bad_all6(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.all' op result must be of logical type}}
  %0 = hlfir.all %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<!fir.logical<4>>
}

// -----
func.func @bad_count1(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.count' op result must be an array}}
  %0 = hlfir.count %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_count2(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: i32){
  // expected-error@+1 {{'hlfir.count' op result rank must be one less than MASK}}
  %0 = hlfir.count %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, i32) -> !hlfir.expr<?x?x!fir.logical<4>>
}

// -----
func.func @bad_count3(%arg0: !hlfir.expr<?x!fir.logical<4>>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.count' op result must be of numerical array type}}
  %0 = hlfir.count %arg0 dim %arg1 : (!hlfir.expr<?x!fir.logical<4>>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_count4(%arg0: !hlfir.expr<?x!fir.logical<4>>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.count' op result must be of numerical scalar type}}
  %0 = hlfir.count %arg0 dim %arg1 : (!hlfir.expr<?x!fir.logical<4>>, i32) -> !fir.logical<4>
}

// -----
func.func @bad_maxval1(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxval' op result must have the same element type as ARRAY argument}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> f32
}

// -----
func.func @bad_maxval2(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_maxval3(%arg0: !hlfir.expr<?x5x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?xi32>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_maxval4(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxval' op result rank must be one less than ARRAY}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}

// -----
func.func @bad_maxval5(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxval' op result must be of numerical scalar type}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !fir.logical<4>
}

// -----
func.func @bad_maxval6(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32){
  // expected-error@+1 {{'hlfir.maxval' op result must be an array}}
  %0 = hlfir.maxval %arg0 dim %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_maxval7(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.maxval' op result must be of numerical scalar type}}
  %0 = hlfir.maxval %arg0 : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_maxval8(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxval' op result must have the same element type as ARRAY argument}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> i32
}

// -----
func.func @bad_maxval9(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_maxval10(%arg0: !hlfir.expr<?x5x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?x!fir.char<1,?>>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_maxval11(%arg0: !hlfir.expr<?x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxval' op result rank must be one less than ARRAY}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?x!fir.char<1,?>>
}

// -----
func.func @bad_maxval12(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxval' op result must be scalar character}}
  %0 = hlfir.maxval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x!fir.char<1,?>>
}

// -----
func.func @bad_maxval13(%arg0: !hlfir.expr<?x?x!fir.char<1,?>>, %arg1: i32){
  // expected-error@+1 {{'hlfir.maxval' op result must be an array}}
  %0 = hlfir.maxval %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.char<1,?>>, i32) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_minval1(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minval' op result must have the same element type as ARRAY argument}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> f32
}

// -----
func.func @bad_minval2(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_minval3(%arg0: !hlfir.expr<?x5x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?xi32>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_minval4(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minval' op result rank must be one less than ARRAY}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}

// -----
func.func @bad_minval5(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minval' op result must be of numerical scalar type}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !fir.logical<4>
}

// -----
func.func @bad_minval6(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32){
  // expected-error@+1 {{'hlfir.minval' op result must be an array}}
  %0 = hlfir.minval %arg0 dim %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_minval7(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.minval' op result must be of numerical scalar type}}
  %0 = hlfir.minval %arg0 : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_minval8(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minval' op result must have the same element type as ARRAY argument}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> i32
}

// -----
func.func @bad_minval9(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_minval10(%arg0: !hlfir.expr<?x5x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?x!fir.char<1,?>>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_minval11(%arg0: !hlfir.expr<?x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minval' op result rank must be one less than ARRAY}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?x!fir.char<1,?>>
}

// -----
func.func @bad_minval12(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minval' op result must be scalar character}}
  %0 = hlfir.minval %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x!fir.char<1,?>>
}

// -----
func.func @bad_minval13(%arg0: !hlfir.expr<?x?x!fir.char<1,?>>, %arg1: i32){
  // expected-error@+1 {{'hlfir.minval' op result must be an array}}
  %0 = hlfir.minval %arg0 dim %arg1 : (!hlfir.expr<?x?x!fir.char<1,?>>, i32) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_minloc1(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minloc' op result must be scalar integer}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> f32
}

// -----
func.func @bad_minloc2(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_minloc3(%arg0: !hlfir.expr<?x5x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?xi32>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_minloc4(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minloc' op result rank must be one less than ARRAY}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}

// -----
func.func @bad_minloc5(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minloc' op result must be scalar integer}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !fir.logical<4>
}

// -----
func.func @bad_minloc6(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32){
  // expected-error@+1 {{'hlfir.minloc' op result must be an array}}
  %0 = hlfir.minloc %arg0 dim %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_minloc7(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.minloc' op result must be of numerical expr type}}
  %0 = hlfir.minloc %arg0 : (!hlfir.expr<?xi32>) -> i32
}

// -----
func.func @bad_minloc8(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.minloc' op result must have integer elements}}
  %0 = hlfir.minloc %arg0 : (!hlfir.expr<?xi32>) -> !hlfir.expr<?xf32>
}

// -----
func.func @bad_minloc9(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_minloc10(%arg0: !hlfir.expr<?x5x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?x!fir.char<1,?>>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_minloc11(%arg0: !hlfir.expr<?x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.minloc' op result rank must be one less than ARRAY}}
  %0 = hlfir.minloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}

// -----
func.func @bad_maxloc1(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxloc' op result must be scalar integer}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> f32
}

// -----
func.func @bad_maxloc2(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_maxloc3(%arg0: !hlfir.expr<?x5x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?xi32>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_maxloc4(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxloc' op result rank must be one less than ARRAY}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}

// -----
func.func @bad_maxloc5(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxloc' op result must be scalar integer}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !fir.logical<4>
}

// -----
func.func @bad_maxloc6(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32){
  // expected-error@+1 {{'hlfir.maxloc' op result must be an array}}
  %0 = hlfir.maxloc %arg0 dim %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_maxloc7(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.maxloc' op result must be of numerical expr type}}
  %0 = hlfir.maxloc %arg0 : (!hlfir.expr<?xi32>) -> i32
}

// -----
func.func @bad_maxloc8(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.maxloc' op result must have integer elements}}
  %0 = hlfir.maxloc %arg0 : (!hlfir.expr<?xi32>) -> !hlfir.expr<?xf32>
}

// -----
func.func @bad_maxloc9(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x!fir.char<1,?>>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_maxloc10(%arg0: !hlfir.expr<?x5x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?x!fir.char<1,?>>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<!fir.char<1,?>>
}

// -----
func.func @bad_maxloc11(%arg0: !hlfir.expr<?x?x!fir.char<1,?>>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.maxloc' op result rank must be one less than ARRAY}}
  %0 = hlfir.maxloc %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?x!fir.char<1,?>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}


// -----
func.func @bad_product1(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.product' op result must have the same element type as ARRAY argument}}
  %0 = hlfir.product %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> f32
}

// -----
func.func @bad_product2(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.product %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_product3(%arg0: !hlfir.expr<?x5x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.product %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?xi32>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_product4(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.product' op result rank must be one less than ARRAY}}
  %0 = hlfir.product %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}

// -----
func.func @bad_product5(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.product' op result must be of numerical scalar type}}
  %0 = hlfir.product %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !fir.logical<4>
}

// -----
func.func @bad_product6(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32){
  // expected-error@+1 {{'hlfir.product' op result must be an array}}
  %0 = hlfir.product %arg0 dim %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_product7(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.product' op result must be of numerical scalar type}}
  %0 = hlfir.product %arg0 : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_sum1(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.sum' op result must have the same element type as ARRAY argument}}
  %0 = hlfir.sum %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> f32
}

// -----
func.func @bad_sum2(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.sum %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.array<?x?x?x?x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_sum3(%arg0: !hlfir.expr<?x5x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) {
  // expected-warning@+1 {{MASK must be conformable to ARRAY}}
  %0 = hlfir.sum %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x5x?xi32>, i32, !fir.box<!fir.array<2x6x?x!fir.logical<4>>>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_sum4(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.sum' op result rank must be one less than ARRAY}}
  %0 = hlfir.sum %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?x?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
}

// -----
func.func @bad_sum5(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.sum' op result must be of numerical scalar type}}
  %0 = hlfir.sum %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !fir.logical<4>
}

// -----
func.func @bad_sum6(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32){
  // expected-error@+1 {{'hlfir.sum' op result must be an array}}
  %0 = hlfir.sum %arg0 dim %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<i32>
}

// -----
func.func @bad_sum7(%arg0: !hlfir.expr<?xi32>){
  // expected-error@+1 {{'hlfir.sum' op result must be of numerical scalar type}}
  %0 = hlfir.sum %arg0 : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>
}

// -----
func.func @bad_matmul1(%arg0: !hlfir.expr<?x?x?xi32>, %arg1: !hlfir.expr<?x?xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op array must have either rank 1 or rank 2}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?x?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----
func.func @bad_matmul2(%arg0: !hlfir.expr<?xi32>, %arg1: !hlfir.expr<?xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op at least one array must have rank 2}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----
func.func @bad_matmul3(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: !hlfir.expr<?x?xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op if one array is logical, so should the other be}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----
func.func @bad_matmul4(%arg0: !hlfir.expr<?x2xi32>, %arg1: !hlfir.expr<200x?xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op the last dimension of LHS should match the first dimension of RHS}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x2xi32>, !hlfir.expr<200x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----
func.func @bad_matmul5(%arg0: !hlfir.expr<?x?xi32>, %arg1: !hlfir.expr<?x?xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op the result type should be a logical only if the argument types are logical}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?x!fir.logical<4>>
  return
}

// -----
func.func @bad_matmul6(%arg0: !hlfir.expr<1x2xi32>, %arg1: !hlfir.expr<2x3xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op incorrect result shape}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<1x2xi32>, !hlfir.expr<2x3xi32>) -> !hlfir.expr<10x30xi32>
  return
}

// -----
func.func @bad_matmul7(%arg0: !hlfir.expr<1x2xi32>, %arg1: !hlfir.expr<2xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op incorrect result shape}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<1x2xi32>, !hlfir.expr<2xi32>) -> !hlfir.expr<1x3xi32>
  return
}

// -----
func.func @bad_matmul8(%arg0: !hlfir.expr<2xi32>, %arg1: !hlfir.expr<2x3xi32>) {
  // expected-error@+1 {{'hlfir.matmul' op incorrect result shape}}
  %0 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<2xi32>, !hlfir.expr<2x3xi32>) -> !hlfir.expr<1x3xi32>
  return
}

// -----
func.func @bad_dot_product1(%arg0: !hlfir.expr<2xi32>, %arg1: !hlfir.expr<2x3xi32>) {
  // expected-error@+1 {{'hlfir.dot_product' op both arrays must have rank 1}}
  %0 = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<2xi32>, !hlfir.expr<2x3xi32>) -> i32
  return
}

// -----
func.func @bad_dot_product2(%arg0: !hlfir.expr<2xi32>, %arg1: !hlfir.expr<3xi32>) {
  // expected-error@+1 {{'hlfir.dot_product' op both arrays must have the same size}}
  %0 = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<2xi32>, !hlfir.expr<3xi32>) -> i32
  return
}

// -----
func.func @bad_dot_product3(%arg0: !hlfir.expr<2xi32>, %arg1: !hlfir.expr<2x!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.dot_product' op if one array is logical, so should the other be}}
  %0 = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<2xi32>, !hlfir.expr<2x!fir.logical<4>>) -> i32
  return
}

// -----
func.func @bad_dot_product4(%arg0: !hlfir.expr<2xi32>, %arg1: !hlfir.expr<2xi32>) {
  // expected-error@+1 {{'hlfir.dot_product' op the result type should be a logical only if the argument types are logical}}
  %0 = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<2xi32>, !hlfir.expr<2xi32>) -> !fir.logical<4>
  return
}

// -----
func.func @bad_dot_product5(%arg0: !hlfir.expr<2xi32>, %arg1: !hlfir.expr<2xi32>) {
  // expected-error@+1 {{'hlfir.dot_product' op the result must be of scalar numerical or logical type}}
  %0 = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<2xi32>, !hlfir.expr<2xi32>) -> !hlfir.expr<i32>
  return
}

// -----
func.func @bad_transpose1(%arg0: !hlfir.expr<2xi32>) {
  // expected-error@+1 {{'hlfir.transpose' op input and output arrays should have rank 2}}
  %0 = hlfir.transpose %arg0 : (!hlfir.expr<2xi32>) -> !hlfir.expr<2xi32>
  return
}

// -----
func.func @bad_transpose2(%arg0: !hlfir.expr<2x3xi32>) {
  // expected-error@+1 {{'hlfir.transpose' op output shape does not match input array}}
  %0 = hlfir.transpose %arg0 : (!hlfir.expr<2x3xi32>) -> !hlfir.expr<2x2xi32>
  return
}

// -----
func.func @bad_transpose3(%arg0: !hlfir.expr<2x3xi32>) {
  // expected-error@+1 {{'hlfir.transpose' op input and output arrays should have the same element type}}
  %0 = hlfir.transpose %arg0 : (!hlfir.expr<2x3xi32>) -> !hlfir.expr<3x2xf64>
  return
}

// -----
func.func @bad_matmultranspose1(%arg0: !hlfir.expr<?x?x?xi32>, %arg1: !hlfir.expr<?x?xi32>) {
  // expected-error@+1 {{'hlfir.matmul_transpose' op array must have either rank 1 or rank 2}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<?x?x?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----
func.func @bad_matmultranspose2(%arg0: !hlfir.expr<?xi32>, %arg1: !hlfir.expr<?xi32>) {
  // expected-error@+1 {{'hlfir.matmul_transpose' op array must have either rank 1 or rank 2}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----
func.func @bad_matmultranspose3(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: !hlfir.expr<?x?xi32>) {
  // expected-error@+1 {{'hlfir.matmul_transpose' op if one array is logical, so should the other be}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----
func.func @bad_matmultranspose5(%arg0: !hlfir.expr<?x?xi32>, %arg1: !hlfir.expr<?x?xi32>) {
  // expected-error@+1 {{'hlfir.matmul_transpose' op the result type should be a logical only if the argument types are logical}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<?x?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?x!fir.logical<4>>
  return
}

// -----
func.func @bad_matmultranspose6(%arg0: !hlfir.expr<2x1xi32>, %arg1: !hlfir.expr<2x3xi32>) {
  // expected-error@+1 {{'hlfir.matmul_transpose' op incorrect result shape}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<2x1xi32>, !hlfir.expr<2x3xi32>) -> !hlfir.expr<10x30xi32>
  return
}

// -----
func.func @bad_matmultranspose7(%arg0: !hlfir.expr<2x1xi32>, %arg1: !hlfir.expr<2xi32>) {
  // expected-error@+1 {{'hlfir.matmul_transpose' op incorrect result shape}}
  %0 = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<2x1xi32>, !hlfir.expr<2xi32>) -> !hlfir.expr<1x3xi32>
  return
}

// -----
func.func @bad_assign_1(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<?xi32>>) {
  // expected-error@+1 {{'hlfir.assign' op lhs must be an allocatable when `realloc` is set}}
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
  return
}

// -----
func.func @bad_assign_2(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.box<!fir.array<?xi32>>) {
  // expected-error@+1 {{'hlfir.assign' op `realloc` must be set and lhs must be a character allocatable when `keep_lhs_length_if_realloc` is set}}
  hlfir.assign %arg1 to %arg0 realloc keep_lhs_len : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}

// -----
func.func @bad_parent_comp1(%arg0: !fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
  // expected-error@+1 {{'hlfir.parent_comp' op must be provided a shape if and only if the base is an array}}
  %2 = hlfir.parent_comp %arg0 : (!fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) -> !fir.box<!fir.array<10x!fir.type<t1{i:i32}>>>
  return
}

// -----
func.func @bad_parent_comp2(%arg0: !fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'hlfir.parent_comp' op result type rank must match input type rank}}
  %2 = hlfir.parent_comp %arg0 shape %1 : (!fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x5x!fir.type<t1{i:i32}>>>
  return
}

// -----
func.func @bad_parent_comp3(%arg0: !fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'hlfir.parent_comp' op result type extents are inconsistent with memref type}}
  %2 = hlfir.parent_comp %arg0 shape %1 : (!fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<20x!fir.type<t1{i:i32}>>>
  return
}

// -----
func.func @bad_parent_comp4(%arg0: !fir.ref<!fir.type<t2{i:i32,j:i32}>>) {
  // expected-error@+1 {{'hlfir.parent_comp' op result type and input type must be derived types}}
  %1 = hlfir.parent_comp %arg0 : (!fir.ref<!fir.type<t2{i:i32,j:i32}>>) -> !fir.ref<i32>
  return
}

// -----
func.func @bad_parent_comp5(%arg0: !fir.class<!fir.type<t2{i:i32,j:i32}>>) {
  // expected-error@+1 {{'hlfir.parent_comp' op result type must not be polymorphic}}
  %2 = hlfir.parent_comp %arg0 : (!fir.class<!fir.type<t2{i:i32,j:i32}>>) -> !fir.class<!fir.type<t1{i:i32}>>
  return
}

// -----
func.func @bad_parent_comp6(%arg0: !fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'hlfir.parent_comp' op result type must be a fir.box if the result is an array or has length parameters}}
  %2 = hlfir.parent_comp %arg0 shape %1 : (!fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<10x!fir.type<t1{i:i32}>>>
  return
}

// -----
func.func @bad_shapeof(%arg0: !hlfir.expr<!fir.char<1,10>>) {
  // expected-error@+1 {{'hlfir.shape_of' op cannot get the shape of a shape-less expression}}
  %0 = hlfir.shape_of %arg0 : (!hlfir.expr<!fir.char<1,10>>) -> !fir.shape<1>
}

// -----
func.func @bad_shapeof2(%arg0: !hlfir.expr<10xi32>) {
  // expected-error@+1 {{'hlfir.shape_of' op result rank and expr rank do not match}}
  %0 = hlfir.shape_of %arg0 : (!hlfir.expr<10xi32>) -> !fir.shape<42>
}

// -----
func.func @bad_getextent(%arg0: !fir.shape<1>) {
  // expected-error@+1 {{'hlfir.get_extent' op dimension index out of bounds}}
  %0 = hlfir.get_extent %arg0 {dim = 1 : index} : (!fir.shape<1>) -> index
}

// -----
func.func @bad_region_assign_1(%x: !fir.box<!fir.array<?xf32>>) {
// expected-error@+1 {{'hlfir.region_assign' op right-hand side region must be terminated by an hlfir.yield}}
  hlfir.region_assign {
    %c100 = arith.constant 100 : index
  } to {
    hlfir.yield %x : !fir.box<!fir.array<?xf32>>
  }
  return
}

// -----
func.func @bad_region_assign_2(%x: !fir.box<!fir.array<?xf32>>) {
// expected-error@+1 {{'hlfir.region_assign' op left-hand side region must be terminated by an hlfir.yield or hlfir.elemental_addr}}
  hlfir.region_assign {
    hlfir.yield %x : !fir.box<!fir.array<?xf32>>
  } to {
    %c100 = arith.constant 100 : index
  } user_defined_assign  (%rhs: !fir.ref<i64>) to (%lhs: !fir.ref<f32>) {
  }
  return
}

// -----
func.func @bad_element_addr_1(%x: !fir.ref<!fir.array<20xf32>>) {
  %c20 = arith.constant 20 : index
  %vector_shape = fir.shape %c20 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    hlfir.yield %x : !fir.ref<!fir.array<20xf32>>
  } to {
    // expected-error@+1 {{'hlfir.elemental_addr' op body must compute the address of a scalar entity}}
    hlfir.elemental_addr %vector_shape  : !fir.shape<1> {
      ^bb0(%i: index):
      %c42 = arith.constant 42.0 : f32
      hlfir.yield %c42 : f32
    }
  }
  return
}

// -----
func.func @bad_element_addr_2(%x: !fir.ref<!fir.array<20xf32>>) {
  %c20 = arith.constant 20 : index
  %vector_shape = fir.shape %c20 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    hlfir.yield %x : !fir.ref<!fir.array<20xf32>>
  } to {
    // expected-error@+1 {{'hlfir.elemental_addr' op body must compute the address of a scalar entity}}
    hlfir.elemental_addr %vector_shape  : !fir.shape<1> {
      ^bb0(%i: index):
      hlfir.yield %x : !fir.ref<!fir.array<20xf32>>
    }
  }
  return
}

// -----
func.func @bad_element_addr_3(%x: !fir.ref<!fir.array<20xf32>>) {
  %c20 = arith.constant 20 : index
  %vector_shape = fir.shape %c20 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    hlfir.yield %x : !fir.ref<!fir.array<20xf32>>
  } to {
    // expected-error@+1 {{'hlfir.elemental_addr' op body region must be terminated by an hlfir.yield}}
    hlfir.elemental_addr %vector_shape  : !fir.shape<1> {
      ^bb0(%i: index):
      %c42 = arith.constant 42.0 : f32
    }
  }
  return
}

// -----
func.func @bad_element_addr_4(%x: !fir.ref<!fir.array<20xf32>>, %y: !fir.ref<!fir.array<20x20xf32>>) {
  %c20 = arith.constant 20 : index
  %vector_shape = fir.shape %c20 : (index) -> !fir.shape<1>
  hlfir.region_assign {
    hlfir.yield %x : !fir.ref<!fir.array<20xf32>>
  } to {
    // expected-error@+1 {{'hlfir.elemental_addr' op body number of indices must match shape rank}}
    hlfir.elemental_addr %vector_shape  : !fir.shape<1> {
      ^bb0(%i: index, %j: index):
       %elt = hlfir.designate %y(%i, %j) : (!fir.ref<!fir.array<20x20xf32>>, index, index) -> !fir.ref<f32>
       hlfir.yield %elt : !fir.ref<f32>
    }
  }
  return
}

// -----
func.func @bad_forall(%x : !fir.box<!fir.array<10xf32>>, %y: f32, %bad : !fir.ref<!fir.array<10xindex>>) {
  // expected-error@+1 {{'hlfir.forall' op region #0 ('lb_region') failed to verify constraint: single block region that yields an integer scalar value}}
  hlfir.forall lb {
    hlfir.yield %bad : !fir.ref<!fir.array<10xindex>>
  } ub {
    %c10 = arith.constant 10 : index
    hlfir.yield %c10 : index
  } (%i : index) {
    hlfir.region_assign {
      hlfir.yield %y : f32
    } to {
      %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      hlfir.yield %xi : !fir.ref<f32>
    }
  }
  return
}

// -----
func.func @bad_forall_2(%x : !fir.box<!fir.array<10xf32>>, %y: f32) {
  // expected-error@+1 {{'hlfir.forall' op body region must only contain OrderedAssignmentTreeOpInterface operations or fir.end}}
  hlfir.forall lb {
    %c1 = arith.constant 1 : index
    hlfir.yield %c1 : index
  } ub {
    %c10 = arith.constant 10 : index
    hlfir.yield %c10 : index
  } (%i : index) {
    %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<10xf32>>, index) -> !fir.ref<f32>
    hlfir.assign %y to %xi : f32, !fir.ref<f32>
  }
  return
}

// -----
func.func @bad_forall_mask(%i: index) {
  // expected-error@+1 {{'hlfir.forall_mask' op must be inside the body region of an hlfir.forall}}
  hlfir.forall_mask {
    %mask = fir.call @some_condition(%i) : (index) -> i1
    hlfir.yield %mask : i1
  } do {
  }
  return
}

// -----
func.func @bad_forall_mask_2(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
  %c1 = arith.constant 1 : index
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c1 : index
  } (%i: index) {
    // expected-error@+1 {{'hlfir.forall_mask' op mask region must yield a scalar i1}}
    hlfir.forall_mask {
      hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
    } do {
    }
  }
  return
}

// -----
func.func @bad_where_1(%bad_mask: !fir.ref<!fir.array<10xf32>>) {
  // expected-error@+1 {{'hlfir.where' op mask region must yield a logical array}}
  hlfir.where {
    hlfir.yield %bad_mask : !fir.ref<!fir.array<10xf32>>
  } do {
  }
  return
}

// -----
func.func @bad_where_2(%bad_mask: i1) {
  // expected-error@+1 {{'hlfir.where' op mask region must yield a logical array}}
  hlfir.where {
    hlfir.yield %bad_mask : i1
  } do {
  }
  return
}

// -----
func.func @bad_where_3(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>, %n: index) {
  // expected-error@+1 {{'hlfir.where' op body region must not contain hlfir.forall}}
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.forall lb {
      hlfir.yield %n : index
    } ub {
      hlfir.yield %n : index
    } (%i: index) {
    }
  }
  return
}

// -----
func.func @bad_elsewhere_1(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>, %bad_mask: i1) {
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    // expected-error@+1 {{'hlfir.elsewhere' op mask region must yield a logical array when provided}}
    hlfir.elsewhere mask {
      hlfir.yield %bad_mask : i1
    } do {
    }
  }
  return
}

// -----
func.func @bad_elsewhere_2(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
  // expected-error@+1 {{'hlfir.elsewhere' op expects parent op to be one of 'hlfir.where, hlfir.elsewhere'}}
  hlfir.elsewhere mask {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
  }
  return
}

// -----
func.func @bad_elsewhere_3(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>, %x: !fir.ref<!fir.array<10xf32>>, %y: !fir.box<!fir.array<?xf32>>) {
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    // expected-error@+1 {{'hlfir.elsewhere' op must be the last operation in the parent block}}
    hlfir.elsewhere mask {
      hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
    } do {
    }
    hlfir.region_assign {
      hlfir.yield %y : !fir.box<!fir.array<?xf32>>
    } to {
      hlfir.yield %x : !fir.ref<!fir.array<10xf32>>
    }
  }
  return
}

// -----
func.func @bad_get_length_1(%arg0: !hlfir.expr<i32>) {
  // expected-error@+1 {{'hlfir.get_length' op operand #0 must be any character scalar or array expression type, but got '!hlfir.expr<i32>'}}
  %1 = hlfir.get_length %arg0 : (!hlfir.expr<i32>) -> index
  return
}

// -----
func.func @bad_get_length_2(%arg0: !hlfir.expr<?xi32>) {
  // expected-error@+1 {{'hlfir.get_length' op operand #0 must be any character scalar or array expression type, but got '!hlfir.expr<?xi32>'}}
  %1 = hlfir.get_length %arg0 : (!hlfir.expr<?xi32>) -> index
  return
}

// -----
func.func @bad_get_length_3(%arg0: !hlfir.expr<!fir.boxchar<1>>) {
  // expected-error@+1 {{'hlfir.get_length' op operand #0 must be any character scalar or array expression type, but got '!hlfir.expr<!fir.boxchar<1>>'}}
  %1 = hlfir.get_length %arg0 : (!hlfir.expr<!fir.boxchar<1>>) -> index
  return
}

// -----
func.func @elemental_poly_1(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, %shape : index) {
  %3 = fir.shape %shape : (index) -> !fir.shape<1>
  // expected-error@+1 {{'hlfir.elemental' op operand #1 must be any polymorphic object, but got '!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>'}}
  %4 = hlfir.elemental %3 mold %arg0 unordered : (!fir.shape<1>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?> {
  ^bb0(%arg2: index):
    %6 = fir.undefined !hlfir.expr<!fir.type<_QMtypesTt>?>
    hlfir.yield_element %6 : !hlfir.expr<!fir.type<_QMtypesTt>?>
  }
  return
}

// -----
func.func @elemental_poly_2(%arg0: !hlfir.expr<?x!fir.type<_QMtypesTt>>, %shape : index) {
  %3 = fir.shape %shape : (index) -> !fir.shape<1>
  // expected-error@+1 {{'hlfir.elemental' op operand #1 must be any polymorphic object, but got '!hlfir.expr<?x!fir.type<_QMtypesTt>>'}}
  %4 = hlfir.elemental %3 mold %arg0 unordered : (!fir.shape<1>, !hlfir.expr<?x!fir.type<_QMtypesTt>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?> {
  ^bb0(%arg2: index):
    %6 = fir.undefined !hlfir.expr<!fir.type<_QMtypesTt>?>
    hlfir.yield_element %6 : !hlfir.expr<!fir.type<_QMtypesTt>?>
  }
  return
}

// -----
func.func @elemental_poly_3(%arg0: !hlfir.expr<?x!fir.type<_QMtypesTt>?>, %shape : index) {
  %3 = fir.shape %shape : (index) -> !fir.shape<1>
// expected-error@+1 {{'hlfir.elemental' op result must be polymorphic when mold is present and vice versa}}
  %4 = hlfir.elemental %3 mold %arg0 unordered : (!fir.shape<1>, !hlfir.expr<?x!fir.type<_QMtypesTt>?>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>> {
  ^bb0(%arg2: index):
    %6 = fir.undefined !hlfir.expr<!fir.type<_QMtypesTt>>
    hlfir.yield_element %6 : !hlfir.expr<!fir.type<_QMtypesTt>>
  }
  return
}

// -----
func.func @elemental_poly_4(%shape : index) {
  %3 = fir.shape %shape : (index) -> !fir.shape<1>
// expected-error@+1 {{'hlfir.elemental' op result must be polymorphic when mold is present and vice versa}}
  %4 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?> {
  ^bb0(%arg2: index):
    %6 = fir.undefined !hlfir.expr<!fir.type<_QMtypesTt>?>
    hlfir.yield_element %6 : !hlfir.expr<!fir.type<_QMtypesTt>?>
  }
  return
}

// -----
func.func @destroy_with_finalize(%expr: !hlfir.expr<?xi32>) {
// expected-error@+1 {{'hlfir.destroy' op the element type must be finalizable, when 'finalize' is set}}
  hlfir.destroy %expr finalize : !hlfir.expr<?xi32>
  return
}

// -----

func.func @end_associate_with_alloc_comp(%var: !hlfir.expr<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, %shape: !fir.shape<1>) {
  %4:3 = hlfir.associate %var(%shape) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1)
// expected-error@+1 {{'hlfir.end_associate' op that requires components deallocation must have var operand that is a Fortran entity}}
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.array<?x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, i1
  return
}

// -----

func.func @bad_eval_in_mem_1() {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
// expected-error@+1 {{'hlfir.eval_in_mem' op result #0 must be The type of an array, character, or derived type Fortran expression, but got '!fir.array<10xf32>'}}
  %2 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !fir.array<10xf32> {
  ^bb0(%arg0: !fir.ref<!fir.array<10xf32>>):
  }
  return
}

// -----

func.func @bad_eval_in_mem_2() {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'hlfir.eval_in_mem' op `shape` rank must match the result rank}}
  %2 = hlfir.eval_in_mem shape %1 : (!fir.shape<2>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg0: !fir.ref<!fir.array<10xf32>>):
  }
  return
}

// -----

func.func @bad_eval_in_mem_3() {
  // expected-error@+1 {{'hlfir.eval_in_mem' op must be provided one length parameter when the result is a character}}
  %1 = hlfir.eval_in_mem  : () -> !hlfir.expr<!fir.char<1,?>> {
  ^bb0(%arg0: !fir.ref<!fir.char<1,?>>):
  }
  return
}

// -----

func.func @bad_cshift1(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.cshift' op input and output arrays should have the same element type}}
  %0 = hlfir.cshift %arg0 %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<?x?xf32>
  return
}

// -----

func.func @bad_cshift2(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.cshift' op input and output arrays should have the same rank}}
  %0 = hlfir.cshift %arg0 %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_cshift3(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.cshift' op output array's shape conflicts with the input array's shape}}
  %0 = hlfir.cshift %arg0 %arg1 : (!hlfir.expr<2x2xi32>, i32) -> !hlfir.expr<2x3xi32>
  return
}

// -----

func.func @bad_cshift4(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'hlfir.cshift' op DIM must be >= 1}}
  %0 = hlfir.cshift %arg0 %arg1 dim %c0 : (!hlfir.expr<2x2xi32>, i32, index) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_cshift5(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32) {
  %c10 = arith.constant 10 : index
  // expected-error@+1 {{'hlfir.cshift' op DIM must be <= input array's rank}}
  %0 = hlfir.cshift %arg0 %arg1 dim %c10 : (!hlfir.expr<2x2xi32>, i32, index) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_cshift6(%arg0: !hlfir.expr<2x2xi32>, %arg1: !hlfir.expr<2x2xi32>) {
  // expected-error@+1 {{'hlfir.cshift' op SHIFT's rank must be 1 less than the input array's rank}}
  %0 = hlfir.cshift %arg0 %arg1 : (!hlfir.expr<2x2xi32>, !hlfir.expr<2x2xi32>) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_cshift7(%arg0: !hlfir.expr<?x2xi32>, %arg1: !hlfir.expr<3xi32>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.cshift' op SHAPE(ARRAY)(2) must be equal to SHAPE(SHIFT)(1): 2 != 3}}
  %0 = hlfir.cshift %arg0 %arg1 dim %c1 : (!hlfir.expr<?x2xi32>, !hlfir.expr<3xi32>, index) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_cshift8(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32) {
  // expected-error@+2 {{'hlfir.cshift' op character KIND mismatch}}
  // expected-error@+1 {{'hlfir.cshift' op input and output arrays should have the same element type}}
  %0 = hlfir.cshift %arg0 %arg1 : (!hlfir.expr<?x!fir.char<1,?>>, i32) -> !hlfir.expr<?x!fir.char<2,?>>
  return
}

// -----

func.func @bad_cshift9(%arg0: !hlfir.expr<?x!fir.char<1,1>>, %arg1: i32) {
  // expected-error@+2 {{'hlfir.cshift' op character LEN mismatch}}
  // expected-error@+1 {{'hlfir.cshift' op input and output arrays should have the same element type}}
  %0 = hlfir.cshift %arg0 %arg1 : (!hlfir.expr<?x!fir.char<1,1>>, i32) -> !hlfir.expr<?x!fir.char<1,2>>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op ARRAY and the result must have the same element type}}
  %0 = hlfir.reshape %arg0 %arg0 : (!hlfir.expr<1xi32>, !hlfir.expr<1xi32>) -> !hlfir.expr<?xf32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<?x!fir.type<whatever>?>, %arg1: !hlfir.expr<1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op ARRAY must be polymorphic iff result is polymorphic}}
  %0 = hlfir.reshape %arg0 %arg1 : (!hlfir.expr<?x!fir.type<whatever>?>, !hlfir.expr<1xi32>) -> !hlfir.expr<?x!fir.type<whatever>>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<?x!fir.type<whatever>>, %arg1: !hlfir.expr<1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op ARRAY must be polymorphic iff result is polymorphic}}
  %0 = hlfir.reshape %arg0 %arg1 : (!hlfir.expr<?x!fir.type<whatever>>, !hlfir.expr<1xi32>) -> !hlfir.expr<?x!fir.type<whatever>?>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1x1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op SHAPE must be an array of rank 1}}
  %0 = hlfir.reshape %arg0 %arg0 : (!hlfir.expr<1x1xi32>, !hlfir.expr<1x1xi32>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1xf32>) {
  // expected-error@+1 {{'hlfir.reshape' op SHAPE must be an integer array}}
  %0 = hlfir.reshape %arg0 %arg0 : (!hlfir.expr<1xf32>, !hlfir.expr<1xf32>) -> !hlfir.expr<?xf32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<?xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op SHAPE must have known size}}
  %0 = hlfir.reshape %arg0 %arg0 : (!hlfir.expr<?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op SHAPE's extent must match the result rank}}
  %0 = hlfir.reshape %arg0 %arg0 : (!hlfir.expr<1xi32>, !hlfir.expr<1xi32>) -> !hlfir.expr<?x?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1xi32>, %arg1: !hlfir.expr<?xi16>) {
  // expected-error@+1 {{'hlfir.reshape' op ARRAY and PAD must be of the same type}}
  %0 = hlfir.reshape %arg0 %arg0 pad %arg1 : (!hlfir.expr<1xi32>, !hlfir.expr<1xi32>, !hlfir.expr<?xi16>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1xi32>, %arg1: !hlfir.expr<?x?xi16>) {
  // expected-error@+1 {{'hlfir.reshape' op ORDER must be an array of rank 1}}
  %0 = hlfir.reshape %arg0 %arg0 order %arg1 : (!hlfir.expr<1xi32>, !hlfir.expr<1xi32>, !hlfir.expr<?x?xi16>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1xi32>, %arg1: !hlfir.expr<?xf16>) {
  // expected-error@+1 {{'hlfir.reshape' op ORDER must be an integer array}}
  %0 = hlfir.reshape %arg0 %arg0 order %arg1 : (!hlfir.expr<1xi32>, !hlfir.expr<1xi32>, !hlfir.expr<?xf16>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !hlfir.expr<1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op operand #0 must be any array-like entity}}
  %0 = hlfir.reshape %arg0 %arg1 : (!fir.ref<!fir.array<?xi32>>, !hlfir.expr<1xi32>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !hlfir.expr<?xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op operand #1 must be any array-like entity containing a numerical type}}
  %0 = hlfir.reshape %arg1 %arg0 : (!hlfir.expr<?xi32>, !fir.ref<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !hlfir.expr<1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op operand #2 must be any array-like entity}}
  %0 = hlfir.reshape %arg1 %arg1 pad %arg0 : (!hlfir.expr<1xi32>, !hlfir.expr<1xi32>, !fir.ref<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !fir.ref<!fir.array<?xi32>>, %arg1: !hlfir.expr<1xi32>) {
  // expected-error@+1 {{'hlfir.reshape' op operand #3 must be any array-like entity containing a numerical type}}
  %0 = hlfir.reshape %arg1 %arg1 pad %arg1 order %arg0 : (!hlfir.expr<1xi32>, !hlfir.expr<1xi32>, !hlfir.expr<1xi32>, !fir.ref<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1x!fir.char<1,2>>, %arg1: !hlfir.expr<1xi32>) {
  // expected-error@+2 {{'hlfir.reshape' op character KIND mismatch}}
  // expected-error@+1 {{'hlfir.reshape' op ARRAY and the result must have the same element type}}
  %0 = hlfir.reshape %arg0 %arg1 : (!hlfir.expr<1x!fir.char<1,2>>, !hlfir.expr<1xi32>) -> !hlfir.expr<?x!fir.char<2,?>>
  return
}

// -----

func.func @bad_reshape(%arg0: !hlfir.expr<1x!fir.char<1,2>>, %arg1: !hlfir.expr<1xi32>, %arg2: !hlfir.expr<1x!fir.char<2,?>>) {
  // expected-error@+2 {{'hlfir.reshape' op character KIND mismatch}}
  // expected-error@+1 {{'hlfir.reshape' op ARRAY and PAD must be of the same type}}
  %0 = hlfir.reshape %arg0 %arg1 pad %arg2 : (!hlfir.expr<1x!fir.char<1,2>>, !hlfir.expr<1xi32>, !hlfir.expr<1x!fir.char<2,?>>) -> !hlfir.expr<?x!fir.char<1,?>>
  return
}

// -----

func.func @bad_eoshift1(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.eoshift' op input and output arrays should have the same element type}}
  %0 = hlfir.eoshift %arg0 %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<?x?xf32>
  return
}

// -----

func.func @bad_eoshift2(%arg0: !hlfir.expr<?x?xi32>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.eoshift' op input and output arrays should have the same rank}}
  %0 = hlfir.eoshift %arg0 %arg1 : (!hlfir.expr<?x?xi32>, i32) -> !hlfir.expr<?xi32>
  return
}

// -----

func.func @bad_eoshift3(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32) {
  // expected-error@+1 {{'hlfir.eoshift' op output array's shape conflicts with the input array's shape}}
  %0 = hlfir.eoshift %arg0 %arg1 : (!hlfir.expr<2x2xi32>, i32) -> !hlfir.expr<2x3xi32>
  return
}

// -----

func.func @bad_eoshift4(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{'hlfir.eoshift' op DIM must be >= 1}}
  %0 = hlfir.eoshift %arg0 %arg1 dim %c0 : (!hlfir.expr<2x2xi32>, i32, index) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_eoshift5(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32) {
  %c10 = arith.constant 10 : index
  // expected-error@+1 {{'hlfir.eoshift' op DIM must be <= input array's rank}}
  %0 = hlfir.eoshift %arg0 %arg1 dim %c10 : (!hlfir.expr<2x2xi32>, i32, index) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_eoshift6(%arg0: !hlfir.expr<2x2xi32>, %arg1: !hlfir.expr<2x2xi32>) {
  // expected-error@+1 {{'hlfir.eoshift' op SHIFT's rank must be 1 less than the input array's rank}}
  %0 = hlfir.eoshift %arg0 %arg1 : (!hlfir.expr<2x2xi32>, !hlfir.expr<2x2xi32>) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_eoshift7(%arg0: !hlfir.expr<?x2xi32>, %arg1: !hlfir.expr<3xi32>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'hlfir.eoshift' op SHAPE(ARRAY)(2) must be equal to SHAPE(SHIFT)(1): 2 != 3}}
  %0 = hlfir.eoshift %arg0 %arg1 dim %c1 : (!hlfir.expr<?x2xi32>, !hlfir.expr<3xi32>, index) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_eoshift8(%arg0: !hlfir.expr<?x!fir.char<1,?>>, %arg1: i32) {
  // expected-error@+2 {{'hlfir.eoshift' op character KIND mismatch}}
  // expected-error@+1 {{'hlfir.eoshift' op input and output arrays should have the same element type}}
  %0 = hlfir.eoshift %arg0 %arg1 : (!hlfir.expr<?x!fir.char<1,?>>, i32) -> !hlfir.expr<?x!fir.char<2,?>>
  return
}

// -----

func.func @bad_eoshift9(%arg0: !hlfir.expr<?x!fir.char<1,1>>, %arg1: i32) {
  // expected-error@+2 {{'hlfir.eoshift' op character LEN mismatch}}
  // expected-error@+1 {{'hlfir.eoshift' op input and output arrays should have the same element type}}
  %0 = hlfir.eoshift %arg0 %arg1 : (!hlfir.expr<?x!fir.char<1,1>>, i32) -> !hlfir.expr<?x!fir.char<1,2>>
  return
}

// -----

func.func @bad_eoshift10(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32, %arg2: f32) {
  // expected-error@+1 {{'hlfir.eoshift' op ARRAY and BOUNDARY operands must have the same element type}}
  %0 = hlfir.eoshift %arg0 %arg1 boundary %arg2 : (!hlfir.expr<2x2xi32>, i32, f32) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @bad_eoshift11(%arg0: !hlfir.expr<2x2xi32>, %arg1: i32, %arg2: !hlfir.expr<2x2xi32>) {
  // expected-error@+1 {{'hlfir.eoshift' op BOUNDARY's rank must be 1 less than the input array's rank}}
  %0 = hlfir.eoshift %arg0 %arg1 boundary %arg2 : (!hlfir.expr<2x2xi32>, i32, !hlfir.expr<2x2xi32>) -> !hlfir.expr<2x2xi32>
  return
}

// -----

func.func @fir_declare_bad_storage_offset(%arg0: !fir.ref<!fir.array<8xi8>>) {
  %c0 = arith.constant 0 : index
  %addr = fir.address_of(@block_) : !fir.ref<!fir.array<8xi8>>
  %2 = fir.convert %addr : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %var = fir.coordinate_of %2, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  // expected-error@+1 {{negative integer literal not valid for unsigned integer type}}
  %decl:2 = hlfir.declare %var storage (%addr[-1]) {uniq_name = "a"} : (!fir.ref<i8>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ref<i8>, !fir.ref<i8>)
  return
}

// -----

"func.func"() <{function_type = (!fir.ref<!fir.array<8xi8>>) -> (), sym_name = "fir_declare_bad_storage_offset"}> ({
^bb0(%arg0: !fir.ref<!fir.array<8xi8>>):
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "fir.address_of"() <{symbol = @block_}> : () -> !fir.ref<!fir.array<8xi8>>
  %2 = "fir.convert"(%1) : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %3 = "fir.coordinate_of"(%2, %0) <{baseType = !fir.ref<!fir.array<?xi8>>}> : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
// expected-error@+1 {{storage offset specified without the storage reference}}
  %4:2 = "hlfir.declare"(%3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, storage_offset = 1 : ui64, uniq_name = "a"}> : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
  "func.return"() : () -> ()
}) : () -> ()
