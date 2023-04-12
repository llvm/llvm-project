// HLFIR ops diagnotic tests

// RUN: fir-opt -split-input-file -verify-diagnostics %s

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
func.func @bad_designate_out_type(%arg0 : !fir.ref<!fir.complex<4>>) {
  // expected-error@+1 {{'hlfir.designate' op result element type is not consistent with operands, expected '!fir.real<4>'}}
  %0 = hlfir.designate %arg0 imag:  (!fir.ref<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
  return
}

// -----
func.func @bad_designate_out_type(%arg0 : !fir.ref<!fir.box<!fir.complex<4>>>) {
  // expected-error@+1 {{'hlfir.designate' op result type must only be a box address type if it designates a component that is a fir.box or fir.class and if there are no indices, substrings, and complex part}}
  %0 = hlfir.designate %arg0 imag:  (!fir.ref<!fir.box<!fir.complex<4>>>) -> !fir.ref<!fir.box<!fir.real<4>>>
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
  // expected-error@+1 {{'hlfir.concat' op operand #0 must be any character scalar type, but got '!fir.ref<!fir.array<100x!fir.char<1,10>>>'}}
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>, index) -> (!hlfir.expr<100x!fir.char<1,30>>)
  return
}

// -----
func.func @bad_concat_3(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<i32>) {
  %c30 = arith.constant 30 : index
  // expected-error@+1 {{'hlfir.concat' op operand #1 must be any character scalar type, but got '!fir.ref<i32>'}}
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
func.func @bad_sum1(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.sum' op result must have the same element type as ARRAY argument}}
  %0 = hlfir.sum %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<f32>
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
func.func @bad_sum4(%arg0: !hlfir.expr<?xi32>, %arg1: i32, %arg2: !fir.box<!fir.logical<4>>) {
  // expected-error@+1 {{'hlfir.sum' op result rank must be one less than ARRAY}}
  %0 = hlfir.sum %arg0 dim %arg1 mask %arg2 : (!hlfir.expr<?xi32>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x?xi32>
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
