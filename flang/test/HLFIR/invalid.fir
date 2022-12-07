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
