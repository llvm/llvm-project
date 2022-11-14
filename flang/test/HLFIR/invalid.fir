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
