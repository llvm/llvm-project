// Test rewrite of fir.declare. The result is replaced by the memref operand.
// RUN: fir-opt --cg-rewrite %s -o - | FileCheck %s


func.func @test(%arg0: !fir.ref<!fir.array<12x23xi32>>) {
  %c-1 = arith.constant -1 : index
  %c12 = arith.constant 12 : index
  %c-2 = arith.constant -2 : index
  %c23 = arith.constant 23 : index
  %0 = fir.shape_shift %c12, %c-1, %c23, %c-2 : (index, index, index, index) -> !fir.shapeshift<2>
  %1 = fir.declare %arg0(%0) {uniq_name = "_QFarray_numeric_lboundsEx"} : (!fir.ref<!fir.array<12x23xi32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<12x23xi32>>
  fir.call @bar(%1) : (!fir.ref<!fir.array<12x23xi32>>) -> ()
  return
}
func.func private @bar(%arg0: !fir.ref<!fir.array<12x23xi32>>)


// CHECK-LABEL: func.func @test(
// CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<12x23xi32>>) {
// CHECK-NEXT: fir.call @bar(%[[arg0]]) : (!fir.ref<!fir.array<12x23xi32>>) -> ()

func.func @useless_shape_with_duplicate_extent_operand(%arg0: !fir.ref<!fir.array<3x3xf32>>) {
  %c3 = arith.constant 3 : index
  %1 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %2 = fir.declare %arg0(%1) {uniq_name = "u"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  return
}

// CHECK-LABEL: func.func @useless_shape_with_duplicate_extent_operand(
// CHECK-NEXT: return
