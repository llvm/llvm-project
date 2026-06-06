// Test rewrite of fir.declare. The result is replaced by the memref operand.
// RUN: fir-opt --cg-rewrite="preserve-declare=true" %s -o - | FileCheck %s --check-prefixes DECL
// RUN: fir-opt --cg-rewrite="preserve-declare=false" %s -o - | FileCheck %s --check-prefixes NODECL
// RUN: fir-opt --cg-rewrite %s -o - | FileCheck %s --check-prefixes NODECL


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


// NODECL-LABEL: func.func @test(
// NODECL-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<12x23xi32>>) {
// NODECL-NEXT: fir.call @bar(%[[arg0]]) : (!fir.ref<!fir.array<12x23xi32>>) -> ()

// DECL-LABEL: func.func @test(
// DECL-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<12x23xi32>>) {
// DECL: fircg.ext_declare


func.func @useless_shape_with_duplicate_extent_operand(%arg0: !fir.ref<!fir.array<3x3xf32>>) {
  %c3 = arith.constant 3 : index
  %1 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %2 = fir.declare %arg0(%1) {uniq_name = "u"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  return
}

// NODECL-LABEL: func.func @useless_shape_with_duplicate_extent_operand(
// NODECL-NEXT: return

// DECL-LABEL: func.func @useless_shape_with_duplicate_extent_operand(
// DECL: fircg.ext_declare

// Test DCE does not crash because of unreachable code.
func.func @unreachable_code(%arg0: !fir.ref<!fir.char<1,10>>) {
  %c10 = arith.constant 10 : index
  %2 = fir.declare %arg0 typeparams %c10 {uniq_name = "live_code"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>)
  return
^bb2:  // no predecessors
  %3 = fir.declare %arg0 typeparams %c10 {uniq_name = "dead_code"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>)
  fir.unreachable
}
// NODECL-LABEL:   func.func @unreachable_code(
// NODECL-NOT:       uniq_name = "live_code"
// DECL-LABEL:    func.func @unreachable_code(
// DECL:             uniq_name = "live_code"

// Test that storage and storage_offset operands are preserved during conversion
func.func @test_storage_operands() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = fir.address_of(@common_block) : !fir.ref<!fir.array<8xi8>>
  %1 = fir.coordinate_of %0, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  %2 = fir.convert %1 : (!fir.ref<i8>) -> !fir.ref<f32>
  %3 = fir.declare %2 storage(%0[0]) {uniq_name = "_QFEx"} : (!fir.ref<f32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<f32>
  %4 = fir.coordinate_of %0, %c4 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  %5 = fir.convert %4 : (!fir.ref<i8>) -> !fir.ref<i32>
  %6 = fir.declare %5 storage(%0[4]) {uniq_name = "_QFEy"} : (!fir.ref<i32>, !fir.ref<!fir.array<8xi8>>) -> !fir.ref<i32>
  return
}
fir.global @common_block : !fir.array<8xi8>

// DECL-LABEL: func.func @test_storage_operands()
// DECL: %[[STORAGE:.*]] = fir.address_of(@common_block) : !fir.ref<!fir.array<8xi8>>
// DECL: fircg.ext_declare {{.*}} storage(%[[STORAGE]][0]) {uniq_name = "_QFEx"}
// DECL: fircg.ext_declare {{.*}} storage(%[[STORAGE]][4]) {uniq_name = "_QFEy"}
