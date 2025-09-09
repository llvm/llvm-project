// Test codegen of fir.box_addr inside fir.global
// RUN: tco %s | FileCheck %s

fir.global @x_addr constant : !fir.type<sometype{p:i64}> {
  %c-1 = arith.constant -1 : index
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %c-3 = arith.constant -3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0 = fir.undefined !fir.type<sometype{p:i64}>
  %1 = fir.address_of(@_QFEx) : !fir.ref<!fir.array<2x3x5x!fir.type<_QFTt1{c:i32}>>>
  %2 = fir.shape_shift %c1, %c2, %c-3, %c3, %c1, %c5 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
  %3 = fir.field_index c, !fir.type<_QFTt1{c:i32}>
  %4 = fir.slice %c1, %c2, %c1, %c-3, %c-1, %c1, %c1, %c5, %c1 path %3 : (index, index, index, index, index, index, index, index, index, !fir.field) -> !fir.slice<3>
  %5 = fir.embox %1(%2) [%4] : (!fir.ref<!fir.array<2x3x5x!fir.type<_QFTt1{c:i32}>>>, !fir.shapeshift<3>, !fir.slice<3>) -> !fir.box<!fir.ref<!fir.array<2x3x5xi32>>>
  %6 = fir.box_addr %5 : (!fir.box<!fir.ref<!fir.array<2x3x5xi32>>>) -> !fir.ref<!fir.array<2x3x5xi32>>
  %7 = fir.convert %6 : (!fir.ref<!fir.array<2x3x5xi32>>) -> i64
  %8 = fir.insert_value %0, %7, ["p", !fir.type<sometype{p:i64}>] : (!fir.type<sometype{p:i64}>, i64) -> !fir.type<sometype{p:i64}>
  fir.has_value %8 : !fir.type<sometype{p:i64}>
}
fir.global @_QFEx target : !fir.array<2x3x5x!fir.type<_QFTt1{c:i32}>>

// CHECK: @x_addr = constant %sometype { i64 ptrtoint (ptr @_QFEx to i64) }
