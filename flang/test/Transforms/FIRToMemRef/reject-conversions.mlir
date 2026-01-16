// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s


// reject zero shaped arrays, so none should be converted to memref
// CHECK-LABEL: func.func @reject_zero_array
// CHECK: fir.do_loop
// CHECK: fir.do_loop
// CHECK: fir.array_coor
// CHECK-NOT: memref.load
func.func @reject_zero_array() {
  %c1 = arith.constant 1 : index
  %c-2 = arith.constant -2 : index
  %c0 = arith.constant 0 : index
  %c-5 = arith.constant -5 : index
  %c-3 = arith.constant -3 : index
  %c-6 = arith.constant -6 : index
  %1 = fir.address_of(@_QFEc) : !fir.ref<!fir.array<0x0xi32>>
  %2 = fir.shape_shift %c-2, %c0, %c-5, %c0 : (index, index, index, index) -> !fir.shapeshift<2>
  %3 = fir.declare %1(%2) {uniq_name = "c"} : (!fir.ref<!fir.array<0x0xi32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<0x0xi32>>
  fir.do_loop %arg0 = %c1 to %c0 step %c1 unordered {
    fir.do_loop %arg1 = %c1 to %c0 step %c1 unordered {
      %11 = arith.addi %arg1, %c-3 : index
      %12 = arith.addi %arg0, %c-6 : index
      %13 = fir.array_coor %3(%2) %11, %12 : (!fir.ref<!fir.array<0x0xi32>>, !fir.shapeshift<2>, index, index) -> !fir.ref<i32>
      %14 = fir.load %13 : !fir.ref<i32>
    }
  }
  return
}
