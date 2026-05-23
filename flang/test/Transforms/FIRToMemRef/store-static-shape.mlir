// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @store_scalar
// CHECK:       [[CONST7:%.+]] = arith.constant 7 : i32
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]] {uniq_name = "a"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:       memref.store [[CONST7]], [[CONVERT]][] : memref<i32>
func.func @store_scalar(%arg0: !fir.ref<i32>) {
  %c7 = arith.constant 7 : i32
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "a"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  fir.store %c7 to %1 : !fir.ref<i32>
  return
}


// CHECK-LABEL: func.func @store_array1d_const
// CHECK:       [[CONST1A:%.+]] = arith.constant 1 : index
// CHECK:       [[CONST3:%.+]] = arith.constant 3 : index
// CHECK:       [[CONST7:%.+]] = arith.constant 7 : i32
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[SHAPE:%[0-9]+]] = fir.shape [[CONST3]] : (index) -> !fir.shape<1>
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHAPE]]) dummy_scope [[DUMMY]] {uniq_name = "a"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xi32>>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<3xi32>>) -> memref<3xi32>
// CHECK:       [[CONST1B:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB:%.+]] = arith.subi [[CONST1A]], [[CONST1B]] : index
// CHECK:       [[MUL:%.+]] = arith.muli [[SUB]], [[CONST1B]] : index
// CHECK:       [[SUBA:%.+]] = arith.subi [[CONST1B]], [[CONST1B]] : index
// CHECK:       [[ADD:%.+]] = arith.addi [[MUL]], [[SUBA]] : index
// CHECK:       memref.store [[CONST7]], [[CONVERT]][[[ADD]]] : memref<3xi32>
func.func @store_array1d_const(%arg0: !fir.ref<!fir.array<3xi32>>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : i32
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%shape) dummy_scope %0 {uniq_name = "a"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xi32>>
  %2 = fir.array_coor %1(%shape) %c1 : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c7 to %2 : !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @store_array2d_const
// CHECK:       [[CONST2:%.+]] = arith.constant 2 : index
// CHECK:       [[CONST3:%.+]] = arith.constant 3 : index
// CHECK:       [[CONST5:%.+]] = arith.constant 5 : index
// CHECK:       [[CONST6:%.+]] = arith.constant 6 : index
// CHECK:       [[CONST7:%.+]] = arith.constant 7 : i32
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[SHAPE:%[0-9]+]] = fir.shape [[CONST5]], [[CONST6]] : (index, index) -> !fir.shape<2>
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHAPE]]) dummy_scope [[DUMMY]] {uniq_name = "a"} : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<5x6xi32>>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<5x6xi32>>) -> memref<6x5xi32>
// CHECK:       [[CONST1:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB1:%[0-9]+]] = arith.subi [[CONST2]], [[CONST1]] : index
// CHECK:       [[MUL1:%[0-9]+]] = arith.muli [[SUB1]], [[CONST1]] : index
// CHECK:       [[SUB1A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi [[MUL1]], [[SUB1A]] : index
// CHECK:       [[SUB2:%[0-9]+]] = arith.subi [[CONST3]], [[CONST1]] : index
// CHECK:       [[MUL2:%[0-9]+]] = arith.muli [[SUB2]], [[CONST1]] : index
// CHECK:       [[SUB2A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD2:%[0-9]+]] = arith.addi [[MUL2]], [[SUB2A]] : index
// CHECK:       memref.store [[CONST7]], [[CONVERT]][[[ADD2]], [[ADD1]]] : memref<6x5xi32>
func.func @store_array2d_const(%arg0: !fir.ref<!fir.array<5x6xi32>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : i32
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%shape) dummy_scope %0 {uniq_name = "a"} : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<5x6xi32>>
  %2 = fir.array_coor %1(%shape) %c2, %c3 : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
  fir.store %c7 to %2 : !fir.ref<i32>
  return
}
