// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @load_scalar
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]] {uniq_name = "a"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<f32>) -> memref<f32>
// CHECK:       [[LOAD:%[0-9]+]] = memref.load [[CONVERT]][] : memref<f32>
func.func @load_scalar(%arg0: !fir.ref<f32>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "a"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  return
}


// CHECK-LABEL: func.func @load_array1d_const
// CHECK:       [[C1:%.+]] = arith.constant 1 : index
// CHECK:       [[C3:%.+]] = arith.constant 3 : index
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[SHAPE:%[0-9]+]] = fir.shape [[C3]] : (index) -> !fir.shape<1>
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHAPE]]) dummy_scope [[DUMMY]] {uniq_name = "a"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xf32>>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<3xf32>>) -> memref<3xf32>
// CHECK:       [[C1_0:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB:%[0-9]+]] = arith.subi [[C1]], [[C1_0]] : index
// CHECK:       [[MUL:%[0-9]+]] = arith.muli [[SUB]], [[C1_0]] : index
// CHECK:       [[SUBA:%[0-9]+]] = arith.subi [[C1_0]], [[C1_0]] : index
// CHECK:       [[ADD:%[0-9]+]] = arith.addi [[MUL]], [[SUBA]] : index
// CHECK:       [[LOAD:%[0-9]+]] = memref.load [[CONVERT]][[[ADD]]] : memref<3xf32>
func.func @load_array1d_const(%arg0: !fir.ref<!fir.array<3xf32>>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%shape) dummy_scope %0 {uniq_name = "a"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xf32>>
  %2 = fir.array_coor %1(%shape) %c1 : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %3 = fir.load %2 : !fir.ref<f32>
  return
}

// CHECK-LABEL: func.func @load_array2d_const
// CHECK:       [[C2:%.+]] = arith.constant 2 : index
// CHECK:       [[C3:%.+]] = arith.constant 3 : index
// CHECK:       [[C6:%.+]] = arith.constant 6 : index
// CHECK:       [[C5:%.+]] = arith.constant 5 : index
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[SHAPE:%[0-9]+]] = fir.shape [[C5]], [[C6]] : (index, index) -> !fir.shape<2>
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHAPE]]) dummy_scope [[DUMMY]] {uniq_name = "a"} : (!fir.ref<!fir.array<5x6xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<5x6xf32>>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<5x6xf32>>) -> memref<6x5xf32>
// CHECK:       [[C1:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB1:%[0-9]+]] = arith.subi [[C2]], [[C1]] : index
// CHECK:       [[MUL1:%[0-9]+]] = arith.muli [[SUB1]], [[C1]] : index
// CHECK:       [[SUB1A:%[0-9]+]] = arith.subi [[C1]], [[C1]] : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi [[MUL1]], [[SUB1A]] : index
// CHECK:       [[SUB2:%[0-9]+]] = arith.subi [[C3]], [[C1]] : index
// CHECK:       [[MUL2:%[0-9]+]] = arith.muli [[SUB2]], [[C1]] : index
// CHECK:       [[SUB2A:%[0-9]+]] = arith.subi [[C1]], [[C1]] : index
// CHECK:       [[ADD2:%[0-9]+]] = arith.addi [[MUL2]], [[SUB2A]] : index
// CHECK:       [[LOAD:%[0-9]+]] = memref.load [[CONVERT]][[[ADD2]], [[ADD1]]] : memref<6x5xf32>
func.func @load_array2d_const(%arg0: !fir.ref<!fir.array<5x6xf32>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c5 = arith.constant 5 : index
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%shape) dummy_scope %0 {uniq_name = "a"} : (!fir.ref<!fir.array<5x6xf32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<5x6xf32>>
  %2 = fir.array_coor %1(%shape) %c2, %c3 : (!fir.ref<!fir.array<5x6xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  %3 = fir.load %2 : !fir.ref<f32>
  return
}