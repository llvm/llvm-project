// Test conversions to index

// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @load_array1d_var
// CHECK:       [[C3:%.+]]       = arith.constant 3 : index
// CHECK:       [[DUMMY:%.+]]    = fir.undefined !fir.dscope
// CHECK:       [[SHAPE:%.+]]    = fir.shape [[C3]] : (index) -> !fir.shape<1>
// CHECK:       [[DECLARE0:%.+]] = fir.declare %arg0([[SHAPE]]) dummy_scope [[DUMMY]]
// CHECK:       [[DECLARE1:%.+]] = fir.declare %arg1 dummy_scope [[DUMMY]]
// CHECK:       [[CONVERT1:%.+]] = fir.convert [[DECLARE1]] : (!fir.ref<i64>) -> memref<i64>
// CHECK:       [[LOAD:%[0-9]+]] = memref.load [[CONVERT1]][] : memref<i64>
// CHECK:       [[CONVERT0:%.+]] = fir.convert [[DECLARE0]] : (!fir.ref<!fir.array<3xf32>>) -> memref<3xf32>
// CHECK:       [[C1:%.+]]       = arith.constant 1 : index
// CHECK-NEXT:  [[C0:%.+]]       = arith.constant 0 : index
// CHECK-NEXT:  [[CAST:%[0-9]+]] = arith.index_cast [[LOAD]] : i64 to index
// CHECK:       [[SUB:%.+]]      = arith.subi [[CAST]], [[C1]] : index
// CHECK-NEXT:  [[MUL:%.+]]      = arith.muli [[SUB]], [[C1]] : index
// CHECK-NEXT:  [[SUBA:%.+]]     = arith.subi [[C1]], [[C1]] : index
// CHECK-NEXT:  [[ADD:%.+]]      = arith.addi [[MUL]], [[SUBA]] : index
// CHECK-NEXT:  memref.load [[CONVERT0]][[[ADD]]] : memref<3xf32>
func.func @load_array1d_var(%arg0: !fir.ref<!fir.array<3xf32>>, %arg1: !fir.ref<i64>) {
  %c3 = arith.constant 3 : index
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%shape) dummy_scope %0 {uniq_name = "a"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xf32>>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "i"} : (!fir.ref<i64>, !fir.dscope) -> !fir.ref<i64>
  %3 = fir.load %2 : !fir.ref<i64>
  %4 = fir.array_coor %1(%shape) %3 : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
  %5 = fir.load %4 : !fir.ref<f32>
  return
}