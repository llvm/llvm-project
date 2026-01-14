// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// subroutine store_dynamic_1d(x)
//   real, dimension(:) :: x
//   x(2) = 3
// end subroutine store_dynamic_1d
// CHECK-LABEL: func.func @store_dynamic_1d
// CHECK:       [[CONST2:%.+]] = arith.constant 2 : index
// CHECK:       [[CST:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:       [[DUMMY:%[0-9]+]] = fir.dummy_scope : !fir.dscope
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]] {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK:       [[REBOX:%[0-9]+]] = fir.rebox [[DECLARE]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:       [[BOXADDR:%[0-9]+]] = fir.box_addr [[REBOX]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:       [[CONST1:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB1:%[0-9]+]] = arith.subi [[CONST2]], [[CONST1]] : index
// CHECK:       [[MUL1:%[0-9]+]] = arith.muli [[SUB1]], [[CONST1]] : index
// CHECK:       [[SUB1A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi [[MUL1]], [[SUB1A]] : index
// CHECK:       [[ELSIZE:%[0-9]+]] = fir.box_elesize [[REBOX]] : (!fir.box<!fir.array<?xf32>>) -> index
// CHECK:       [[CONST0:%.+]] = arith.constant 0 : index
// CHECK:       [[DIMS:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[CONST0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:       [[DIV:%[0-9]+]] = arith.divsi [[DIMS]]#2, [[ELSIZE]] : index
// CHECK:       [[CONST0_0:%.+]] = arith.constant 0 : index
// CHECK:       [[REINTERPRET:%.+]] = memref.reinterpret_cast [[CONVERT]] to offset: [[[CONST0_0]]], sizes: [[[DIMS]]#1], strides: [[[DIV]]] : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:       memref.store [[CST]], [[REINTERPRET]][[[ADD1]]] : memref<?xf32, strided<[?], offset: ?>>
func.func @store_dynamic_1d(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c2 = arith.constant 2 : index
  %cst = arith.constant 3.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %3 = fir.array_coor %2 %c2 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  fir.store %cst to %3 : !fir.ref<f32>
  return
}

// subroutine store_dynamic_2d(x)
//   real, dimension(:, :) :: x
//   x(2, 2) = 3
// end subroutine store_dynamic_2d
// CHECK-LABEL: func.func @store_dynamic_2d
// CHECK:       [[CONST2:%.+]] = arith.constant 2 : index
// CHECK:       [[CST:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:       [[DUMMY:%[0-9]+]] = fir.dummy_scope : !fir.dscope
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]] {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:       [[REBOX:%[0-9]+]] = fir.rebox [[DECLARE]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:       [[BOXADDR:%[0-9]+]] = fir.box_addr [[REBOX]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<?x?xf32>>) -> memref<?x?xf32>
// CHECK:       [[CONST1:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB1:%[0-9]+]] = arith.subi [[CONST2]], [[CONST1]] : index
// CHECK:       [[MUL1:%[0-9]+]] = arith.muli [[SUB1]], [[CONST1]] : index
// CHECK:       [[SUB1A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi [[MUL1]], [[SUB1A]] : index
// CHECK:       [[SUB2:%[0-9]+]] = arith.subi [[CONST2]], [[CONST1]] : index
// CHECK:       [[MUL2:%[0-9]+]] = arith.muli [[SUB2]], [[CONST1]] : index
// CHECK:       [[SUB2A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD2:%[0-9]+]] = arith.addi [[MUL2]], [[SUB2A]] : index
// CHECK:       [[ELSIZE:%[0-9]+]] = fir.box_elesize [[REBOX]] : (!fir.box<!fir.array<?x?xf32>>) -> index
// CHECK:       [[CONST1_1:%.+]] = arith.constant 1 : index
// CHECK:       [[DIMS1:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[CONST1_1]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:       [[DIV1:%[0-9]+]] = arith.divsi [[DIMS1]]#2, [[ELSIZE]] : index
// CHECK:       [[CONST0:%.+]] = arith.constant 0 : index
// CHECK:       [[DIMS0:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[CONST0]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:       [[DIV0:%[0-9]+]] = arith.divsi [[DIMS0]]#2, [[ELSIZE]] : index
// CHECK:       [[CONST0_2:%.+]] = arith.constant 0 : index
// CHECK:       [[REINTERPRET:%.+]] = memref.reinterpret_cast [[CONVERT]] to offset: [[[CONST0_2]]], sizes: [[[DIMS1]]#1, [[DIMS0]]#1], strides: [[[DIV1]], [[DIV0]]] : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:       memref.store [[CST]], [[REINTERPRET]][[[ADD2]], [[ADD1]]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
func.func @store_dynamic_2d(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %c2 = arith.constant 2 : index
  %cst = arith.constant 3.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %3 = fir.array_coor %2 %c2, %c2 : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
  fir.store %cst to %3 : !fir.ref<f32>
  return
}

// subroutine store_dynamic_3d(x)
//   real, dimension(:, :, :) :: x
//   x(2, 2, 3) = 3
// end subroutine store_dynamic_3d
// CHECK-LABEL: func.func @store_dynamic_3d
// CHECK:       [[CONST3:%.+]] = arith.constant 3 : index
// CHECK:       [[CONST2:%.+]] = arith.constant 2 : index
// CHECK:       [[CST:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:       [[DUMMY:%[0-9]+]] = fir.dummy_scope : !fir.dscope
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]] {uniq_name = "x"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:       [[REBOX:%[0-9]+]] = fir.rebox [[DECLARE]] : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:       [[BOXADDR:%[0-9]+]] = fir.box_addr [[REBOX]] : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.ref<!fir.array<?x?x?xf32>>
// CHECK:       [[CONVERT:%[0-9]+]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<?x?x?xf32>>) -> memref<?x?x?xf32>
// CHECK:       [[CONST1:%.+]] = arith.constant 1 : index
// CHECK:       [[SUB1:%[0-9]+]] = arith.subi [[CONST2]], [[CONST1]] : index
// CHECK:       [[MUL1:%[0-9]+]] = arith.muli [[SUB1]], [[CONST1]] : index
// CHECK:       [[SUB1A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD1:%[0-9]+]] = arith.addi [[MUL1]], [[SUB1A]] : index
// CHECK:       [[SUB2:%[0-9]+]] = arith.subi [[CONST2]], [[CONST1]] : index
// CHECK:       [[MUL2:%[0-9]+]] = arith.muli [[SUB2]], [[CONST1]] : index
// CHECK:       [[SUB2A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD2:%[0-9]+]] = arith.addi [[MUL2]], [[SUB2A]] : index
// CHECK:       [[SUB3:%[0-9]+]] = arith.subi [[CONST3]], [[CONST1]] : index
// CHECK:       [[MUL3:%[0-9]+]] = arith.muli [[SUB3]], [[CONST1]] : index
// CHECK:       [[SUB3A:%[0-9]+]] = arith.subi [[CONST1]], [[CONST1]] : index
// CHECK:       [[ADD3:%[0-9]+]] = arith.addi [[MUL3]], [[SUB3A]] : index
// CHECK:       [[ELSIZE:%[0-9]+]] = fir.box_elesize [[REBOX]] : (!fir.box<!fir.array<?x?x?xf32>>) -> index
// CHECK:       [[CONST2_0:%.+]] = arith.constant 2 : index
// CHECK:       [[DIMS2:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[CONST2_0]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:       [[DIV2:%[0-9]+]] = arith.divsi [[DIMS2]]#2, [[ELSIZE]] : index
// CHECK:       [[CONST1_1:%.+]] = arith.constant 1 : index
// CHECK:       [[DIMS1:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[CONST1_1]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:       [[DIV1:%[0-9]+]] = arith.divsi [[DIMS1]]#2, [[ELSIZE]] : index
// CHECK:       [[CONST0:%.+]] = arith.constant 0 : index
// CHECK:       [[DIMS0:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[CONST0]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:       [[DIV0:%[0-9]+]] = arith.divsi [[DIMS0]]#2, [[ELSIZE]] : index
// CHECK:       [[CONST0_2:%.+]] = arith.constant 0 : index
// CHECK:       [[REINTERPRET:%.+]] = memref.reinterpret_cast [[CONVERT]] to offset: [[[CONST0_2]]], sizes: [[[DIMS2]]#1, [[DIMS1]]#1, [[DIMS0]]#1], strides: [[[DIV2]], [[DIV1]], [[DIV0]]] : memref<?x?x?xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK:       memref.store [[CST]], [[REINTERPRET]][[[ADD3]], [[ADD2]], [[ADD1]]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
func.func @store_dynamic_3d(%arg0: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "x"}) {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 3.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "x"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?x?xf32>>
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
  %3 = fir.array_coor %2 %c2, %c2, %c3 : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index) -> !fir.ref<f32>
  fir.store %cst to %3 : !fir.ref<f32>
  return
}