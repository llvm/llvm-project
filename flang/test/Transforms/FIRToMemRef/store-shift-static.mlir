// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// subroutine store_shift_1d()
//   real, dimension(2:10) :: x
//   x(9) = 2
// end subroutine store_shift_1d
// CHECK-LABEL: func.func @store_shift_1d
// CHECK: [[CST:%.*]] = arith.constant 2.000000e+00 : f32
// CHECK: [[C2:%.*]] = arith.constant 2 : index
// CHECK: [[C9:%.*]] = arith.constant 9 : index
// CHECK: [[ALLOCA:%.*]] = memref.alloca() {bindc_name = "x", uniq_name = "x"} : memref<9xf32>
// CHECK: [[CONVERT1:%[0-9]+]] = fir.convert [[ALLOCA]] : (memref<9xf32>) -> !fir.ref<!fir.array<9xf32>>
// CHECK: [[SHAPE_SHIFT:%[0-9]+]] = fir.shape_shift [[C2]], [[C9]] : (index, index) -> !fir.shapeshift<1>
// CHECK: [[DECLARE:%[0-9]+]] = fir.declare [[CONVERT1]]([[SHAPE_SHIFT]]) {uniq_name = "x"} : (!fir.ref<!fir.array<9xf32>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<9xf32>>
// CHECK: [[CONVERT2:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<9xf32>>) -> memref<9xf32>
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[SUBI1:%[0-9]+]] = arith.subi [[C9]], [[C2]] : index
// CHECK: [[MULI1:%[0-9]+]] = arith.muli [[SUBI1]], [[C1]] : index
// CHECK: [[SUBI2:%[0-9]+]] = arith.subi [[C2]], [[C2]] : index
// CHECK: [[ADDI1:%[0-9]+]] = arith.addi [[MULI1]], [[SUBI2]] : index
// CHECK: memref.store [[CST]], [[CONVERT2]][[[ADDI1]]] : memref<9xf32>
func.func @store_shift_1d() {
  %cst = arith.constant 2.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %0 = fir.alloca !fir.array<9xf32> {bindc_name = "x", uniq_name = "x"}
  %1 = fir.shape_shift %c2, %c9 : (index, index) -> !fir.shapeshift<1>
  %2 = fir.declare %0(%1) {uniq_name = "x"} : (!fir.ref<!fir.array<9xf32>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<9xf32>>
  %3 = fir.array_coor %2(%1) %c9 : (!fir.ref<!fir.array<9xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  fir.store %cst to %3 : !fir.ref<f32>
  return
}

// subroutine store_shift_2d()
//   real, dimension(2:10,1:10) :: x
//   x(9,1) = 2
// end subroutine store_shift_2d
// CHECK-LABEL: func.func @store_shift_2d
// CHECK: [[CST:%.*]] = arith.constant 2.000000e+00 : f32
// CHECK: [[C2:%.*]] = arith.constant 2 : index
// CHECK: [[C9:%.*]] = arith.constant 9 : index
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[C10:%.*]] = arith.constant 10 : index
// CHECK: [[ALLOCA:%.*]] = memref.alloca() {bindc_name = "x", uniq_name = "_QFstore_shift_2dEx"} : memref<10x9xf32>
// CHECK: [[CONVERT1:%[0-9]+]] = fir.convert [[ALLOCA]] : (memref<10x9xf32>) -> !fir.ref<!fir.array<9x10xf32>>
// CHECK: [[SHAPE_SHIFT:%[0-9]+]] = fir.shape_shift [[C2]], [[C9]], [[C1]], [[C10]] : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK: [[DECLARE:%[0-9]+]] = fir.declare [[CONVERT1]]([[SHAPE_SHIFT]]) {uniq_name = "_QFstore_shift_2dEx"} : (!fir.ref<!fir.array<9x10xf32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<9x10xf32>>
// CHECK: [[CONVERT2:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<9x10xf32>>) -> memref<10x9xf32>
// CHECK: [[C1_0:%.*]] = arith.constant 1 : index
// CHECK: [[SUBI1:%[0-9]+]] = arith.subi [[C9]], [[C2]] : index
// CHECK: [[MULI1:%[0-9]+]] = arith.muli [[SUBI1]], [[C1_0]] : index
// CHECK: [[SUBI2:%[0-9]+]] = arith.subi [[C2]], [[C2]] : index
// CHECK: [[ADDI1:%[0-9]+]] = arith.addi [[MULI1]], [[SUBI2]] : index
// CHECK: [[SUBI3:%[0-9]+]] = arith.subi [[C1]], [[C1]] : index
// CHECK: [[MULI2:%[0-9]+]] = arith.muli [[SUBI3]], [[C1_0]] : index
// CHECK: [[SUBI4:%[0-9]+]] = arith.subi [[C1]], [[C1]] : index
// CHECK: [[ADDI2:%[0-9]+]] = arith.addi [[MULI2]], [[SUBI4]] : index
// CHECK: memref.store [[CST]], [[CONVERT2]][[[ADDI2]], [[ADDI1]]] : memref<10x9xf32>
func.func @store_shift_2d() {
  %cst = arith.constant 2.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<9x10xf32> {bindc_name = "x", uniq_name = "_QFstore_shift_2dEx"}
  %1 = fir.shape_shift %c2, %c9, %c1, %c10 : (index, index, index, index) -> !fir.shapeshift<2>
  %2 = fir.declare %0(%1) {uniq_name = "_QFstore_shift_2dEx"} : (!fir.ref<!fir.array<9x10xf32>>, !fir.shapeshift<2>) -> !fir.ref<!fir.array<9x10xf32>>
  %3 = fir.array_coor %2(%1) %c9, %c1 : (!fir.ref<!fir.array<9x10xf32>>, !fir.shapeshift<2>, index, index) -> !fir.ref<f32>
  fir.store %cst to %3 : !fir.ref<f32>
  return
}

// subroutine store_shift_3d()
//   real, dimension(2:10,1:10,3:8) :: x
//   x(9,1,9) = 2
// end subroutine store_shift_3d
// CHECK-LABEL: func.func @store_shift_3d
// CHECK: [[CST:%.*]] = arith.constant 2.000000e+00 : f32
// CHECK: [[C2:%.*]] = arith.constant 2 : index
// CHECK: [[C9:%.*]] = arith.constant 9 : index
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[C10:%.*]] = arith.constant 10 : index
// CHECK: [[C3:%.*]] = arith.constant 3 : index
// CHECK: [[C8:%.*]] = arith.constant 8 : index
// CHECK: [[ALLOCA:%.*]] = memref.alloca() {bindc_name = "x", uniq_name = "x"} : memref<8x10x9xf32>
// CHECK: [[CONVERT1:%[0-9]+]] = fir.convert [[ALLOCA]] : (memref<8x10x9xf32>) -> !fir.ref<!fir.array<9x10x8xf32>>
// CHECK: [[SHAPE_SHIFT:%[0-9]+]] = fir.shape_shift [[C2]], [[C9]], [[C1]], [[C10]], [[C3]], [[C8]] : (index, index, index, index, index, index) -> !fir.shapeshift<3>
// CHECK: [[DECLARE:%[0-9]+]] = fir.declare [[CONVERT1]]([[SHAPE_SHIFT]]) {uniq_name = "_QFstore_shift_3dEx"} : (!fir.ref<!fir.array<9x10x8xf32>>, !fir.shapeshift<3>) -> !fir.ref<!fir.array<9x10x8xf32>>
// CHECK: [[CONVERT2:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<9x10x8xf32>>) -> memref<8x10x9xf32>
// CHECK: [[C1_0:%.*]] = arith.constant 1 : index
// CHECK: [[SUBI1:%[0-9]+]] = arith.subi [[C9]], [[C2]] : index
// CHECK: [[MULI1:%[0-9]+]] = arith.muli [[SUBI1]], [[C1_0]] : index
// CHECK: [[SUBI2:%[0-9]+]] = arith.subi [[C2]], [[C2]] : index
// CHECK: [[ADDI1:%[0-9]+]] = arith.addi [[MULI1]], [[SUBI2]] : index
// CHECK: [[SUBI3:%[0-9]+]] = arith.subi [[C1]], [[C1]] : index
// CHECK: [[MULI2:%[0-9]+]] = arith.muli [[SUBI3]], [[C1_0]] : index
// CHECK: [[SUBI4:%[0-9]+]] = arith.subi [[C1]], [[C1]] : index
// CHECK: [[ADDI2:%[0-9]+]] = arith.addi [[MULI2]], [[SUBI4]] : index
// CHECK: [[SUBI5:%[0-9]+]] = arith.subi [[C9]], [[C3]] : index
// CHECK: [[MULI3:%[0-9]+]] = arith.muli [[SUBI5]], [[C1_0]] : index
// CHECK: [[SUBI6:%[0-9]+]] = arith.subi [[C3]], [[C3]] : index
// CHECK: [[ADDI3:%[0-9]+]] = arith.addi [[MULI3]], [[SUBI6]] : index
// CHECK: memref.store [[CST]], [[CONVERT2]][[[ADDI3]], [[ADDI2]], [[ADDI1]]] : memref<8x10x9xf32>
func.func @store_shift_3d() {
  %cst = arith.constant 2.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %0 = fir.alloca !fir.array<9x10x8xf32> {bindc_name = "x", uniq_name = "x"}
  %1 = fir.shape_shift %c2, %c9, %c1, %c10, %c3, %c8 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
  %2 = fir.declare %0(%1) {uniq_name = "_QFstore_shift_3dEx"} : (!fir.ref<!fir.array<9x10x8xf32>>, !fir.shapeshift<3>) -> !fir.ref<!fir.array<9x10x8xf32>>
  %3 = fir.array_coor %2(%1) %c9, %c1, %c9 : (!fir.ref<!fir.array<9x10x8xf32>>, !fir.shapeshift<3>, index, index, index) -> !fir.ref<f32>
  fir.store %cst to %3 : !fir.ref<f32>
  return
}