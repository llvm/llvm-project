// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// test fir.shift op
// derived from:
// subroutine load_shift_1d(x, y)
//   !generate fir.shift
//   real, dimension(2:) :: x
//   !fir.shape_shift
//   !real, dimension(2:10) :: x
//   real :: y
//   y = x(6)
// end subroutine load_shift_1d
// CHECK-LABEL: func.func @load_shift_1d
// CHECK: [[C6:%.*]] = arith.constant 6 : index
// CHECK: [[C2_I64:%.*]] = arith.constant 2 : i64
// CHECK: [[DUMMY_SCOPE:%[0-9]+]] = fir.dummy_scope : !fir.dscope
// CHECK: [[INDEX_CAST:%[0-9]+]] = arith.index_cast [[C2_I64]] : i64 to index
// CHECK: [[SHIFT:%[0-9]+]] = fir.shift [[INDEX_CAST]] : (index) -> !fir.shift<1>
// CHECK: [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHIFT]]) dummy_scope [[DUMMY_SCOPE]] {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK: [[REBOX:%[0-9]+]] = fir.rebox [[DECLARE]]([[SHIFT]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
// CHECK: [[BOX_ADDR:%[0-9]+]] = fir.box_addr [[REBOX]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK: [[CONVERT:%[0-9]+]] = fir.convert [[BOX_ADDR]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[SUBI:%[0-9]+]] = arith.subi [[C6]], [[C1]] : index
// CHECK: [[MULI:%[0-9]+]] = arith.muli [[SUBI]], [[C1]] : index
// CHECK: [[SUBI2:%[0-9]+]] = arith.subi [[C1]], [[C1]] : index
// CHECK: [[ADDI:%[0-9]+]] = arith.addi [[MULI]], [[SUBI2]] : index
// CHECK: [[BOX_ELESIZE:%[0-9]+]] = fir.box_elesize [[REBOX]] : (!fir.box<!fir.array<?xf32>>) -> index
// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[BOX_DIMS:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK: [[DIVSI:%[0-9]+]] = arith.divsi [[BOX_DIMS]]#2, [[BOX_ELESIZE]] : index
// CHECK: [[C0_0:%.*]] = arith.constant 0 : index
// CHECK: [[REINTERPRET_CAST:%.*]] = memref.reinterpret_cast [[CONVERT]] to offset: [[[C0_0]]], sizes: [[[BOX_DIMS]]#1], strides: [[[DIVSI]]] : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK: [[LOAD:%[0-9]+]] = memref.load [[REINTERPRET_CAST]][[[ADDI]]] : memref<?xf32, strided<[?], offset: ?>>
func.func @load_shift_1d(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c6 = arith.constant 6 : index
  %c2_i64 = arith.constant 2 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = arith.index_cast %c2_i64 : i64 to index
  %2 = fir.shift %1 : (index) -> !fir.shift<1>
  %3 = fir.declare %arg0(%2) dummy_scope %0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %4 = fir.rebox %3(%2) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
  %5 = fir.array_coor %4 %c6 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %6 = fir.load %5 : !fir.ref<f32>
  return
}

// test fir.shift op for 2D array
// derived from:
// subroutine load_shift_2d(x, y)
//   real, dimension(2:,3:) :: x
//   !real, dimension(2:10,3:10) :: x
//   real :: y
//   y = x(6,7)
// end subroutine load_shift_2d
// CHECK-LABEL: func.func @load_shift_2d
// CHECK: [[C7:%.*]] = arith.constant 7 : index
// CHECK: [[C6:%.*]] = arith.constant 6 : index
// CHECK: [[C3_I64:%.*]] = arith.constant 3 : i64
// CHECK: [[C2_I64:%.*]] = arith.constant 2 : i64
// CHECK: [[DUMMY_SCOPE:%[0-9]+]] = fir.dummy_scope : !fir.dscope
// CHECK: [[INDEX_CAST1:%[0-9]+]] = arith.index_cast [[C2_I64]] : i64 to index
// CHECK: [[INDEX_CAST2:%[0-9]+]] = arith.index_cast [[C3_I64]] : i64 to index
// CHECK: [[SHIFT:%[0-9]+]] = fir.shift [[INDEX_CAST1]], [[INDEX_CAST2]] : (index, index) -> !fir.shift<2>
// CHECK: [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHIFT]]) dummy_scope [[DUMMY_SCOPE]] {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
// CHECK: [[REBOX:%[0-9]+]] = fir.rebox [[DECLARE]]([[SHIFT]]) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK: [[BOX_ADDR:%[0-9]+]] = fir.box_addr [[REBOX]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK: [[CONVERT:%[0-9]+]] = fir.convert [[BOX_ADDR]] : (!fir.ref<!fir.array<?x?xf32>>) -> memref<?x?xf32>
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[SUBI1:%[0-9]+]] = arith.subi [[C6]], [[INDEX_CAST1]] : index
// CHECK: [[MULI1:%[0-9]+]] = arith.muli [[SUBI1]], [[C1]] : index
// CHECK: [[SUBI2:%[0-9]+]] = arith.subi [[INDEX_CAST1]], [[INDEX_CAST1]] : index
// CHECK: [[ADDI1:%[0-9]+]] = arith.addi [[MULI1]], [[SUBI2]] : index
// CHECK: [[SUBI3:%[0-9]+]] = arith.subi [[C7]], [[INDEX_CAST2]] : index
// CHECK: [[MULI2:%[0-9]+]] = arith.muli [[SUBI3]], [[C1]] : index
// CHECK: [[SUBI4:%[0-9]+]] = arith.subi [[INDEX_CAST2]], [[INDEX_CAST2]] : index
// CHECK: [[ADDI2:%[0-9]+]] = arith.addi [[MULI2]], [[SUBI4]] : index
// CHECK: [[BOX_ELESIZE:%[0-9]+]] = fir.box_elesize [[REBOX]] : (!fir.box<!fir.array<?x?xf32>>) -> index
// CHECK: [[C1_0:%.*]] = arith.constant 1 : index
// CHECK: [[BOX_DIMS1:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[C1_0]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK: [[DIVSI1:%[0-9]+]] = arith.divsi [[BOX_DIMS1]]#2, [[BOX_ELESIZE]] : index
// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[BOX_DIMS2:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[C0]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK: [[DIVSI2:%[0-9]+]] = arith.divsi [[BOX_DIMS2]]#2, [[BOX_ELESIZE]] : index
// CHECK: [[C0_1:%.*]] = arith.constant 0 : index
// CHECK: [[REINTERPRET_CAST:%.*]] = memref.reinterpret_cast [[CONVERT]] to offset: [[[C0_1]]], sizes: [[[BOX_DIMS1]]#1, [[BOX_DIMS2]]#1], strides: [[[DIVSI1]], [[DIVSI2]]] : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK: [[LOAD:%[0-9]+]] = memref.load [[REINTERPRET_CAST]][[[ADDI2]], [[ADDI1]]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
func.func @load_shift_2d(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %c7 = arith.constant 7 : index
  %c6 = arith.constant 6 : index
  %c3_i64 = arith.constant 3 : i64
  %c2_i64 = arith.constant 2 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = arith.index_cast %c2_i64 : i64 to index
  %2 = arith.index_cast %c3_i64 : i64 to index
  %3 = fir.shift %1, %2 : (index, index) -> !fir.shift<2>
  %4 = fir.declare %arg0(%3) dummy_scope %0 {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %5 = fir.rebox %4(%3) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
  %7 = fir.array_coor %5(%3) %c6, %c7 : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, index, index) -> !fir.ref<f32>
  %8 = fir.load %7 : !fir.ref<f32>
  return
}

// test fir.shift op for 3D array
// derived from:
// subroutine load_shift_3d(x, y)
//   real, dimension(2:,1:,3:) :: x
//   !real, dimension(2:10,1:10,3:10) :: x
//   real :: y
//   y = x(9,10,9)
// end subroutine load_shift_3d
// CHECK-LABEL: func.func @load_shift_3d
// CHECK: [[C10:%.*]] = arith.constant 10 : index
// CHECK: [[C9:%.*]] = arith.constant 9 : index
// CHECK: [[C3_I64:%.*]] = arith.constant 3 : i64
// CHECK: [[C1_I64:%.*]] = arith.constant 1 : i64
// CHECK: [[C2_I64:%.*]] = arith.constant 2 : i64
// CHECK: [[DUMMY_SCOPE:%[0-9]+]] = fir.dummy_scope : !fir.dscope
// CHECK: [[INDEX_CAST1:%[0-9]+]] = arith.index_cast [[C2_I64]] : i64 to index
// CHECK: [[INDEX_CAST2:%[0-9]+]] = arith.index_cast [[C1_I64]] : i64 to index
// CHECK: [[INDEX_CAST3:%[0-9]+]] = arith.index_cast [[C3_I64]] : i64 to index
// CHECK: [[SHIFT:%[0-9]+]] = fir.shift [[INDEX_CAST1]], [[INDEX_CAST2]], [[INDEX_CAST3]] : (index, index, index) -> !fir.shift<3>
// CHECK: [[DECLARE:%[0-9]+]] = fir.declare %arg0([[SHIFT]]) dummy_scope [[DUMMY_SCOPE]] {uniq_name = "x"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, !fir.dscope) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK: [[REBOX:%[0-9]+]] = fir.rebox [[DECLARE]]([[SHIFT]]) : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK: [[BOX_ADDR:%[0-9]+]] = fir.box_addr [[REBOX]] : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.ref<!fir.array<?x?x?xf32>>
// CHECK: [[CONVERT:%[0-9]+]] = fir.convert [[BOX_ADDR]] : (!fir.ref<!fir.array<?x?x?xf32>>) -> memref<?x?x?xf32>
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[SUBI1:%[0-9]+]] = arith.subi [[C9]], [[INDEX_CAST1]] : index
// CHECK: [[MULI1:%[0-9]+]] = arith.muli [[SUBI1]], [[C1]] : index
// CHECK: [[SUBI2:%[0-9]+]] = arith.subi [[INDEX_CAST1]], [[INDEX_CAST1]] : index
// CHECK: [[ADDI1:%[0-9]+]] = arith.addi [[MULI1]], [[SUBI2]] : index
// CHECK: [[SUBI3:%[0-9]+]] = arith.subi [[C10]], [[INDEX_CAST2]] : index
// CHECK: [[MULI2:%[0-9]+]] = arith.muli [[SUBI3]], [[C1]] : index
// CHECK: [[SUBI4:%[0-9]+]] = arith.subi [[INDEX_CAST2]], [[INDEX_CAST2]] : index
// CHECK: [[ADDI2:%[0-9]+]] = arith.addi [[MULI2]], [[SUBI4]] : index
// CHECK: [[SUBI5:%[0-9]+]] = arith.subi [[C9]], [[INDEX_CAST3]] : index
// CHECK: [[MULI3:%[0-9]+]] = arith.muli [[SUBI5]], [[C1]] : index
// CHECK: [[SUBI6:%[0-9]+]] = arith.subi [[INDEX_CAST3]], [[INDEX_CAST3]] : index
// CHECK: [[ADDI3:%[0-9]+]] = arith.addi [[MULI3]], [[SUBI6]] : index
// CHECK: [[BOX_ELESIZE:%[0-9]+]] = fir.box_elesize [[REBOX]] : (!fir.box<!fir.array<?x?x?xf32>>) -> index
// CHECK: [[C2:%.*]] = arith.constant 2 : index
// CHECK: [[BOX_DIMS1:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[C2]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK: [[DIVSI1:%[0-9]+]] = arith.divsi [[BOX_DIMS1]]#2, [[BOX_ELESIZE]] : index
// CHECK: [[C1_0:%.*]] = arith.constant 1 : index
// CHECK: [[BOX_DIMS2:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[C1_0]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK: [[DIVSI2:%[0-9]+]] = arith.divsi [[BOX_DIMS2]]#2, [[BOX_ELESIZE]] : index
// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[BOX_DIMS3:%[0-9]+]]:3 = fir.box_dims [[REBOX]], [[C0]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK: [[DIVSI3:%[0-9]+]] = arith.divsi [[BOX_DIMS3]]#2, [[BOX_ELESIZE]] : index
// CHECK: [[C0_1:%.*]] = arith.constant 0 : index
// CHECK: [[REINTERPRET_CAST:%.*]] = memref.reinterpret_cast [[CONVERT]] to offset: [[[C0_1]]], sizes: [[[BOX_DIMS1]]#1, [[BOX_DIMS2]]#1, [[BOX_DIMS3]]#1], strides: [[[DIVSI1]], [[DIVSI2]], [[DIVSI3]]] : memref<?x?x?xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
// CHECK: [[LOAD:%[0-9]+]] = memref.load [[REINTERPRET_CAST]][[[ADDI3]], [[ADDI2]], [[ADDI1]]] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
func.func @load_shift_3d(%arg0: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "x"}) {
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %c3_i64 = arith.constant 3 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = arith.index_cast %c2_i64 : i64 to index
  %2 = arith.index_cast %c1_i64 : i64 to index
  %3 = arith.index_cast %c3_i64 : i64 to index
  %4 = fir.shift %1, %2, %3 : (index, index, index) -> !fir.shift<3>
  %5 = fir.declare %arg0(%4) dummy_scope %0 {uniq_name = "x"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, !fir.dscope) -> !fir.box<!fir.array<?x?x?xf32>>
  %6 = fir.rebox %5(%4) : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
  %8 = fir.array_coor %6(%4) %c9, %c10, %c9 : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, index, index, index) -> !fir.ref<f32>
  %9 = fir.load %8 : !fir.ref<f32>
  return
}
