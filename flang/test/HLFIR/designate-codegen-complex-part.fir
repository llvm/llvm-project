// Test code generation to FIR of hlfir.designate operations
// with complex parts.
// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test_set_scalar(%arg0: !fir.ref<complex<f32>>, %arg1: !fir.ref<f32>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.load %1#0 : !fir.ref<f32>
  %3 = hlfir.designate %0#0  imag : (!fir.ref<complex<f32>>) -> !fir.ref<f32>
  hlfir.assign %2 to %3 : f32, !fir.ref<f32>
  return
}
// CHECK-LABEL: func.func @test_set_scalar(
// CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<complex<f32>>, %[[VAL_1:.*]]: !fir.ref<f32>) {
// CHECK: %[[VAL_2:.*]] = fir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.ref<complex<f32>>) -> !fir.ref<complex<f32>>
// CHECK: %[[VAL_3:.*]] = fir.declare %[[VAL_1]] {uniq_name = "b"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<f32>
// CHECK: %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_5]] : (!fir.ref<complex<f32>>, index) -> !fir.ref<f32>
// CHECK: fir.store %[[VAL_4]] to %[[VAL_6]] : !fir.ref<f32>

func.func @test_scalar_at_index(%arg0: !fir.box<!fir.array<?xcomplex<f32>>>, %arg1: !fir.ref<i32>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xcomplex<f32>>>) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.load %1#0 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> i64
  %4 = hlfir.designate %0#0 (%3) real : (!fir.box<!fir.array<?xcomplex<f32>>>, i64) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: func.func @test_scalar_at_index(
// CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?xcomplex<f32>>>, %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK: %[[VAL_2:.*]] = fir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<!fir.array<?xcomplex<f32>>>
// CHECK: %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<!fir.array<?xcomplex<f32>>>
// CHECK: %[[VAL_4:.*]] = fir.declare %[[VAL_1]] {uniq_name = "b"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
// CHECK: %[[VAL_7:.*]] = fir.array_coor %[[VAL_3]] %[[VAL_6]] : (!fir.box<!fir.array<?xcomplex<f32>>>, i64) -> !fir.ref<complex<f32>>
// CHECK: %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_8]] : (!fir.ref<complex<f32>>, index) -> !fir.ref<f32>

func.func @test_complete_slice(%arg0: !fir.box<!fir.array<?xcomplex<f32>>>) {
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xcomplex<f32>>>) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
  %1:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xcomplex<f32>>>, index) -> (index, index, index)
  %2 = fir.shape %1#1 : (index) -> !fir.shape<1>
  %3 = hlfir.designate %0#0  imag shape %2 : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}
// CHECK-LABEL: func.func @test_complete_slice(
// CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?xcomplex<f32>>>) {
// CHECK: %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_2:.*]] = fir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<!fir.array<?xcomplex<f32>>>
// CHECK: %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<!fir.array<?xcomplex<f32>>>
// CHECK: %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_1]] : (!fir.box<!fir.array<?xcomplex<f32>>>, index) -> (index, index, index)
// CHECK: %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
// CHECK: %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_8]] : (!fir.box<!fir.array<?xcomplex<f32>>>, index) -> (index, index, index)
// CHECK: %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_11:.*]] = fir.slice %[[VAL_7]], %[[VAL_9]]#1, %[[VAL_6]] path %[[VAL_10]] : (index, index, index, index) -> !fir.slice<1>
// CHECK: %[[VAL_12:.*]] = fir.rebox %[[VAL_3]] [%[[VAL_11]]] : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>

func.func @test_slice_steps(%arg0: !fir.box<!fir.array<?xcomplex<f32>>>) {
  %c3 = arith.constant 3 : index
  %c12 = arith.constant 12 : index
  %c4 = arith.constant 4 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xcomplex<f32>>>) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
  %1 = fir.shape %c3 : (index) -> !fir.shape<1>
  %2 = hlfir.designate %0#0 (%c4:%c12:%c3) real shape %1 : (!fir.box<!fir.array<?xcomplex<f32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<3xf32>>
  return
}
// CHECK-LABEL: func.func @test_slice_steps(
// CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?xcomplex<f32>>>) {
// CHECK: %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK: %[[VAL_2:.*]] = arith.constant 12 : index
// CHECK: %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK: %[[VAL_4:.*]] = fir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<!fir.array<?xcomplex<f32>>>
// CHECK: %[[VAL_5:.*]] = fir.rebox %[[VAL_4]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<!fir.array<?xcomplex<f32>>>
// CHECK: %[[VAL_6:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK: %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_8:.*]] = fir.slice %[[VAL_3]], %[[VAL_2]], %[[VAL_1]] path %[[VAL_7]] : (index, index, index, index) -> !fir.slice<1>
// CHECK: %[[VAL_9:.*]] = fir.rebox %[[VAL_5]] [%[[VAL_8]]] : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.slice<1>) -> !fir.box<!fir.array<3xf32>>
