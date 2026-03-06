// Test hlfir.get_extent code generation to FIR
// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test1(%arg0: i1, %arg1: i32, %arg2: i64, %arg3: index) -> (index, index, index, index) {
  %0 = fir.shape %arg0, %arg1, %arg2, %arg3 : (i1, i32, i64, index) -> !fir.shape<4>
  %1 = hlfir.get_extent %0 {dim = 0 : index} : (!fir.shape<4>) -> index
  %2 = hlfir.get_extent %0 {dim = 1 : index} : (!fir.shape<4>) -> index
  %3 = hlfir.get_extent %0 {dim = 2 : index} : (!fir.shape<4>) -> index
  %4 = hlfir.get_extent %0 {dim = 3 : index} : (!fir.shape<4>) -> index
  return %1, %2, %3, %4 : index, index, index, index
}
// CHECK-LABEL:   func.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                     %[[VAL_1:.*]]: i32,
// CHECK-SAME:                     %[[VAL_2:.*]]: i64,
// CHECK-SAME:                     %[[VAL_3:.*]]: index) -> (index, index, index, index) {
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : (i1, i32, i64, index) -> !fir.shape<4>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_0]] : (i1) -> index
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
// CHECK:           return %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_3]] : index, index, index, index
// CHECK:         }
