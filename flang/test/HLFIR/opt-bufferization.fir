// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// simplified example
func.func @simple(%arg: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @simple(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           fir.do_loop %[[VAL_6:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_9]] to %[[VAL_10]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// check we support reads that don't alias the transformed array
func.func @read_no_alias(%arg: !fir.ref<!fir.array<42xi32>>, %arg1: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %other:2 = hlfir.declare %arg1(%shape) {uniq_name = "other"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %other_ref = hlfir.designate %other#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %other_val = fir.load %other_ref : !fir.ref<i32>
    %sub = arith.subi %val, %other_val : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @read_no_alias(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_4]]) {uniq_name = "other"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] unordered {
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_12]] to %[[VAL_13]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }


// check we don't transform when there is another use of the elemental expr
func.func @two_uses(%arg: !fir.ref<!fir.array<42xi32>>) -> i32 {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  %bad = hlfir.apply %elemental, %c42 : (!hlfir.expr<42xi32>, index) -> i32
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return %bad : i32
}
// CHECK-LABEL:   func.func @two_uses(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:             hlfir.yield_element %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_10:.*]] to %[[VAL_4]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           %[[VAL_11:.*]] = hlfir.apply %[[VAL_10]], %[[VAL_1]] : (!hlfir.expr<42xi32>, index) -> i32
// CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<42xi32>
// CHECK:           return %[[VAL_11]] : i32
// CHECK:         }

// two dimensional array
func.func @two_dimensional(%arg: !fir.ref<!fir.array<42x42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42, %c42 : (index, index) -> !fir.shape<2>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<42x42xi32> {
  ^bb0(%i: index, %j: index):
    %ref = hlfir.designate %array#0 (%i, %j) : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42x42xi32>, !fir.ref<!fir.array<42x42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42x42xi32>
  return
}
// CHECK-LABEL:   func.func @two_dimensional(
// CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.array<42x42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
// CHECK:           fir.do_loop %[[VAL_6:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:             fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]], %[[VAL_6]])  : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:               %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_11:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]], %[[VAL_6]])  : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_10]] to %[[VAL_11]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// don't transform when elements are accessessed out of order (e.g. transposed)
func.func @transposed(%arg: !fir.ref<!fir.array<42x42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42, %c42 : (index, index) -> !fir.shape<2>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<42x42xi32> {
  ^bb0(%i: index, %j: index):
    %ref = hlfir.designate %array#0 (%j, %i) : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42x42xi32>, !fir.ref<!fir.array<42x42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42x42xi32>
  return
}
// CHECK-LABEL:   func.func @transposed(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.array<42x42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<2>) -> !hlfir.expr<42x42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_7]], %[[VAL_6]])  : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:             hlfir.yield_element %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_11:.*]] to %[[VAL_4]]#0 : !hlfir.expr<42x42xi32>, !fir.ref<!fir.array<42x42xi32>>
// CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<42x42xi32>
// CHECK:           return
// CHECK:         }

// don't transform when there's an operation with unknown effects
func.func @unknown(%arg: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    %res = fir.call @impure(%sub) : (i32) -> i32
    hlfir.yield_element %res : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @unknown(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_10:.*]] = fir.call @impure(%[[VAL_9]]) : (i32) -> i32
// CHECK:             hlfir.yield_element %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_11:.*]] to %[[VAL_4]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<42xi32>
// CHECK:           return
// CHECK:         }

// don't transform when there's an operation with write effects
func.func @write(%arg: !fir.ref<!fir.array<42xi32>>, %arg1: !fir.ref<!fir.array<42xi32>>) {
  %alloc = fir.alloca i32
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %array2:2 = hlfir.declare %arg1(%shape) {uniq_name = "array2"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    hlfir.assign %array2#0 to %array#0 : !fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    fir.store %sub to %alloc : !fir.ref<i32>
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @write(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>,
// CHECK-SAME:                     %[[ARG_1:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5B:.*]]:2 = hlfir.declare %[[ARG_1]](%[[VAL_4]]) {uniq_name = "array2"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             hlfir.assign
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_1]] : i32
// CHECK:             fir.store %[[VAL_10]] to %[[VAL_3]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_11:.*]] to %[[VAL_5]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<42xi32>
// CHECK:           return
// CHECK:         }

// don't transform when there is an aliasing read
func.func @readAlias(%arg: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %arrayDup:2 = hlfir.declare %arg(%shape) {uniq_name = "arrayDup"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %refDup = hlfir.designate %arrayDup#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %valDup = fir.load %refDup : !fir.ref<i32>
    %sub = arith.subi %val, %valDup : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @readAlias(
// CHECK-SAME:                         %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "arrayDup"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_2]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:             hlfir.yield_element %[[VAL_11]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_12:.*]] to %[[VAL_3]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<42xi32>
// CHECK:           return
// CHECK:         }

// don't transform when moving the elemental to the assignment could change the results of a read
func.func @write_conflict(%arg: !fir.ref<!fir.array<42xi32>>) -> index {
  %alloc = fir.alloca index
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    // write in elemental:
    hlfir.assign %i to %alloc : index, !fir.ref<index>
    hlfir.yield_element %sub : i32
  }
  // conflicting read:
  %conflict = fir.load %alloc : !fir.ref<index>
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return %conflict : index
}
// CHECK-LABEL:   func.func @write_conflict(
// CHECK-SAME:                              %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) -> index {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_3:.*]] = fir.alloca index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_1]] : i32
// CHECK:             hlfir.assign %[[VAL_7]] to %[[VAL_3]] : index, !fir.ref<index>
// CHECK:             hlfir.yield_element %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
// CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_5]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_6]] : !hlfir.expr<42xi32>
// CHECK:           return %[[VAL_11]] : index
// CHECK:         }

// don't transform when moving the elemental to the assignment could change the results of a read #2
func.func @read_conflict(%arg: !fir.ref<!fir.array<42xi32>>) {
  %alloc = fir.alloca i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %alloc : i32, !fir.ref<i32>
  %c42 = arith.constant 42 : index
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    // conflicting read:
    %read = fir.load %alloc : !fir.ref<i32>
    %sub = arith.subi %val, %read : i32
    hlfir.yield_element %sub : i32
  }
  // conflicting write:
  hlfir.assign %c0_i32 to %alloc : i32, !fir.ref<i32>
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @read_conflict(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = fir.alloca i32
// CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_4]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
// CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:             hlfir.yield_element %[[VAL_12]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_3]] to %[[VAL_4]] : i32, !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_7]] to %[[VAL_6]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<42xi32>
// CHECK:           return
// CHECK:         }

fir.global @_QMmEblock : !fir.array<9x9x9xi32> {
  %0 = fir.undefined !fir.array<9x9x9xi32>
  fir.has_value %0 : !fir.array<9x9x9xi32>
}
fir.global @_QMmECr constant : i32 {
  %c9_i32 = arith.constant 9 : i32
  fir.has_value %c9_i32 : i32
}

// does it work for the intended case?
func.func @_QMmPrepro(%arg0: !fir.ref<i32> {fir.bindc_name = "imin"}, %arg1: !fir.ref<i32> {fir.bindc_name = "imax"}, %arg2: !fir.ref<i32> {fir.bindc_name = "row"}) {
  %c10_i32 = arith.constant 10 : i32
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %0 = fir.address_of(@_QMmEblock) : !fir.ref<!fir.array<9x9x9xi32>>
  %1 = fir.shape %c9, %c9, %c9 : (index, index, index) -> !fir.shape<3>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QMmEblock"} : (!fir.ref<!fir.array<9x9x9xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<9x9x9xi32>>, !fir.ref<!fir.array<9x9x9xi32>>)
  %3 = fir.address_of(@_QMmECr) : !fir.ref<i32>
  %4:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMmECr"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFreproEi1"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QMmFreproEi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7:2 = hlfir.declare %arg1 {uniq_name = "_QMmFreproEimax"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8:2 = hlfir.declare %arg0 {uniq_name = "_QMmFreproEimin"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9:2 = hlfir.declare %arg2 {uniq_name = "_QMmFreproErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.load %8#0 : !fir.ref<i32>
  %11 = fir.convert %10 : (i32) -> index
  %12 = fir.load %7#0 : !fir.ref<i32>
  %13 = fir.convert %12 : (i32) -> index
  %14 = fir.convert %11 : (index) -> i32
  %15:2 = fir.do_loop %arg3 = %11 to %13 step %c1 iter_args(%arg4 = %14) -> (index, i32) {
    fir.store %arg4 to %6#1 : !fir.ref<i32>
    %16 = fir.load %9#0 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> i64
    %18 = fir.load %6#0 : !fir.ref<i32>
    %19 = fir.convert %18 : (i32) -> i64
    %20 = fir.shape %c8 : (index) -> !fir.shape<1>
    %21 = hlfir.designate %2#0 (%17, %c2:%c9:%c1, %19)  shape %20 : (!fir.ref<!fir.array<9x9x9xi32>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<8xi32>>
    %22 = hlfir.elemental %20 unordered : (!fir.shape<1>) -> !hlfir.expr<8xi32> {
    ^bb0(%arg5: index):
      %27 = hlfir.designate %21 (%arg5)  : (!fir.box<!fir.array<8xi32>>, index) -> !fir.ref<i32>
      %28 = fir.load %27 : !fir.ref<i32>
      %29 = arith.subi %28, %c10_i32 : i32
      hlfir.yield_element %29 : i32
    }
    hlfir.assign %22 to %21 : !hlfir.expr<8xi32>, !fir.box<!fir.array<8xi32>>
    hlfir.destroy %22 : !hlfir.expr<8xi32>
    %23 = arith.addi %arg3, %c1 : index
    %24 = fir.convert %c1 : (index) -> i32
    %25 = fir.load %6#1 : !fir.ref<i32>
    %26 = arith.addi %25, %24 : i32
    fir.result %23, %26 : index, i32
  }
  fir.store %15#1 to %6#1 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QMmPrepro(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "imin"},
// CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "imax"},
// CHECK-SAME:                          %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "row"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QMmEblock) : !fir.ref<!fir.array<9x9x9xi32>>
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_7]], %[[VAL_7]], %[[VAL_7]] : (index, index, index) -> !fir.shape<3>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) {uniq_name = "_QMmEblock"} : (!fir.ref<!fir.array<9x9x9xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<9x9x9xi32>>, !fir.ref<!fir.array<9x9x9xi32>>)
// CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QMmECr) : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMmECr"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_13:.*]] = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFreproEi1"}
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QMmFreproEi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QMmFreproEimax"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QMmFreproEimin"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QMmFreproErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_16]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
// CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
// CHECK:           %[[VAL_23:.*]]:2 = fir.do_loop %[[VAL_24:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_6]] iter_args(%[[VAL_25:.*]] = %[[VAL_22]]) -> (index, i32) {
// CHECK:             fir.store %[[VAL_25]] to %[[VAL_14]]#1 : !fir.ref<i32>
// CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
// CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
// CHECK:             %[[VAL_30:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_31:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_27]], %[[VAL_5]]:%[[VAL_7]]:%[[VAL_6]], %[[VAL_29]])  shape %[[VAL_30]] : (!fir.ref<!fir.array<9x9x9xi32>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<8xi32>>
// CHECK:             fir.do_loop %[[VAL_32:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_6]] unordered {
// CHECK:               %[[VAL_33:.*]] = hlfir.designate %[[VAL_31]] (%[[VAL_32]])  : (!fir.box<!fir.array<8xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_34:.*]] = fir.load %[[VAL_33]] : !fir.ref<i32>
// CHECK:               %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_36:.*]] = hlfir.designate %[[VAL_31]] (%[[VAL_32]])  : (!fir.box<!fir.array<8xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_35]] to %[[VAL_36]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_24]], %[[VAL_6]] : index
// CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_6]] : (index) -> i32
// CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_14]]#1 : !fir.ref<i32>
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_38]] : i32
// CHECK:             fir.result %[[VAL_37]], %[[VAL_40]] : index, i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_41:.*]]#1 to %[[VAL_14]]#1 : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// support for z = x + y
func.func @other_reads(%z_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "z"}, %x_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %y_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
  %c0 = arith.constant 0 : index
  %box_dims:3 = fir.box_dims %z_arg, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  // ignore lower bound etc
  %shape = fir.shape %box_dims#1 : (index) -> !fir.shape<1>

  %z:2 = hlfir.declare %z_arg(%shape) {uniq_name = "z"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %x:2 = hlfir.declare %x_arg(%shape) {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %y:2 = hlfir.declare %y_arg(%shape) {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)

  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %x_ref = hlfir.designate %x#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %x_val = fir.load %x_ref : !fir.ref<f32>
    %y_ref = hlfir.designate %y#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %y_val = fir.load %y_ref : !fir.ref<f32>
    %add = arith.addf %x_val, %y_val : f32
    hlfir.yield_element %add : f32
  }
  hlfir.assign %elemental to %z#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %elemental : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @other_reads(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "z"},
// CHECK-SAME:                      %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                      %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_6]]) {uniq_name = "z"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_6]]) {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_6]]) {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           fir.do_loop %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_5]]#1 step %[[VAL_3]] unordered {
// CHECK:             %[[VAL_11:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<f32>
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_12]], %[[VAL_14]] : f32
// CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_15]] to %[[VAL_16]] : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// support for z = x + y, when z is declared after the elemental
func.func @other_reads_late_decl(%z_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "z"}, %x_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %y_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
  %c0 = arith.constant 0 : index
  %box_dims:3 = fir.box_dims %z_arg, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  // ignore lower bound etc
  %shape = fir.shape %box_dims#1 : (index) -> !fir.shape<1>

  %x:2 = hlfir.declare %x_arg(%shape) {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %y:2 = hlfir.declare %y_arg(%shape) {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)

  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %x_ref = hlfir.designate %x#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %x_val = fir.load %x_ref : !fir.ref<f32>
    %y_ref = hlfir.designate %y#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %y_val = fir.load %y_ref : !fir.ref<f32>
    %add = arith.addf %x_val, %y_val : f32
    hlfir.yield_element %add : f32
  }
  %z:2 = hlfir.declare %z_arg(%shape) {uniq_name = "z"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  hlfir.assign %elemental to %z#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %elemental : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @other_reads_late_decl(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "z"},
// CHECK-SAME:                      %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                      %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
// CHECK-DAG:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_6]]) {uniq_name = "z"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK-DAG:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_6]]) {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK-DAG:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_6]]) {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           fir.do_loop %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_5]]#1 step %[[VAL_3]] unordered {
// CHECK:             %[[VAL_11:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<f32>
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_12]], %[[VAL_14]] : f32
// CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_15]] to %[[VAL_16]] : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// support for z = x + y, when z is declared after the elemental and the shape values are different
func.func @other_reads_odd_shape(%z_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "z"}, %x_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %y_arg: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
  %c0 = arith.constant 0 : index
  %x_box_dims:3 = fir.box_dims %x_arg, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  // ignore lower bound etc
  %xy_shape = fir.shape %x_box_dims#1 : (index) -> !fir.shape<1>

  %x:2 = hlfir.declare %x_arg(%xy_shape) {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %y:2 = hlfir.declare %y_arg(%xy_shape) {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)

  %elemental = hlfir.elemental %xy_shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%i: index):
    %x_ref = hlfir.designate %x#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %x_val = fir.load %x_ref : !fir.ref<f32>
    %y_ref = hlfir.designate %y#0 (%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %y_val = fir.load %y_ref : !fir.ref<f32>
    %add = arith.addf %x_val, %y_val : f32
    hlfir.yield_element %add : f32
  }

  %z_box_dims:3 = fir.box_dims %z_arg, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  // ignore lower bound etc
  %z_shape = fir.shape %z_box_dims#1 : (index) -> !fir.shape<1>
  %z:2 = hlfir.declare %z_arg(%z_shape) {uniq_name = "z"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  // assume the length of z is 10 longer than the length of x
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %slice_extent = arith.addi %c10, %x_box_dims#1 : index
  %slice_shape = fir.shape %slice_extent : (index) -> !fir.shape<1>
  %z_slice = hlfir.designate %z#0 (%c10:%slice_extent:%c1) shape %slice_shape : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>

  hlfir.assign %elemental to %z_slice : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %elemental : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL: func.func @other_reads_odd_shape(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "z"},
// CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                                   %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
// CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:         %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
// CHECK:         %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:         %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_7]]) {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:         %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_10]]#1 : (index) -> !fir.shape<1>
// CHECK:         %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_11]]) {uniq_name = "z"} : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:         %[[VAL_13:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_4]] : index
// CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:         %[[VAL_15:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_4]]:%[[VAL_13]]:%[[VAL_3]])  shape %[[VAL_14]] : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CHECK:         fir.do_loop %[[VAL_16:.*]] = %[[VAL_3]] to %[[VAL_6]]#1 step %[[VAL_3]] unordered {
// CHECK:           %[[VAL_17:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_16]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<f32>
// CHECK:           %[[VAL_19:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_16]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<f32>
// CHECK:           %[[VAL_21:.*]] = arith.addf %[[VAL_18]], %[[VAL_20]] : f32
// CHECK:           %[[VAL_22:.*]] = hlfir.designate %[[VAL_15]] (%[[VAL_16]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_22]] : f32, !fir.ref<f32>
// CHECK:         }
// CHECK:         return
// CHECK:       }

// full test from intended code samplemodule attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
func.func @_QPddx(%arg0: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "array"}) -> !fir.array<?x?xf64> {
  %c-1 = arith.constant -1 : index
  %c-2 = arith.constant -2 : index
  %c1_i32 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %c2_i32 = arith.constant 2 : i32
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFddxEarray"} : (!fir.box<!fir.array<?x?xf64>>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>)
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFddxEi"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFddxEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFddxEj"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFddxEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
  %6 = fir.convert %5#1 : (index) -> i64
  %7 = fir.convert %6 : (i64) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10:3 = fir.box_dims %0#0, %c1 : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
  %11 = fir.convert %10#1 : (index) -> i64
  %12 = fir.convert %11 : (i64) -> index
  %13 = arith.cmpi sgt, %12, %c0 : index
  %14 = arith.select %13, %12, %c0 : index
  %15 = fir.alloca !fir.array<?x?xf64>, %9, %14 {bindc_name = "ddx", uniq_name = "_QFddxEddx"}
  %16 = fir.shape %9, %14 : (index, index) -> !fir.shape<2>
  %17:2 = hlfir.declare %15(%16) {uniq_name = "_QFddxEddx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %18 = fir.convert %5#1 : (index) -> i32
  hlfir.assign %18 to %2#0 : i32, !fir.ref<i32>
  %19 = fir.convert %10#1 : (index) -> i32
  hlfir.assign %19 to %4#0 : i32, !fir.ref<i32>
  %20 = fir.load %2#0 : !fir.ref<i32>
  %21 = fir.convert %20 : (i32) -> index
  %22 = arith.addi %21, %c-2 : index
  %23 = arith.cmpi sgt, %22, %c0 : index
  %24 = arith.select %23, %22, %c0 : index
  %25 = fir.load %4#0 : !fir.ref<i32>
  %26 = fir.convert %25 : (i32) -> index
  %27 = arith.cmpi sgt, %26, %c0 : index
  %28 = arith.select %27, %26, %c0 : index
  %29 = fir.shape %24, %28 : (index, index) -> !fir.shape<2>
  %30 = hlfir.designate %0#0 (%c3:%21:%c1, %c1:%26:%c1)  shape %29 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
  %31 = arith.subi %20, %c2_i32 : i32
  %32 = fir.convert %31 : (i32) -> index
  %33 = arith.cmpi sgt, %32, %c0 : index
  %34 = arith.select %33, %32, %c0 : index
  %35 = fir.shape %34, %28 : (index, index) -> !fir.shape<2>
  %36 = hlfir.designate %0#0 (%c1:%32:%c1, %c1:%26:%c1)  shape %35 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
  %37 = hlfir.elemental %29 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf64> {
  ^bb0(%arg1: index, %arg2: index):
    %46 = hlfir.designate %30 (%arg1, %arg2)  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
    %47 = hlfir.designate %36 (%arg1, %arg2)  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
    %48 = fir.load %46 : !fir.ref<f64>
    %49 = fir.load %47 : !fir.ref<f64>
    %50 = arith.subf %48, %49 fastmath<contract> : f64
    hlfir.yield_element %50 : f64
  }
  %38 = arith.subi %20, %c1_i32 : i32
  %39 = fir.convert %38 : (i32) -> index
  %40 = arith.addi %39, %c-1 : index
  %41 = arith.cmpi sgt, %40, %c0 : index
  %42 = arith.select %41, %40, %c0 : index
  %43 = fir.shape %42, %28 : (index, index) -> !fir.shape<2>
  %44 = hlfir.designate %17#0 (%c2:%39:%c1, %c1:%26:%c1)  shape %43 : (!fir.box<!fir.array<?x?xf64>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
  hlfir.assign %37 to %44 : !hlfir.expr<?x?xf64>, !fir.box<!fir.array<?x?xf64>>
  hlfir.destroy %37 : !hlfir.expr<?x?xf64>
  %45 = fir.load %17#1 : !fir.ref<!fir.array<?x?xf64>>
  return %45 : !fir.array<?x?xf64>
}
// CHECK-LABEL:   func.func @_QPddx(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "array"}) -> !fir.array<?x?xf64> {
// CHECK:           %[[VAL_1:.*]] = arith.constant -1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant -2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFddxEarray"} : (!fir.box<!fir.array<?x?xf64>>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>)
// CHECK:           %[[VAL_10:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFddxEi"}
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFddxEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_12:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFddxEj"}
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFddxEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_9]]#0, %[[VAL_8]] : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]]#1 : (index) -> i64
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
// CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_8]] : index
// CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_16]], %[[VAL_8]] : index
// CHECK:           %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_9]]#0, %[[VAL_7]] : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]]#1 : (index) -> i64
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
// CHECK:           %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_21]], %[[VAL_8]] : index
// CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_21]], %[[VAL_8]] : index
// CHECK:           %[[VAL_24:.*]] = fir.alloca !fir.array<?x?xf64>, %[[VAL_18]], %[[VAL_23]] {bindc_name = "ddx", uniq_name = "_QFddxEddx"}
// CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_18]], %[[VAL_23]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_24]](%[[VAL_25]]) {uniq_name = "_QFddxEddx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_14]]#1 : (index) -> i32
// CHECK:           hlfir.assign %[[VAL_27]] to %[[VAL_11]]#0 : i32, !fir.ref<i32>
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_19]]#1 : (index) -> i32
// CHECK:           hlfir.assign %[[VAL_28]] to %[[VAL_13]]#0 : i32, !fir.ref<i32>
// CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
// CHECK:           %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_2]] : index
// CHECK:           %[[VAL_32:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_8]] : index
// CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_31]], %[[VAL_8]] : index
// CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> index
// CHECK:           %[[VAL_36:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_8]] : index
// CHECK:           %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_35]], %[[VAL_8]] : index
// CHECK:           %[[VAL_38:.*]] = fir.shape %[[VAL_33]], %[[VAL_37]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_39:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_6]]:%[[VAL_30]]:%[[VAL_7]], %[[VAL_7]]:%[[VAL_35]]:%[[VAL_7]])  shape %[[VAL_38]] : (!fir.box<!fir.array<?x?xf64>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
// CHECK:           %[[VAL_40:.*]] = arith.subi %[[VAL_29]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> index
// CHECK:           %[[VAL_42:.*]] = arith.cmpi sgt, %[[VAL_41]], %[[VAL_8]] : index
// CHECK:           %[[VAL_43:.*]] = arith.select %[[VAL_42]], %[[VAL_41]], %[[VAL_8]] : index
// CHECK:           %[[VAL_44:.*]] = fir.shape %[[VAL_43]], %[[VAL_37]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_45:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_7]]:%[[VAL_41]]:%[[VAL_7]], %[[VAL_7]]:%[[VAL_35]]:%[[VAL_7]])  shape %[[VAL_44]] : (!fir.box<!fir.array<?x?xf64>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
// CHECK:           %[[VAL_46:.*]] = arith.subi %[[VAL_29]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> index
// CHECK:           %[[VAL_48:.*]] = arith.addi %[[VAL_47]], %[[VAL_1]] : index
// CHECK:           %[[VAL_49:.*]] = arith.cmpi sgt, %[[VAL_48]], %[[VAL_8]] : index
// CHECK:           %[[VAL_50:.*]] = arith.select %[[VAL_49]], %[[VAL_48]], %[[VAL_8]] : index
// CHECK:           %[[VAL_51:.*]] = fir.shape %[[VAL_50]], %[[VAL_37]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_52:.*]] = hlfir.designate %[[VAL_26]]#0 (%[[VAL_4]]:%[[VAL_47]]:%[[VAL_7]], %[[VAL_7]]:%[[VAL_35]]:%[[VAL_7]])  shape %[[VAL_51]] : (!fir.box<!fir.array<?x?xf64>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
// CHECK:           fir.do_loop %[[VAL_53:.*]] = %[[VAL_7]] to %[[VAL_37]] step %[[VAL_7]] unordered {
// CHECK:             fir.do_loop %[[VAL_54:.*]] = %[[VAL_7]] to %[[VAL_33]] step %[[VAL_7]] unordered {
// CHECK:               %[[VAL_55:.*]] = hlfir.designate %[[VAL_39]] (%[[VAL_54]], %[[VAL_53]])  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
// CHECK:               %[[VAL_56:.*]] = hlfir.designate %[[VAL_45]] (%[[VAL_54]], %[[VAL_53]])  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
// CHECK:               %[[VAL_57:.*]] = fir.load %[[VAL_55]] : !fir.ref<f64>
// CHECK:               %[[VAL_58:.*]] = fir.load %[[VAL_56]] : !fir.ref<f64>
// CHECK:               %[[VAL_59:.*]] = arith.subf %[[VAL_57]], %[[VAL_58]] fastmath<contract> : f64
// CHECK:               %[[VAL_60:.*]] = hlfir.designate %[[VAL_52]] (%[[VAL_54]], %[[VAL_53]])  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
// CHECK:               hlfir.assign %[[VAL_59]] to %[[VAL_60]] : f64, !fir.ref<f64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_61:.*]] = fir.load %[[VAL_26]]#1 : !fir.ref<!fir.array<?x?xf64>>
// CHECK:           return %[[VAL_61]] : !fir.array<?x?xf64>
// CHECK:         }
