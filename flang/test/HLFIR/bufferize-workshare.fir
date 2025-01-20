// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

// CHECK-LABEL:   func.func @simple(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           omp.parallel {
// CHECK:             omp.workshare {
// CHECK:               %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:               %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:               %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:               %[[VAL_5:.*]] = fir.allocmem !fir.array<42xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:               %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_3]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
// CHECK:               %[[VAL_7:.*]] = arith.constant true
// CHECK:               %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:               omp.workshare.loop_wrapper {
// CHECK:                 omp.loop_nest (%[[VAL_9:.*]]) : index = (%[[VAL_8]]) to (%[[VAL_1]]) inclusive step (%[[VAL_8]]) {
// CHECK:                   %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:                   %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                   %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[VAL_2]] : i32
// CHECK:                   %[[VAL_13:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_9]])  : (!fir.heap<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:                   hlfir.assign %[[VAL_12]] to %[[VAL_13]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:                   omp.yield
// CHECK:                 }
// CHECK:               }
// CHECK:               %[[VAL_14:.*]] = fir.undefined tuple<!fir.heap<!fir.array<42xi32>>, i1>
// CHECK:               %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_7]], [1 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
// CHECK:               %[[VAL_16:.*]] = fir.insert_value %[[VAL_15]], %[[VAL_6]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, !fir.heap<!fir.array<42xi32>>) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
// CHECK:               hlfir.assign %[[VAL_6]]#0 to %[[VAL_4]]#0 : !fir.heap<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
// CHECK:               fir.freemem %[[VAL_6]]#0 : !fir.heap<!fir.array<42xi32>>
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @simple(%arg: !fir.ref<!fir.array<42xi32>>) {
  omp.parallel {
    omp.workshare {
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
      omp.terminator
    }
    omp.terminator
  }
  return
}
