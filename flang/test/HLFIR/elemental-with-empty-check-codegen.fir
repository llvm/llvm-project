// Test hlfir.elemental code generation with a dynamic check
// for empty result array
// RUN: fir-opt %s --bufferize-hlfir=opt-empty-elementals=true | FileCheck %s

func.func @test(%v: i32, %e0: i32, %e1: i32, %e2: i64, %e3: i64) {
  %shape = fir.shape %e0, %e1, %e2, %e3 : (i32, i32, i64, i64) -> !fir.shape<4>
  %result = hlfir.elemental %shape : (!fir.shape<4>) -> !hlfir.expr<?x?x?x?xi32> {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    hlfir.yield_element %v : i32
  }
  return
}
// CHECK-LABEL:   func.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32,
// CHECK-SAME:                    %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64) {
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (i32, i32, i64, i64) -> !fir.shape<4>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
// CHECK:           %[[VAL_10:.*]] = fir.allocmem !fir.array<?x?x?x?xi32>, %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_5]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?x?x?x?xi32>>, !fir.shape<4>) -> (!fir.box<!fir.array<?x?x?x?xi32>>, !fir.heap<!fir.array<?x?x?x?xi32>>)
// CHECK:           %[[VAL_12:.*]] = arith.constant true
// CHECK:           %[[VAL_13:.*]] = arith.constant false
// CHECK:           %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_6]], %[[C0_1]] : index
// CHECK:           %[[VAL_16:.*]] = arith.ori %[[VAL_13]], %[[VAL_15]] : i1
// CHECK:           %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_7]], %[[C0_2]] : index
// CHECK:           %[[VAL_18:.*]] = arith.ori %[[VAL_16]], %[[VAL_17]] : i1
// CHECK:           %[[C0_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_8]], %[[C0_3]] : index
// CHECK:           %[[VAL_20:.*]] = arith.ori %[[VAL_18]], %[[VAL_19]] : i1
// CHECK:           %[[C0_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_9]], %[[C0_4]] : index
// CHECK:           %[[VAL_22:.*]] = arith.ori %[[VAL_20]], %[[VAL_21]] : i1
// CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[C0_1]], %[[VAL_6]] : index
// CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_22]], %[[C0_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_25:.*]] = arith.select %[[VAL_22]], %[[C0_3]], %[[VAL_8]] : index
// CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_22]], %[[C0_4]], %[[VAL_9]] : index
// CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_28:.*]] = %[[VAL_27]] to %[[VAL_26]] step %[[VAL_27]] {
// CHECK:             fir.do_loop %[[VAL_29:.*]] = %[[VAL_27]] to %[[VAL_25]] step %[[VAL_27]] {
// CHECK:               fir.do_loop %[[VAL_30:.*]] = %[[VAL_27]] to %[[VAL_24]] step %[[VAL_27]] {
// CHECK:                 fir.do_loop %[[VAL_31:.*]] = %[[VAL_27]] to %[[VAL_23]] step %[[VAL_27]] {
// CHECK:                   %[[VAL_32:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_31]], %[[VAL_30]], %[[VAL_29]], %[[VAL_28]])  : (!fir.box<!fir.array<?x?x?x?xi32>>, index, index, index, index) -> !fir.ref<i32>
// CHECK:                   hlfir.assign %[[VAL_0]] to %[[VAL_32]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x?x?x?xi32>>, i1>
// CHECK:           %[[VAL_34:.*]] = fir.insert_value %[[VAL_33]], %[[VAL_12]], [1 : index] : (tuple<!fir.box<!fir.array<?x?x?x?xi32>>, i1>, i1) -> tuple<!fir.box<!fir.array<?x?x?x?xi32>>, i1>
// CHECK:           %[[VAL_35:.*]] = fir.insert_value %[[VAL_34]], %[[VAL_11]]#0, [0 : index] : (tuple<!fir.box<!fir.array<?x?x?x?xi32>>, i1>, !fir.box<!fir.array<?x?x?x?xi32>>) -> tuple<!fir.box<!fir.array<?x?x?x?xi32>>, i1>
// CHECK:           return
// CHECK:         }
