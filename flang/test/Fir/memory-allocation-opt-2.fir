// Test memory allocation pass for fir.alloca outside of function entry block
// RUN: fir-opt --memory-allocation-opt="dynamic-array-on-heap=true" %s | FileCheck %s

func.func @test_loop() {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  fir.do_loop %arg0 = %c1 to %c100 step %c1 {
    %1 = fir.alloca !fir.array<?xf32>, %arg0
    fir.call @bar(%1) : (!fir.ref<!fir.array<?xf32>>) -> ()
    fir.result
  }
  return
}
// CHECK-LABEL:   func.func @test_loop() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_0]] {
// CHECK:             %[[VAL_3:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_2]] {bindc_name = "", uniq_name = ""}
// CHECK:             %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:             fir.call @bar(%[[VAL_4]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
// CHECK:             fir.freemem %[[VAL_3]] : !fir.heap<!fir.array<?xf32>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @test_unstructured(%n : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.alloca index
  fir.store %c100 to %0 : !fir.ref<index>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb4
  %5 = fir.load %0 : !fir.ref<index>
  %6 = arith.cmpi sgt, %5, %c0 : index
  cf.cond_br %6, ^bb2, ^bb5
^bb2:  // pred: ^bb1
  %1 = fir.alloca !fir.array<?xf32>, %5
  fir.call @bar(%1) : (!fir.ref<!fir.array<?xf32>>) -> ()
  %25 = arith.cmpi slt, %5, %n : index
  cf.cond_br %25, ^bb3, ^bb4
^bb3:  // pred: ^bb2
  fir.call @abort() : () -> ()
  fir.unreachable
^bb4:  // pred: ^bb2
  %28 = arith.subi %5, %c1 : index
  fir.store %28 to %0 : !fir.ref<index>
  cf.br ^bb1
^bb5:  // pred: ^bb1
  return
}
// CHECK-LABEL:   func.func @test_unstructured(
// CHECK-SAME:                                 %[[VAL_0:.*]]: index) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_2:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca index
// CHECK:           fir.store %[[VAL_6]] to %[[VAL_7]] : !fir.ref<index>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<index>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_4]] : index
// CHECK:           cf.cond_br %[[VAL_9]], ^bb2, ^bb5
// CHECK:         ^bb2:
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.heap<!fir.array<?xf32>>) -> i64
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_3]] : i64
// CHECK:           fir.if %[[VAL_12]] {
// CHECK:             fir.freemem %[[VAL_10]] : !fir.heap<!fir.array<?xf32>>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_8]] {bindc_name = "", uniq_name = ""}
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           fir.call @bar(%[[VAL_14]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
// CHECK:           %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_0]] : index
// CHECK:           cf.cond_br %[[VAL_15]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           fir.call @abort() : () -> ()
// CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.heap<!fir.array<?xf32>>) -> i64
// CHECK:           %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_17]], %[[VAL_3]] : i64
// CHECK:           fir.if %[[VAL_18]] {
// CHECK:             fir.freemem %[[VAL_16]] : !fir.heap<!fir.array<?xf32>>
// CHECK:           }
// CHECK:           fir.unreachable
// CHECK:         ^bb4:
// CHECK:           %[[VAL_19:.*]] = arith.subi %[[VAL_8]], %[[VAL_5]] : index
// CHECK:           fir.store %[[VAL_19]] to %[[VAL_7]] : !fir.ref<index>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb5:
// CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.heap<!fir.array<?xf32>>) -> i64
// CHECK:           %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_3]] : i64
// CHECK:           fir.if %[[VAL_22]] {
// CHECK:             fir.freemem %[[VAL_20]] : !fir.heap<!fir.array<?xf32>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @alloca_dominate_return_in_cycle(%arg0: index) {
  %0 = fir.alloca index
  %c1 = arith.constant 1 : index
  fir.store %c1 to %0 : !fir.ref<index>
  cf.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb2
  %1 = fir.load %0 : !fir.ref<index>
  %2 = fir.alloca !fir.array<?xf32>, %1
  fir.call @bar(%2) : (!fir.ref<!fir.array<?xf32>>) -> ()
  %3 = arith.addi %1, %c1 : index
  fir.store %3 to %0 : !fir.ref<index>
  %4 = arith.cmpi slt, %3, %arg0 : index
  cf.cond_br %4, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb1
^bb3:  // pred: ^bb1
  return
}
// CHECK-LABEL:   func.func @alloca_dominate_return_in_cycle(
// CHECK-SAME:                                               %[[VAL_0:.*]]: index) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_2:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = fir.alloca index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<index>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<?xf32>>) -> i64
// CHECK:           %[[VAL_9:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_3]] : i64
// CHECK:           fir.if %[[VAL_9]] {
// CHECK:             fir.freemem %[[VAL_7]] : !fir.heap<!fir.array<?xf32>>
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_6]] {bindc_name = "", uniq_name = ""}
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:           fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           fir.call @bar(%[[VAL_11]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_6]], %[[VAL_5]] : index
// CHECK:           fir.store %[[VAL_12]] to %[[VAL_4]] : !fir.ref<index>
// CHECK:           %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_0]] : index
// CHECK:           cf.cond_br %[[VAL_13]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb1
// CHECK:         ^bb3:
// CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.heap<!fir.array<?xf32>>) -> i64
// CHECK:           %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_15]], %[[VAL_3]] : i64
// CHECK:           fir.if %[[VAL_16]] {
// CHECK:             fir.freemem %[[VAL_14]] : !fir.heap<!fir.array<?xf32>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func private @bar(!fir.ref<!fir.array<?xf32>>)
func.func private @abort()
