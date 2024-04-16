// RUN: fir-opt --cfg-conversion-on-func-opt="always-execute-loop-body=true" %s | FileCheck %s
// RUN: fir-opt --cfg-conversion-on-func-opt %s | FileCheck %s --check-prefix=NOOPT

func.func @x(%addr : !fir.ref<index>) {
  %bound = arith.constant 452 : index
  %step = arith.constant 1 : index
  fir.do_loop %iv = %bound to %bound step %step {
    fir.call @y(%addr) : (!fir.ref<index>) -> ()
  }
  return
}

func.func private @y(%addr : !fir.ref<index>)


// CHECK-LABEL:   func @x(
// CHECK-SAME:            %[[VAL_0:.*]]: !fir.ref<index>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 452 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_1]] : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_2]] : index
// CHECK:           %[[VAL_5:.*]] = arith.divsi %[[VAL_4]], %[[VAL_2]] : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.cmpi sle, %[[VAL_5]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_7]], %[[VAL_8]], %[[VAL_5]] : index
// CHECK:           br ^bb1(%[[VAL_1]], %[[VAL_9]] : index, index)
// CHECK:         ^bb1(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
// CHECK:           cond_br %[[VAL_13]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           fir.call @y(%[[VAL_0]]) : (!fir.ref<index>) -> ()
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_2]] : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_16:.*]] = arith.subi %[[VAL_11]], %[[VAL_15]] : index
// CHECK:           br ^bb1(%[[VAL_14]], %[[VAL_16]] : index, index)
// CHECK:         ^bb3:
// CHECK:           return
// CHECK:         }
// CHECK:         func private @y(!fir.ref<index>)

// NOOPT-LABEL:   func @x(
// NOOPT-SAME:            %[[VAL_0:.*]]: !fir.ref<index>) {
// NOOPT:           %[[VAL_1:.*]] = arith.constant 452 : index
// NOOPT:           %[[VAL_2:.*]] = arith.constant 1 : index
// NOOPT:           %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_1]] : index
// NOOPT:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_2]] : index
// NOOPT:           %[[VAL_5:.*]] = arith.divsi %[[VAL_4]], %[[VAL_2]] : index
// NOOPT:           br ^bb1(%[[VAL_1]], %[[VAL_5]] : index, index)
// NOOPT:         ^bb1(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
// NOOPT:           %[[VAL_8:.*]] = arith.constant 0 : index
// NOOPT:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_8]] : index
// NOOPT:           cond_br %[[VAL_9]], ^bb2, ^bb3
// NOOPT:         ^bb2:
// NOOPT:           fir.call @y(%[[VAL_0]]) : (!fir.ref<index>) -> ()
// NOOPT:           %[[VAL_10:.*]] = arith.addi %[[VAL_6]], %[[VAL_2]] : index
// NOOPT:           %[[VAL_11:.*]] = arith.constant 1 : index
// NOOPT:           %[[VAL_12:.*]] = arith.subi %[[VAL_7]], %[[VAL_11]] : index
// NOOPT:           br ^bb1(%[[VAL_10]], %[[VAL_12]] : index, index)
// NOOPT:         ^bb3:
// NOOPT:           return
// NOOPT:         }
// NOOPT:         func private @y(!fir.ref<index>)
