// Test array-copy-value pass (copy elision) with fir.array_modify
// RUN: fir-opt %s --array-value-copy | FileCheck %s

// Test user_defined_assignment(arg0(:), arg1(:))
func.func @no_overlap(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %3, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6:2 = fir.array_modify %arg3, %arg2 : (!fir.array<100xf32>, index) -> (!fir.ref<f32>, !fir.array<100xf32>)
    fir.store %5 to %0 : !fir.ref<f32>
    fir.call @user_defined_assignment(%6#0, %0) : (!fir.ref<f32>, !fir.ref<f32>) -> ()
    fir.result %6#1 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %4 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}
// CHECK-LABEL:   func @no_overlap(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                     %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>) {
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 99 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca f32
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = fir.undefined !fir.array<100xf32>
// CHECK:           %[[VAL_9:.*]] = fir.undefined !fir.array<100xf32>
// CHECK:           %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_4]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_8]]) -> (!fir.array<100xf32>) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_13]] : index
// CHECK:             %[[VAL_15:.*]] = fir.array_coor %[[VAL_1]](%[[VAL_7]]) %[[VAL_14]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<f32>
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_11]], %[[VAL_17]] : index
// CHECK:             %[[VAL_19:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) %[[VAL_18]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             fir.store %[[VAL_16]] to %[[VAL_6]] : !fir.ref<f32>
// CHECK:             fir.call @user_defined_assignment(%[[VAL_19]], %[[VAL_6]]) : (!fir.ref<f32>, !fir.ref<f32>) -> ()
// CHECK:             fir.result %[[VAL_8]] : !fir.array<100xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }


// Test user_defined_assignment(arg0(:), arg0(100:1:-1))
func.func @overlap(%arg0: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.slice %c100, %c1, %c-1 : (index, index, index) -> !fir.slice<1>
  %4 = fir.array_load %arg0(%1) [%3] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  %5 = fir.do_loop %arg1 = %c0 to %c99 step %c1 unordered iter_args(%arg2 = %2) -> (!fir.array<100xf32>) {
    %6 = fir.array_fetch %4, %arg1 : (!fir.array<100xf32>, index) -> f32
    %7:2 = fir.array_modify %arg2, %arg1 : (!fir.array<100xf32>, index) -> (!fir.ref<f32>, !fir.array<100xf32>)
    fir.store %6 to %0 : !fir.ref<f32>
    fir.call @user_defined_assignment(%7#0, %0) : (!fir.ref<f32>, !fir.ref<f32>) -> ()
    fir.result %7#1 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %5 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}
// CHECK-LABEL:   func @overlap(
// CHECK-SAME:                  %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>) {
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 99 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant -1 : index
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca f32
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = fir.allocmem !fir.array<100xf32>{{$}}
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (index) -> index
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_12:.*]] = arith.subi %[[VAL_9]], %[[VAL_11]] : index
// CHECK:           fir.do_loop %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_11]] {
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : index
// CHECK:             %[[VAL_16:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) %[[VAL_15]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_13]], %[[VAL_18]] : index
// CHECK:             %[[VAL_20:.*]] = fir.array_coor %[[VAL_8]](%[[VAL_7]]) %[[VAL_19]] : (!fir.heap<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<f32>
// CHECK:             fir.store %[[VAL_17]] to %[[VAL_20]] : !fir.ref<f32>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = fir.undefined !fir.array<100xf32>
// CHECK:           %[[VAL_22:.*]] = fir.slice %[[VAL_1]], %[[VAL_3]], %[[VAL_4]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_23:.*]] = fir.undefined !fir.array<100xf32>
// CHECK:           %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_3]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_21]]) -> (!fir.array<100xf32>) {
// CHECK:             %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_25]], %[[VAL_27]] : index
// CHECK:             %[[VAL_29:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) {{\[}}%[[VAL_22]]] %[[VAL_28]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<f32>
// CHECK:             %[[VAL_31:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_25]], %[[VAL_31]] : index
// CHECK:             %[[VAL_33:.*]] = fir.array_coor %[[VAL_8]](%[[VAL_7]]) %[[VAL_32]] : (!fir.heap<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             fir.store %[[VAL_30]] to %[[VAL_6]] : !fir.ref<f32>
// CHECK:             fir.call @user_defined_assignment(%[[VAL_33]], %[[VAL_6]]) : (!fir.ref<f32>, !fir.ref<f32>) -> ()
// CHECK:             fir.result %[[VAL_21]] : !fir.array<100xf32>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_1]] : (index) -> index
// CHECK:           %[[VAL_35:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_36:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_34]], %[[VAL_36]] : index
// CHECK:           fir.do_loop %[[VAL_38:.*]] = %[[VAL_35]] to %[[VAL_37]] step %[[VAL_36]] {
// CHECK:             %[[VAL_39:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_38]], %[[VAL_39]] : index
// CHECK:             %[[VAL_41:.*]] = fir.array_coor %[[VAL_8]](%[[VAL_7]]) %[[VAL_40]] : (!fir.heap<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_43:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_38]], %[[VAL_43]] : index
// CHECK:             %[[VAL_45:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) %[[VAL_44]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_42:.*]] = fir.load %[[VAL_41]] : !fir.ref<f32>
// CHECK:             fir.store %[[VAL_42]] to %[[VAL_45]] : !fir.ref<f32>
// CHECK:           }
// CHECK:           fir.freemem %[[VAL_8]] : !fir.heap<!fir.array<100xf32>>
// CHECK:           return
// CHECK:         }

func.func private @user_defined_assignment(!fir.ref<f32>, !fir.ref<f32>)
