// Test code generation of hlfir.forall and hlfir.where when temporary
// storage is needed and requires a runtime managed stack.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @test_runtime_stack(%arg0: !fir.box<!fir.array<?xi32>>, %n: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %c99_i32 = arith.constant 99 : i32
  %c100_i32 = arith.constant 100 : i32
  %c1_i32 = arith.constant 1 : i32
  %1:2 = hlfir.declare %n {uniq_name = "n"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = fir.load %1#1 : !fir.ref<i32>
  hlfir.forall lb {
    hlfir.yield %c1_i32 : i32
  } ub {
    hlfir.yield %c100_i32 : i32
  } step {
    hlfir.yield %3 : i32
  }  (%arg1: i32) {
    hlfir.region_assign {
      %4 = arith.subi %c100_i32, %arg1 : i32
      %5 = arith.subi %c99_i32, %arg1 : i32
      %6 = fir.load %1#0 : !fir.ref<i32>
      %7 = arith.subi %5, %6 : i32
      %8 = fir.convert %4 : (i32) -> index
      %9 = fir.convert %7 : (i32) -> index
      %10 = arith.subi %9, %8 : index
      %11 = arith.addi %10, %c-1 : index
      %12 = arith.divsi %11, %c-1 : index
      %13 = arith.cmpi sgt, %12, %c0 : index
      %14 = arith.select %13, %12, %c0 : index
      %15 = fir.shape %14 : (index) -> !fir.shape<1>
      %16 = hlfir.designate %2#0 (%8:%9:%c-1)  shape %15 : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
      hlfir.yield %16 : !fir.box<!fir.array<?xi32>>
    } to {
      %4 = fir.load %1#0 : !fir.ref<i32>
      %5 = arith.addi %arg1, %4 : i32
      %6 = arith.subi %5, %c1_i32 : i32
      %7 = fir.convert %arg1 : (i32) -> index
      %8 = fir.convert %6 : (i32) -> index
      %9 = arith.subi %8, %7 : index
      %10 = arith.addi %9, %c1 : index
      %11 = arith.cmpi sgt, %10, %c0 : index
      %12 = arith.select %11, %10, %c0 : index
      %13 = fir.shape %12 : (index) -> !fir.shape<1>
      %14 = hlfir.designate %2#0 (%7:%8:%c1)  shape %13 : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
      hlfir.yield %14 : !fir.box<!fir.array<?xi32>>
    }
  }
  return
}

// CHECK-LABEL:   func.func @test_runtime_stack(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK:           %[[VAL_3:.*]] = fir.alloca i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant -1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 99 : i32
// CHECK:           %[[VAL_8:.*]] = arith.constant 100 : i32
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "n"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_10]]#1 : !fir.ref<i32>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i64
// CHECK:           fir.store %[[VAL_16]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           %[[VAL_22:.*]] = fir.call @_FortranACreateValueStack(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK:           fir.do_loop %[[VAL_23:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_15]] {
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (index) -> i32
// CHECK:             %[[VAL_25:.*]] = arith.subi %[[VAL_8]], %[[VAL_24]] : i32
// CHECK:             %[[VAL_26:.*]] = arith.subi %[[VAL_7]], %[[VAL_24]] : i32
// CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i32>
// CHECK:             %[[VAL_28:.*]] = arith.subi %[[VAL_26]], %[[VAL_27]] : i32
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_25]] : (i32) -> index
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (i32) -> index
// CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_29]] : index
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_6]] : index
// CHECK:             %[[VAL_33:.*]] = arith.divsi %[[VAL_32]], %[[VAL_6]] : index
// CHECK:             %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_33]], %[[VAL_5]] : index
// CHECK:             %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_33]], %[[VAL_5]] : index
// CHECK:             %[[VAL_36:.*]] = fir.shape %[[VAL_35]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_37:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_29]]:%[[VAL_30]]:%[[VAL_6]])  shape %[[VAL_36]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:             %[[VAL_39:.*]] = fir.call @_FortranAPushValue(%[[VAL_22]], %[[VAL_38]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
// CHECK:           }
// CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
// CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
// CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
// CHECK:           fir.store %[[VAL_16]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:           fir.do_loop %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_41]] step %[[VAL_42]] {
// CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (index) -> i32
// CHECK:             %[[VAL_45:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_46:.*]] = arith.addi %[[VAL_45]], %[[VAL_17]] : i64
// CHECK:             fir.store %[[VAL_46]] to %[[VAL_3]] : !fir.ref<i64>
// CHECK:             %[[VAL_47:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_48:.*]] = fir.call @_FortranAValueAt(%[[VAL_22]], %[[VAL_45]], %[[VAL_47]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> none
// CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:             %[[VAL_50:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i32>
// CHECK:             %[[VAL_51:.*]] = arith.addi %[[VAL_44]], %[[VAL_50]] : i32
// CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_51]], %[[VAL_9]] : i32
// CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_44]] : (i32) -> index
// CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_52]] : (i32) -> index
// CHECK:             %[[VAL_55:.*]] = arith.subi %[[VAL_54]], %[[VAL_53]] : index
// CHECK:             %[[VAL_56:.*]] = arith.addi %[[VAL_55]], %[[VAL_4]] : index
// CHECK:             %[[VAL_57:.*]] = arith.cmpi sgt, %[[VAL_56]], %[[VAL_5]] : index
// CHECK:             %[[VAL_58:.*]] = arith.select %[[VAL_57]], %[[VAL_56]], %[[VAL_5]] : index
// CHECK:             %[[VAL_59:.*]] = fir.shape %[[VAL_58]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_60:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_53]]:%[[VAL_54]]:%[[VAL_4]])  shape %[[VAL_59]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             hlfir.assign %[[VAL_49]] to %[[VAL_60]] : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>
// CHECK:           }
// CHECK:           %[[VAL_61:.*]] = fir.call @_FortranADestroyValueStack(%[[VAL_22]]) : (!fir.llvm_ptr<i8>) -> none
// CHECK:           return
// CHECK:         }


// i1 needs to be passed as logical<> to be stored. Check converts are
// introduced as expected.
func.func @_QPdealing_with_i1(%x: !fir.ref<!fir.array<10x20xi32>>) {
  %c42_i32 = arith.constant 42 : i32
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  hlfir.forall lb {
    hlfir.yield %c1_i64 : i64
  } ub {
    hlfir.yield %c10_i64 : i64
  }  (%arg1: i64) {
    hlfir.forall lb {
      hlfir.yield %c1_i64 : i64
    } ub {
      hlfir.yield %arg1 : i64
    }  (%arg2: i64) {
      hlfir.forall_mask {
        %1 = hlfir.designate %x (%arg2, %arg1)  : (!fir.ref<!fir.array<10x20xi32>>, i64, i64) -> !fir.ref<i32>
        %2 = fir.load %1 : !fir.ref<i32>
        %3 = arith.cmpi sgt, %2, %c42_i32 : i32
        hlfir.yield %3 : i1
      } do {
        hlfir.region_assign {
          hlfir.yield %c42_i32 : i32
        } to {
          %1 = hlfir.designate %x (%arg1, %arg2)  : (!fir.ref<!fir.array<10x20xi32>>, i64, i64) -> !fir.ref<i32>
          hlfir.yield %1 : !fir.ref<i32>
        }
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @_QPdealing_with_i1(
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.logical<1>
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.logical<1>>>
// CHECK:           fir.do_loop
// CHECK:             fir.do_loop
// CHECK:               %[[VAL_26:.*]] = arith.cmpi sgt
// CHECK:               %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i1) -> !fir.logical<1>
// CHECK:               fir.store %[[VAL_27]] to %[[VAL_1]] : !fir.ref<!fir.logical<1>>
// CHECK:               %[[VAL_28:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.logical<1>>) -> !fir.box<!fir.logical<1>>
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.box<!fir.logical<1>>) -> !fir.box<none>
// CHECK:               %[[VAL_30:.*]] = fir.call @_FortranAPushValue(%{{.*}}, %[[VAL_29]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> none
// CHECK:             }
// CHECK:           }
// CHECK:           fir.do_loop
// CHECK:             fir.do_loop
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.logical<1>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:               %[[VAL_44:.*]] = fir.call @_FortranAValueAt(%{{.*}}, %{{.*}}, %[[VAL_43]])
// CHECK:               %[[VAL_45:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.logical<1>>>>
// CHECK:               %[[VAL_46:.*]] = fir.box_addr %[[VAL_45]] : (!fir.box<!fir.heap<!fir.logical<1>>>) -> !fir.heap<!fir.logical<1>>
// CHECK:               %[[VAL_47:.*]] = fir.load %[[VAL_46]] : !fir.heap<!fir.logical<1>>
// CHECK:               %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (!fir.logical<1>) -> i1
// CHECK:             }
// CHECK:           }
