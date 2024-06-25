// Test code generation of hlfir.forall and hlfir.where when temporary
// storage is needed and can be allocated inline.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @test_scalar_save(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  hlfir.forall lb {
    hlfir.yield %c1_i32 : i32
  } ub {
    hlfir.yield %c10_i32 : i32
  }  (%arg1: i32) {
    hlfir.region_assign {
      %1 = fir.convert %arg1 : (i32) -> i64
      %2 = hlfir.designate %0#0 (%1)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      %3 = fir.load %2 : !fir.ref<i32>
      hlfir.yield %3 : i32
    } to {
      %1 = arith.addi %arg1, %c1_i32 : i32
      %2 = fir.convert %1 : (i32) -> i64
      %3 = hlfir.designate %0#0 (%2)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      hlfir.yield %3 : !fir.ref<i32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_scalar_save(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_6]], %[[VAL_5]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_7]] : index
// CHECK:           %[[VAL_11:.*]] = arith.divsi %[[VAL_10]], %[[VAL_7]] : index
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_8]] : index
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_8]] : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:           %[[VAL_16:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_13]] {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_16]](%[[VAL_17]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.do_loop %[[VAL_19:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_7]] {
// CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
// CHECK:             %[[VAL_22:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_21]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<i32>
// CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_15]] : index
// CHECK:             fir.store %[[VAL_25]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:             %[[VAL_26:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_24]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_23]] to %[[VAL_26]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:           fir.do_loop %[[VAL_30:.*]] = %[[VAL_27]] to %[[VAL_28]] step %[[VAL_29]] {
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (index) -> i32
// CHECK:             %[[VAL_32:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_15]] : index
// CHECK:             fir.store %[[VAL_33]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_32]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<i32>
// CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_31]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> i64
// CHECK:             %[[VAL_38:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_37]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_35]] to %[[VAL_38]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           fir.freemem %[[VAL_16]] : !fir.heap<!fir.array<?xi32>>
// CHECK:           return
// CHECK:         }

func.func @mask_and_rhs_conflict(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c42_i32 = arith.constant 42 : i32
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  hlfir.forall lb {
    hlfir.yield %c1_i32 : i32
  } ub {
    hlfir.yield %c10_i32 : i32
  }  (%arg1: i32) {
    hlfir.forall_mask {
      %1 = fir.convert %arg1 : (i32) -> i64
      %2 = hlfir.designate %0#0 (%1)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      %3 = fir.load %2 : !fir.ref<i32>
      %4 = arith.cmpi sgt, %3, %c42_i32 : i32
      hlfir.yield %4 : i1
    } do {
      hlfir.region_assign {
        %1 = fir.convert %arg1 : (i32) -> i64
        %2 = hlfir.designate %0#0 (%1)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
        %3 = fir.load %2 : !fir.ref<i32>
        hlfir.yield %3 : i32
      } to {
        %1 = arith.addi %arg1, %c1_i32 : i32
        %2 = fir.convert %1 : (i32) -> i64
        %3 = hlfir.designate %0#0 (%2)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
        hlfir.yield %3 : !fir.ref<i32>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @mask_and_rhs_conflict(
// CHECK-SAME:                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca index
// CHECK:           %[[VAL_2:.*]] = fir.alloca index
// CHECK:           %[[VAL_3:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = arith.subi %[[VAL_8]], %[[VAL_7]] : index
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_9]] : index
// CHECK:           %[[VAL_13:.*]] = arith.divsi %[[VAL_12]], %[[VAL_9]] : index
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_10]] : index
// CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_13]], %[[VAL_10]] : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_16]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:           %[[VAL_18:.*]] = fir.allocmem !fir.array<?xi1>, %[[VAL_15]] {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_18]](%[[VAL_19]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi1>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi1>>, !fir.heap<!fir.array<?xi1>>)
// CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_8]], %[[VAL_7]] : index
// CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_9]] : index
// CHECK:           %[[VAL_24:.*]] = arith.divsi %[[VAL_23]], %[[VAL_9]] : index
// CHECK:           %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_21]] : index
// CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_24]], %[[VAL_21]] : index
// CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_28:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_27]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:           %[[VAL_29:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_26]] {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[VAL_30:.*]] = fir.shape %[[VAL_26]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_31:.*]]:2 = hlfir.declare %[[VAL_29]](%[[VAL_30]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.do_loop %[[VAL_32:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_9]] {
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (index) -> i32
// CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
// CHECK:             %[[VAL_35:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_34]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:             %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<i32>
// CHECK:             %[[VAL_37:.*]] = arith.cmpi sgt, %[[VAL_36]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
// CHECK:             %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_17]] : index
// CHECK:             fir.store %[[VAL_39]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:             %[[VAL_40:.*]] = hlfir.designate %[[VAL_20]]#0 (%[[VAL_38]])  : (!fir.box<!fir.array<?xi1>>, index) -> !fir.ref<i1>
// CHECK:             hlfir.assign %[[VAL_37]] to %[[VAL_40]] : i1, !fir.ref<i1>
// CHECK:             fir.if %[[VAL_37]] {
// CHECK:               %[[VAL_41:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
// CHECK:               %[[VAL_42:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_41]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:               %[[VAL_43:.*]] = fir.load %[[VAL_42]] : !fir.ref<i32>
// CHECK:               %[[VAL_44:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_28]] : index
// CHECK:               fir.store %[[VAL_45]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:               %[[VAL_46:.*]] = hlfir.designate %[[VAL_31]]#0 (%[[VAL_44]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_43]] to %[[VAL_46]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
// CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
// CHECK:           %[[VAL_49:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_16]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:           fir.store %[[VAL_27]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:           fir.do_loop %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_48]] step %[[VAL_49]] {
// CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (index) -> i32
// CHECK:             %[[VAL_52:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
// CHECK:             %[[VAL_53:.*]] = arith.addi %[[VAL_52]], %[[VAL_17]] : index
// CHECK:             fir.store %[[VAL_53]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:             %[[VAL_54:.*]] = hlfir.designate %[[VAL_20]]#0 (%[[VAL_52]])  : (!fir.box<!fir.array<?xi1>>, index) -> !fir.ref<i1>
// CHECK:             %[[VAL_55:.*]] = fir.load %[[VAL_54]] : !fir.ref<i1>
// CHECK:             fir.if %[[VAL_55]] {
// CHECK:               %[[VAL_56:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
// CHECK:               %[[VAL_57:.*]] = arith.addi %[[VAL_56]], %[[VAL_28]] : index
// CHECK:               fir.store %[[VAL_57]] to %[[VAL_1]] : !fir.ref<index>
// CHECK:               %[[VAL_58:.*]] = hlfir.designate %[[VAL_31]]#0 (%[[VAL_56]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_59:.*]] = fir.load %[[VAL_58]] : !fir.ref<i32>
// CHECK:               %[[VAL_60:.*]] = arith.addi %[[VAL_51]], %[[VAL_5]] : i32
// CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i32) -> i64
// CHECK:               %[[VAL_62:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_61]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_59]] to %[[VAL_62]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       fir.freemem %[[VAL_18]] : !fir.heap<!fir.array<?xi1>>
// CHECK-DAG:       fir.freemem %[[VAL_29]] : !fir.heap<!fir.array<?xi32>>
// CHECK:           return
// CHECK:         }

func.func @test_where_mask_save(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : index
  %c42_i32 = arith.constant 42 : i32
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  hlfir.where {
    %1:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
    %2 = fir.shape %1#1 : (index) -> !fir.shape<1>
    %3 = hlfir.elemental %2 : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
    ^bb0(%arg1: index):
      %4 = hlfir.designate %0#0 (%arg1)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      %5 = fir.load %4 : !fir.ref<i32>
      %6 = arith.cmpi sgt, %5, %c42_i32 : i32
      %7 = fir.convert %6 : (i1) -> !fir.logical<4>
      hlfir.yield_element %7 : !fir.logical<4>
    }
    hlfir.yield %3 : !hlfir.expr<?x!fir.logical<4>> cleanup {
      hlfir.destroy %3 : !hlfir.expr<?x!fir.logical<4>>
    }
  } do {
    hlfir.region_assign {
      hlfir.yield %c42_i32 : i32
    } to {
      hlfir.yield %0#0 : !fir.box<!fir.array<?xi32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_where_mask_save(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_7]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i1) -> !fir.logical<4>
// CHECK:             hlfir.yield_element %[[VAL_11]] : !fir.logical<4>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]]:3 = hlfir.associate %[[VAL_13:.*]](%[[VAL_5]]) {uniq_name = ".tmp.where"} : (!hlfir.expr<?x!fir.logical<4>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.ref<!fir.array<?x!fir.logical<4>>>, i1)
// CHECK:           hlfir.destroy %[[VAL_13]] : !hlfir.expr<?x!fir.logical<4>>
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_15:.*]] = %[[VAL_14]] to %[[VAL_4]]#1 step %[[VAL_14]] {
// CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_15]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_18]] {
// CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_15]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_2]] to %[[VAL_19]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.end_associate %[[VAL_12]]#1, %[[VAL_12]]#2 : !fir.ref<!fir.array<?x!fir.logical<4>>>, i1
// CHECK:           return
// CHECK:         }

func.func @test_where_rhs_save(%x: !fir.ref<!fir.array<10xi32>>, %mask: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %2 = hlfir.designate %x (%c10:%c1:%c-1)  shape %1 :
(!fir.ref<!fir.array<10xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
      hlfir.yield %2 : !fir.ref<!fir.array<10xi32>>
    } to {
      hlfir.yield %x : !fir.ref<!fir.array<10xi32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_where_rhs_save(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
// CHECK:           %[[VAL_2:.*]] = fir.alloca index
// CHECK:           %[[VAL_3:.*]] = arith.constant -1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]] = arith.subi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_9]] : index
// CHECK:           %[[VAL_14:.*]] = arith.divsi %[[VAL_13]], %[[VAL_9]] : index
// CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_11]] : index
// CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_14]], %[[VAL_11]] : index
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_17]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:           %[[VAL_19:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_16]] {bindc_name = ".tmp.where", uniq_name = ""}
// CHECK:           %[[VAL_20:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_19]](%[[VAL_20]]) {uniq_name = ".tmp.where"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.do_loop %[[VAL_22:.*]] = %[[VAL_9]] to %[[VAL_7]] step %[[VAL_9]] {
// CHECK:             %[[VAL_23:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_22]])  : (!fir.ref<!fir.array<10x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_25]] {
// CHECK:           %[[VAL_10:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_5]]:%[[VAL_4]]:%[[VAL_3]])  shape %[[VAL_6]] : (!fir.ref<!fir.array<10xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_10]] (%[[VAL_22]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<i32>
// CHECK:               %[[VAL_28:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
// CHECK:               %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_18]] : index
// CHECK:               fir.store %[[VAL_29]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:               %[[VAL_30:.*]] = hlfir.designate %[[VAL_21]]#0 (%[[VAL_28]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_27]] to %[[VAL_30]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_32:.*]] = fir.shape %[[VAL_31]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_33:.*]] = arith.constant 1 : index
// CHECK:           fir.store %[[VAL_17]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:           fir.do_loop %[[VAL_34:.*]] = %[[VAL_33]] to %[[VAL_31]] step %[[VAL_33]] {
// CHECK:             %[[VAL_35:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_34]])  : (!fir.ref<!fir.array<10x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_37]] {
// CHECK:               %[[VAL_38:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
// CHECK:               %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_18]] : index
// CHECK:               fir.store %[[VAL_39]] to %[[VAL_2]] : !fir.ref<index>
// CHECK:               %[[VAL_40:.*]] = hlfir.designate %[[VAL_21]]#0 (%[[VAL_38]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.ref<i32>
// CHECK:               %[[VAL_42:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_34]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_41]] to %[[VAL_42]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.freemem %[[VAL_19]] : !fir.heap<!fir.array<?xi32>>
// CHECK:           return
// CHECK:         }
