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
// CHECK:           %[[VAL_1:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : index
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_6]] : index
// CHECK:           %[[VAL_10:.*]] = arith.divsi %[[VAL_9]], %[[VAL_6]] : index
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_7]] : index
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_7]] : index
// CHECK:           %[[VAL_13:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_12]] {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[VAL_14:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_12]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_14]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.do_loop %[[VAL_16:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
// CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_18]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
// CHECK:             %[[VAL_21:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_16]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_20]] to %[[VAL_21]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_24]] {
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
// CHECK:             %[[VAL_27:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_25]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<i32>
// CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_26]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
// CHECK:             %[[VAL_31:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_30]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_28]] to %[[VAL_31]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           fir.freemem %[[VAL_13]] : !fir.heap<!fir.array<?xi32>>
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
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : i32
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
// CHECK:           %[[MASK_TEMP:.*]] = fir.allocmem !fir.array<?xi1>, %[[VAL_13]] {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[MASK_SHAPE:.*]] = fir.shape_shift %[[VAL_5]], %[[VAL_13]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[MASK_DECL:.*]]:2 = hlfir.declare %[[MASK_TEMP]](%[[MASK_SHAPE]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi1>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi1>>, !fir.heap<!fir.array<?xi1>>)
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_17:.*]] = arith.subi %[[VAL_6]], %[[VAL_5]] : index
// CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_7]] : index
// CHECK:           %[[VAL_19:.*]] = arith.divsi %[[VAL_18]], %[[VAL_7]] : index
// CHECK:           %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_16]] : index
// CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_19]], %[[VAL_16]] : index
// CHECK:           %[[RHS_TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_21]] {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[RHS_SHAPE:.*]] = fir.shape_shift %[[VAL_5]], %[[VAL_21]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[RHS_DECL:.*]]:2 = hlfir.declare %[[RHS_TEMP]](%[[RHS_SHAPE]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.do_loop %[[VAL_24:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_7]] {
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
// CHECK:             %[[VAL_27:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_26]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_27]] : !fir.ref<i32>
// CHECK:             %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_28]], %[[VAL_1]] : i32
// CHECK:             %[[VAL_30:.*]] = hlfir.designate %[[MASK_DECL]]#0 (%[[VAL_24]])  : (!fir.box<!fir.array<?xi1>>, index) -> !fir.ref<i1>
// CHECK:             hlfir.assign %[[VAL_29]] to %[[VAL_30]] : i1, !fir.ref<i1>
// CHECK:             fir.if %[[VAL_29]] {
// CHECK:               %[[VAL_31:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
// CHECK:               %[[VAL_32:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_31]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:               %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<i32>
// CHECK:               %[[VAL_34:.*]] = hlfir.designate %[[RHS_DECL]]#0 (%[[VAL_24]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_33]] to %[[VAL_34]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
// CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           %[[VAL_37:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_38:.*]] = %[[VAL_35]] to %[[VAL_36]] step %[[VAL_37]] {
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (index) -> i32
// CHECK:             %[[VAL_40:.*]] = hlfir.designate %[[MASK_DECL]]#0 (%[[VAL_38]])  : (!fir.box<!fir.array<?xi1>>, index) -> !fir.ref<i1>
// CHECK:             %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.ref<i1>
// CHECK:             fir.if %[[VAL_41]] {
// CHECK:               %[[VAL_42:.*]] = hlfir.designate %[[RHS_DECL]]#0 (%[[VAL_38]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_43:.*]] = fir.load %[[VAL_42]] : !fir.ref<i32>
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_39]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
// CHECK:               %[[VAL_46:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_45]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_43]] to %[[VAL_46]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       fir.freemem %[[MASK_TEMP]] : !fir.heap<!fir.array<?xi1>>
// CHECK-DAG:       fir.freemem %[[RHS_TEMP]] : !fir.heap<!fir.array<?xi32>>
// CHECK:           return
// CHECK:         }

func.func @test_where_mask_save(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : index
  %c42_i32 = arith.constant 42 : i32
  %c1 = arith.constant 1 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  hlfir.where {
    %1:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
    %2 = fir.shape %1#1 : (index) -> !fir.shape<1>
    %3 = hlfir.elemental %2 : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
    ^bb0(%arg1: index):
      %add = arith.addi %arg1, %c1 : index
      %4 = hlfir.designate %0#0 (%add)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
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
// This case uses hlfir.associate (SimpleCopy) to save the whole mask outside
// of the where loop nest, which is unrelated to the new ArrayTemp path.
// CHECK-LABEL:   func.func @test_where_mask_save(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : i32
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[ADD:.*]] = arith.addi %[[VAL_7]], %[[C1]] : index
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[ADD]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
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
// CHECK:           %[[VAL_2:.*]] = arith.constant -1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_6]], %[[VAL_8]] : index
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_8]] : index
// CHECK:           %[[VAL_12:.*]] = arith.divsi %[[VAL_11]], %[[VAL_8]] : index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_9]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_9]] : index
// CHECK:           %[[VAL_15:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_14]] {bindc_name = ".tmp.where", uniq_name = ""}
// CHECK:           %[[VAL_16:.*]] = fir.shape_shift %[[VAL_8]], %[[VAL_14]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_15]](%[[VAL_16]]) {uniq_name = ".tmp.where"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.do_loop %[[VAL_18:.*]] = %[[VAL_8]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_18]])  : (!fir.ref<!fir.array<10x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_21]] {
// CHECK:               %[[SLICE:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_4]]:%[[VAL_3]]:%[[VAL_2]])  shape %[[VAL_5]] : (!fir.ref<!fir.array<10xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
// CHECK:               %[[VAL_22:.*]] = hlfir.designate %[[SLICE]] (%[[VAL_18]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<i32>
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_17]]#0 (%[[VAL_18]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_23]] to %[[VAL_24]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_25]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_28:.*]] = %[[VAL_27]] to %[[VAL_25]] step %[[VAL_27]] {
// CHECK:             %[[VAL_29:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_28]])  : (!fir.ref<!fir.array<10x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_31]] {
// CHECK:               %[[VAL_32:.*]] = hlfir.designate %[[VAL_17]]#0 (%[[VAL_28]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<i32>
// CHECK:               %[[VAL_34:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_28]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_33]] to %[[VAL_34]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.freemem %[[VAL_15]] : !fir.heap<!fir.array<?xi32>>
// CHECK:           return
// CHECK:         }
