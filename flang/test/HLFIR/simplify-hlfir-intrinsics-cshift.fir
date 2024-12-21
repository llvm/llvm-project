// Test hlfir.cshift simplification to hlfir.elemental:
// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

func.func @cshift_vector(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<i32>) -> !hlfir.expr<?xi32>{
  %res = hlfir.cshift %arg0 %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @cshift_vector(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.ref<i32>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[VAL_26:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK:           ^bb0(%[[VAL_9:.*]]: index):
// CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_7]] : i64
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[VAL_5]] : i64
// CHECK:             %[[VAL_13:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:             %[[VAL_14:.*]] = arith.remsi %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_15:.*]] = arith.xori %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_19:.*]] = arith.andi %[[VAL_18]], %[[VAL_17]] : i1
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_14]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_19]], %[[VAL_20]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_5]] : i64
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
// CHECK:             %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_25]]#0, %[[VAL_26]] : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_23]], %[[VAL_27]] : index
// CHECK:             %[[VAL_29:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_28]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_30]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @cshift_2d_by_scalar(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.ref<i32>) -> !hlfir.expr<?x?xi32> {
  %dim = arith.constant 2 : i32
  %res = hlfir.cshift %arg0 %arg1 dim %dim : (!fir.box<!fir.array<?x?xi32>>, !fir.ref<i32>, i32) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @cshift_2d_by_scalar(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.ref<i32>) -> !hlfir.expr<?x?xi32> {
// CHECK:           %[[VAL_20:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_4]]#1, %[[VAL_6]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
// CHECK:           %[[VAL_11:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index):
// CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_10]] : i64
// CHECK:             %[[VAL_16:.*]] = arith.subi %[[VAL_15]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:             %[[VAL_18:.*]] = arith.remsi %[[VAL_16]], %[[VAL_17]] : i64
// CHECK:             %[[VAL_19:.*]] = arith.xori %[[VAL_16]], %[[VAL_17]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_20]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_18]], %[[VAL_20]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.andi %[[VAL_22]], %[[VAL_21]] : i1
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_18]], %[[VAL_17]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_23]], %[[VAL_24]], %[[VAL_18]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
// CHECK:             %[[VAL_29:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_31:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_33:.*]] = arith.subi %[[VAL_29]]#0, %[[VAL_5]] : index
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_12]], %[[VAL_33]] : index
// CHECK:             %[[VAL_35:.*]] = arith.subi %[[VAL_31]]#0, %[[VAL_5]] : index
// CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_27]], %[[VAL_35]] : index
// CHECK:             %[[VAL_37:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_34]], %[[VAL_36]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_37]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_38]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @cshift_2d_by_vector(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?x?xi32> {
  %dim = arith.constant 2 : i32
  %res = hlfir.cshift %arg0 %arg1 dim %dim : (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?xi32>>, i32) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @cshift_2d_by_vector(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?x?xi32> {
// CHECK:           %[[VAL_26:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_4]]#1, %[[VAL_6]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
// CHECK:             %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_15:.*]] = arith.subi %[[VAL_13]]#0, %[[VAL_5]] : index
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_10]], %[[VAL_15]] : index
// CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_16]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<i32>
// CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
// CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
// CHECK:             %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:             %[[VAL_24:.*]] = arith.remsi %[[VAL_22]], %[[VAL_23]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.xori %[[VAL_22]], %[[VAL_23]] : i64
// CHECK:             %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_25]], %[[VAL_26]] : i64
// CHECK:             %[[VAL_28:.*]] = arith.cmpi ne, %[[VAL_24]], %[[VAL_26]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.andi %[[VAL_28]], %[[VAL_27]] : i1
// CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_24]], %[[VAL_23]] : i64
// CHECK:             %[[VAL_31:.*]] = arith.select %[[VAL_29]], %[[VAL_30]], %[[VAL_24]] : i64
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
// CHECK:             %[[VAL_35:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_37:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_39:.*]] = arith.subi %[[VAL_35]]#0, %[[VAL_5]] : index
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_10]], %[[VAL_39]] : index
// CHECK:             %[[VAL_41:.*]] = arith.subi %[[VAL_37]]#0, %[[VAL_5]] : index
// CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_33]], %[[VAL_41]] : index
// CHECK:             %[[VAL_43:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_40]], %[[VAL_42]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_44]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @cshift_vector_char(%arg0: !fir.box<!fir.array<?x!fir.char<2,?>>>, %arg1: !fir.ref<i32>) -> !hlfir.expr<?x!fir.char<2,?>> {
  %res = hlfir.cshift %arg0 %arg1 : (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<i32>) -> !hlfir.expr<?x!fir.char<2,?>>
  return %res : !hlfir.expr<?x!fir.char<2,?>>
}
// CHECK-LABEL:   func.func @cshift_vector_char(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<2,?>>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.ref<i32>) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           %[[VAL_32:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_19:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>) -> index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_5]], %[[VAL_6]] : index
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
// CHECK:           %[[VAL_11:.*]] = hlfir.elemental %[[VAL_4]] typeparams %[[VAL_7]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]] : i64
// CHECK:             %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:             %[[VAL_17:.*]] = arith.remsi %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_18:.*]] = arith.xori %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_17]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.andi %[[VAL_21]], %[[VAL_20]] : i1
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_17]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_22]], %[[VAL_23]], %[[VAL_17]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
// CHECK:             %[[VAL_27:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>) -> index
// CHECK:             %[[VAL_29:.*]] = arith.divsi %[[VAL_27]], %[[VAL_6]] : index
// CHECK:             %[[VAL_31:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_33:.*]] = arith.subi %[[VAL_31]]#0, %[[VAL_32]] : index
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_26]], %[[VAL_33]] : index
// CHECK:             %[[VAL_35:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_34]])  typeparams %[[VAL_29]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index, index) -> !fir.boxchar<2>
// CHECK:             hlfir.yield_element %[[VAL_35]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @cshift_vector_poly(%arg0: !fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, %arg1: i32) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?> {
  %res = hlfir.cshift %arg0 %arg1 : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, i32) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?>
  return %res : !hlfir.expr<?x!fir.type<_QFFtestTt>?>
}
// CHECK-LABEL:   func.func @cshift_vector_poly(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: i32) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?> {
// CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_4]] mold %[[VAL_0]] unordered : (!fir.shape<1>, !fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (index) -> i64
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_6]] : i64
// CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_5]] : i64
// CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:             %[[VAL_13:.*]] = arith.remsi %[[VAL_11]], %[[VAL_12]] : i64
// CHECK:             %[[VAL_14:.*]] = arith.xori %[[VAL_11]], %[[VAL_12]] : i64
// CHECK:             %[[VAL_16:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_15]] : i64
// CHECK:             %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_13]], %[[VAL_15]] : i64
// CHECK:             %[[VAL_18:.*]] = arith.andi %[[VAL_17]], %[[VAL_16]] : i1
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : i64
// CHECK:             %[[VAL_20:.*]] = arith.select %[[VAL_18]], %[[VAL_19]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_5]] : i64
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
// CHECK:             %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_26:.*]] = arith.subi %[[VAL_24]]#0, %[[VAL_25]] : index
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_22]], %[[VAL_26]] : index
// CHECK:             %[[VAL_28:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_27]])  : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, index) -> !fir.class<!fir.type<_QFFtestTt>>
// CHECK:             hlfir.yield_element %[[VAL_28]] : !fir.class<!fir.type<_QFFtestTt>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// negative: non-constant dim argument
func.func @cshift_nonconst_dim(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: i32, %dim : i32) -> !hlfir.expr<?x?xi32> {
  %res = hlfir.cshift %arg0 %arg1 dim %dim : (!fir.box<!fir.array<?x?xi32>>, i32, i32) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @cshift_nonconst_dim(
// CHECK:           hlfir.cshift {{.*}} : (!fir.box<!fir.array<?x?xi32>>, i32, i32) -> !hlfir.expr<?x?xi32>

// negative: invalid constant dim argument
func.func @cshift_invalid_dim(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: i32) -> !hlfir.expr<?x?xi32> {
  %dim = arith.constant 3 : i32
  %res = hlfir.cshift %arg0 %arg1 dim %dim : (!fir.box<!fir.array<?x?xi32>>, i32, i32) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @cshift_invalid_dim(
// CHECK:           hlfir.cshift {{.*}} : (!fir.box<!fir.array<?x?xi32>>, i32, i32) -> !hlfir.expr<?x?xi32>

// When the input array is 1D, we may assume that DIM==1,
// otherwise the program is illegal, and we can do anything
// about it.
func.func @cshift_vector_assumed_dim_1(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: i32) -> !hlfir.expr<?xi32> {
  %dim = arith.constant 3 : i32
  %res = hlfir.cshift %arg0 %arg1 dim %dim : (!fir.box<!fir.array<?xi32>>, i32, i32) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @cshift_vector_assumed_dim_1(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: i32) -> !hlfir.expr<?xi32> {
// CHECK:           %[[VAL_26:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK:           ^bb0(%[[VAL_9:.*]]: index):
// CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_7]] : i64
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[VAL_6]] : i64
// CHECK:             %[[VAL_13:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
// CHECK:             %[[VAL_14:.*]] = arith.remsi %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_15:.*]] = arith.xori %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_19:.*]] = arith.andi %[[VAL_18]], %[[VAL_17]] : i1
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_14]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_19]], %[[VAL_20]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_6]] : i64
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
// CHECK:             %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_25]]#0, %[[VAL_26]] : index
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_23]], %[[VAL_27]] : index
// CHECK:             %[[VAL_29:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_28]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_30]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
