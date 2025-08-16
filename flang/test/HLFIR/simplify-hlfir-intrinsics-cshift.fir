// Test hlfir.cshift simplification to hlfir.elemental and hlfir.eval_in_mem:
// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

func.func @cshift_vector(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<i32>) -> !hlfir.expr<?xi32>{
  %res = hlfir.cshift %arg0 %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @cshift_vector(
// CHECK-SAME:                             %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                             %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
// CHECK:           %[[ISZERO:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_4]] : i64
// CHECK:           %[[EXTENT:.*]] = arith.select %[[ISZERO]], %[[VAL_3]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.remsi %[[VAL_10]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_12:.*]] = arith.xori %[[VAL_10]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_15:.*]] = arith.andi %[[VAL_14]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_11]], %[[EXTENT]] overflow<nsw, nuw> : i64
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_16]], %[[VAL_11]] : i64
// CHECK:           %[[VAL_18:.*]] = hlfir.eval_in_mem shape %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK:           ^bb0(%[[VAL_19:.*]]: !fir.ref<!fir.array<?xi32>>):
// CHECK:             %[[VAL_20:.*]] = fir.embox %[[VAL_19]](%[[VAL_7]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             %[[VAL_3:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> index
// CHECK:             %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)

// CHECK:             %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_21]]#2 : index
// CHECK:             fir.if %[[VAL_22]] {
// CHECK:               %[[VAL_23:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_2]]:%[[VAL_6]]#1:%[[VAL_2]])  shape %[[VAL_23]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:               %[[VAL_25:.*]] = fir.box_addr %[[VAL_24]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK:               %[[VAL_26:.*]] = fir.embox %[[VAL_25]](%[[VAL_23]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:               %[[VAL_36:.*]] = arith.subi %[[VAL_8]], %[[VAL_17]] overflow<nsw, nuw> : i64
// CHECK:               %[[VAL_27:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_28:.*]] = %[[VAL_2]] to %[[VAL_27]] step %[[VAL_2]] unordered {
// CHECK:                 %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
// CHECK:                 %[[VAL_34:.*]] = hlfir.designate %[[VAL_26]] (%[[VAL_29]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:                 %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<i32>
// CHECK:                 %[[VAL_37:.*]] = arith.addi %[[VAL_29]], %[[VAL_36]] overflow<nsw, nuw> : i64
// CHECK:                 %[[VAL_42:.*]] = hlfir.designate %[[VAL_20]] (%[[VAL_37]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:                 hlfir.assign %[[VAL_35]] to %[[VAL_42]] : i32, !fir.ref<i32>
// CHECK:               }
// CHECK:               %[[VAL_43:.*]] = arith.subi %[[VAL_8]], %[[VAL_17]] overflow<nsw, nuw> : i64
// CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_45:.*]] = %[[VAL_2]] to %[[VAL_44]] step %[[VAL_2]] unordered {
// CHECK:                 %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (index) -> i64
// CHECK:                 %[[VAL_47:.*]] = arith.addi %[[VAL_46]], %[[VAL_17]] overflow<nsw, nuw> : i64
// CHECK:                 %[[VAL_52:.*]] = hlfir.designate %[[VAL_26]] (%[[VAL_47]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:                 %[[VAL_53:.*]] = fir.load %[[VAL_52]] : !fir.ref<i32>
// CHECK:                 %[[VAL_58:.*]] = hlfir.designate %[[VAL_20]] (%[[VAL_46]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:                 hlfir.assign %[[VAL_53]] to %[[VAL_58]] : i32, !fir.ref<i32>
// CHECK:               }
// CHECK:             } else {
// CHECK:               %[[VAL_68:.*]] = arith.subi %[[VAL_8]], %[[VAL_17]] overflow<nsw, nuw> : i64
// CHECK:               %[[VAL_59:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_60:.*]] = %[[VAL_2]] to %[[VAL_59]] step %[[VAL_2]] unordered {
// CHECK:                 %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (index) -> i64
// CHECK:                 %[[VAL_62:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_63:.*]] = fir.convert %[[VAL_61]] : (i64) -> index
// CHECK:                 %[[VAL_64:.*]] = arith.subi %[[VAL_62]]#0, %[[VAL_2]] overflow<nsw, nuw> : index
// CHECK:                 %[[VAL_65:.*]] = arith.addi %[[VAL_63]], %[[VAL_64]] overflow<nsw, nuw> : index
// CHECK:                 %[[VAL_66:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_65]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:                 %[[VAL_67:.*]] = fir.load %[[VAL_66]] : !fir.ref<i32>
// CHECK:                 %[[VAL_69:.*]] = arith.addi %[[VAL_61]], %[[VAL_68]] overflow<nsw, nuw> : i64
// CHECK:                 %[[VAL_74:.*]] = hlfir.designate %[[VAL_20]] (%[[VAL_69]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:                 hlfir.assign %[[VAL_67]] to %[[VAL_74]] : i32, !fir.ref<i32>
// CHECK:               }
// CHECK:               %[[VAL_75:.*]] = arith.subi %[[VAL_8]], %[[VAL_17]] overflow<nsw, nuw> : i64
// CHECK:               %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_77:.*]] = %[[VAL_2]] to %[[VAL_76]] step %[[VAL_2]] unordered {
// CHECK:                 %[[VAL_78:.*]] = fir.convert %[[VAL_77]] : (index) -> i64
// CHECK:                 %[[VAL_79:.*]] = arith.addi %[[VAL_78]], %[[VAL_17]] overflow<nsw, nuw> : i64
// CHECK:                 %[[VAL_80:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_81:.*]] = fir.convert %[[VAL_79]] : (i64) -> index
// CHECK:                 %[[VAL_82:.*]] = arith.subi %[[VAL_80]]#0, %[[VAL_2]] overflow<nsw, nuw> : index
// CHECK:                 %[[VAL_83:.*]] = arith.addi %[[VAL_81]], %[[VAL_82]] overflow<nsw, nuw> : index
// CHECK:                 %[[VAL_84:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_83]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:                 %[[VAL_85:.*]] = fir.load %[[VAL_84]] : !fir.ref<i32>
// CHECK:                 %[[VAL_90:.*]] = hlfir.designate %[[VAL_20]] (%[[VAL_78]])  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
// CHECK:                 hlfir.assign %[[VAL_85]] to %[[VAL_90]] : i32, !fir.ref<i32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_18]] : !hlfir.expr<?xi32>
// CHECK:         }

func.func @cshift_2d_by_scalar(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.ref<i32>) -> !hlfir.expr<?x?xi32> {
  %dim = arith.constant 2 : i32
  %res = hlfir.cshift %arg0 %arg1 dim %dim : (!fir.box<!fir.array<?x?xi32>>, !fir.ref<i32>, i32) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @cshift_2d_by_scalar(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME:                                   %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32>) -> !hlfir.expr<?x?xi32> {
// CHECK:           %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_6]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
// CHECK:           %[[ISZERO:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_2]] : i64
// CHECK:           %[[EXTENT:.*]] = arith.select %[[ISZERO]], %[[ONE]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.remsi %[[VAL_10]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_12:.*]] = arith.xori %[[VAL_10]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_15:.*]] = arith.andi %[[VAL_14]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_11]], %[[EXTENT]] overflow<nsw, nuw> : i64
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_16]], %[[VAL_11]] : i64
// CHECK:           %[[VAL_18:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// CHECK:           ^bb0(%[[VAL_19:.*]]: index, %[[VAL_20:.*]]: index):
// CHECK:             %[[VAL_21:.*]] = arith.subi %[[VAL_8]], %[[VAL_17]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_22:.*]] = arith.subi %[[VAL_17]], %[[VAL_8]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
// CHECK:             %[[VAL_24:.*]] = arith.cmpi sle, %[[VAL_23]], %[[VAL_21]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_17]], %[[VAL_22]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_23]], %[[VAL_25]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
// CHECK:             %[[VAL_28:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_29:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_30:.*]] = arith.subi %[[VAL_28]]#0, %[[VAL_3]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_19]], %[[VAL_30]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_32:.*]] = arith.subi %[[VAL_29]]#0, %[[VAL_3]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_27]], %[[VAL_32]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_31]], %[[VAL_33]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_35]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_18]] : !hlfir.expr<?x?xi32>
// CHECK:         }

func.func @cshift_2d_by_vector(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?x?xi32> {
  %dim = arith.constant 2 : i32
  %res = hlfir.cshift %arg0 %arg1 dim %dim : (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?xi32>>, i32) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @cshift_2d_by_vector(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xi32>>,
// CHECK-SAME:                                   %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?x?xi32> {
// CHECK:           %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_6]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
// CHECK:             %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_12]]#0, %[[VAL_3]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_13]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_14]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
// CHECK:             %[[ISZERO:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_2]] : i64
// CHECK:             %[[EXTENT:.*]] = arith.select %[[ISZERO]], %[[ONE]], %[[VAL_8]] : i64
// CHECK:             %[[VAL_18:.*]] = arith.remsi %[[VAL_17]], %[[EXTENT]] : i64
// CHECK:             %[[VAL_19:.*]] = arith.xori %[[VAL_17]], %[[EXTENT]] : i64
// CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_18]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.andi %[[VAL_21]], %[[VAL_20]] : i1
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_18]], %[[EXTENT]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_22]], %[[VAL_23]], %[[VAL_18]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.subi %[[VAL_8]], %[[VAL_24]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_26:.*]] = arith.subi %[[VAL_24]], %[[VAL_8]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
// CHECK:             %[[VAL_28:.*]] = arith.cmpi sle, %[[VAL_27]], %[[VAL_25]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_24]], %[[VAL_26]] : i64
// CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_29]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
// CHECK:             %[[VAL_32:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_33:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_34:.*]] = arith.subi %[[VAL_32]]#0, %[[VAL_3]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_10]], %[[VAL_34]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_36:.*]] = arith.subi %[[VAL_33]]#0, %[[VAL_3]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_31]], %[[VAL_36]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_38:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_35]], %[[VAL_37]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_39]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_9]] : !hlfir.expr<?x?xi32>
// CHECK:         }

func.func @cshift_vector_char(%arg0: !fir.box<!fir.array<?x!fir.char<2,?>>>, %arg1: !fir.ref<i32>) -> !hlfir.expr<?x!fir.char<2,?>> {
  %res = hlfir.cshift %arg0 %arg1 : (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<i32>) -> !hlfir.expr<?x!fir.char<2,?>>
  return %res : !hlfir.expr<?x!fir.char<2,?>>
}
// CHECK-LABEL:   func.func @cshift_vector_char(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x!fir.char<2,?>>>,
// CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<i32>) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>) -> index
// CHECK:           %[[VAL_9:.*]] = arith.divsi %[[VAL_8]], %[[VAL_4]] : index
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
// CHECK:           %[[ISZERO:.*]] = arith.cmpi eq, %[[VAL_10]], %[[VAL_3]] : i64
// CHECK:           %[[EXTENT:.*]] = arith.select %[[ISZERO]], %[[ONE]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_13:.*]] = arith.remsi %[[VAL_12]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_14:.*]] = arith.xori %[[VAL_12]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_13]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_17:.*]] = arith.andi %[[VAL_16]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_13]], %[[EXTENT]] overflow<nsw, nuw> : i64
// CHECK:           %[[VAL_19:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_13]] : i64
// CHECK:           %[[VAL_20:.*]] = hlfir.elemental %[[VAL_7]] typeparams %[[VAL_9]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           ^bb0(%[[VAL_21:.*]]: index):
// CHECK:             %[[VAL_22:.*]] = arith.subi %[[VAL_10]], %[[VAL_19]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_19]], %[[VAL_10]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_21]] : (index) -> i64
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sle, %[[VAL_24]], %[[VAL_22]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_19]], %[[VAL_23]] : i64
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_24]], %[[VAL_26]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
// CHECK:             %[[VAL_29:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>) -> index
// CHECK:             %[[VAL_30:.*]] = arith.divsi %[[VAL_29]], %[[VAL_4]] : index
// CHECK:             %[[VAL_31:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_32:.*]] = arith.subi %[[VAL_31]]#0, %[[VAL_2]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_28]], %[[VAL_32]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_34:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_33]])  typeparams %[[VAL_30]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index, index) -> !fir.boxchar<2>
// CHECK:             hlfir.yield_element %[[VAL_34]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           return %[[VAL_20]] : !hlfir.expr<?x!fir.char<2,?>>
// CHECK:         }

func.func @cshift_vector_poly(%arg0: !fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, %arg1: i32) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?> {
  %res = hlfir.cshift %arg0 %arg1 : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, i32) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?>
  return %res : !hlfir.expr<?x!fir.type<_QFFtestTt>?>
}
// CHECK-LABEL:   func.func @cshift_vector_poly(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>,
// CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[ISZERO:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_3]] : i64
// CHECK:           %[[EXTENT:.*]] = arith.select %[[ISZERO]], %[[ONE]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_9:.*]] = arith.remsi %[[VAL_8]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_10:.*]] = arith.xori %[[VAL_8]], %[[EXTENT]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_13:.*]] = arith.andi %[[VAL_12]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_9]], %[[EXTENT]] overflow<nsw, nuw> : i64
// CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_13]], %[[VAL_14]], %[[VAL_9]] : i64
// CHECK:           %[[VAL_16:.*]] = hlfir.elemental %[[VAL_6]] mold %[[VAL_0]] unordered : (!fir.shape<1>, !fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>) -> !hlfir.expr<?x!fir.type<_QFFtestTt>?> {
// CHECK:           ^bb0(%[[VAL_17:.*]]: index):
// CHECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_7]], %[[VAL_15]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_19:.*]] = arith.subi %[[VAL_15]], %[[VAL_7]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
// CHECK:             %[[VAL_21:.*]] = arith.cmpi sle, %[[VAL_20]], %[[VAL_18]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_15]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_20]], %[[VAL_22]] overflow<nsw, nuw> : i64
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
// CHECK:             %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_26:.*]] = arith.subi %[[VAL_25]]#0, %[[VAL_2]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_24]], %[[VAL_26]] overflow<nsw, nuw> : index
// CHECK:             %[[VAL_28:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_27]])  : (!fir.class<!fir.array<?x!fir.type<_QFFtestTt>>>, index) -> !fir.class<!fir.type<_QFFtestTt>>
// CHECK:             hlfir.yield_element %[[VAL_28]] : !fir.class<!fir.type<_QFFtestTt>>
// CHECK:           }
// CHECK:           return %[[VAL_16]] : !hlfir.expr<?x!fir.type<_QFFtestTt>?>
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
// CHECK-NOT: hlfir.cshift

// Check that hlfir.cshift is converted to hlfir.elemental
// when the argument is an array expression:
func.func @cshift_vector_expr(%arg0: !hlfir.expr<?xi32>, %arg1: !fir.ref<i32>) -> !hlfir.expr<?xi32>{
  %res = hlfir.cshift %arg0 %arg1 : (!hlfir.expr<?xi32>, !fir.ref<i32>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @cshift_vector_expr(
// CHECK: hlfir.elemental
