// Test hlfir.reshape simplification to hlfir.elemental:
// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

func.func @reshape_simple(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xf32> {
  %res = hlfir.reshape %arg0 %arg1 : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xf32>
  return %res : !hlfir.expr<?xf32>
}
// CHECK-LABEL:   func.func @reshape_simple(
// CHECK-SAME:                              %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:                              %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (i32) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]]#0, %[[VAL_2]] overflow<nuw> : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] overflow<nuw> : index
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_11]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<f32>
// CHECK:             hlfir.yield_element %[[VAL_13]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_7]] : !hlfir.expr<?xf32>
// CHECK:         }

func.func @reshape_with_pad(%arg0: !fir.box<!fir.array<?x?x?xf32>>, %arg1: !fir.ref<!fir.array<2xi32>>, %arg2: !fir.box<!fir.array<?x?x?xf32>>) -> !hlfir.expr<?x?xf32> {
  %res = hlfir.reshape %arg0 %arg1 pad %arg2 : (!fir.box<!fir.array<?x?x?xf32>>, !fir.ref<!fir.array<2xi32>>, !fir.box<!fir.array<?x?x?xf32>>) -> !hlfir.expr<?x?xf32>
  return %res : !hlfir.expr<?x?xf32>
}
// CHECK-LABEL:   func.func @reshape_with_pad(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?xf32>>,
// CHECK-SAME:                                %[[VAL_1:.*]]: !fir.ref<!fir.array<2xi32>>,
// CHECK-SAME:                                %[[VAL_2:.*]]: !fir.box<!fir.array<?x?x?xf32>>) -> !hlfir.expr<?x?xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[ARRAY_DIM0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[ARRAY_DIM1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[ARRAY_DIM2:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[ARRAY_DIM0]]#1, %[[ARRAY_DIM1]]#1 overflow<nuw> : index
// CHECK:           %[[ARRAY_SIZE:.*]] = arith.muli %[[VAL_9]], %[[ARRAY_DIM2]]#1 overflow<nuw> : index
// CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_4]])  : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
// CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_3]])  : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
// CHECK:           %[[VAL_20:.*]] = fir.shape %[[VAL_17]], %[[VAL_19]] : (i32, i32) -> !fir.shape<2>
// CHECK:           %[[VAL_21:.*]] = hlfir.elemental %[[VAL_20]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
// CHECK:           ^bb0(%[[VAL_22:.*]]: index, %[[VAL_23:.*]]: index):
// CHECK:             %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_4]] overflow<nuw> : index
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
// CHECK:             %[[VAL_26:.*]] = arith.muli %[[VAL_24]], %[[VAL_25]] overflow<nuw> : index
// CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_22]], %[[VAL_4]] overflow<nuw> : index
// CHECK:             %[[LINEAR_INDEX:.*]] = arith.addi %[[VAL_26]], %[[VAL_27]] overflow<nuw> : index
// CHECK:             %[[IS_WITHIN_ARRAY:.*]] = arith.cmpi ult, %[[LINEAR_INDEX]], %[[ARRAY_SIZE]] : index
// CHECK:             %[[VAL_30:.*]] = fir.if %[[IS_WITHIN_ARRAY]] -> (f32) {
// CHECK:               %[[VAL_31:.*]] = arith.remui %[[LINEAR_INDEX]], %[[ARRAY_DIM0]]#1 : index
// CHECK:               %[[VAL_32:.*]] = arith.divui %[[LINEAR_INDEX]], %[[ARRAY_DIM0]]#1 : index
// CHECK:               %[[ARRAY_IDX0:.*]] = arith.addi %[[VAL_31]], %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_34:.*]] = arith.remui %[[VAL_32]], %[[ARRAY_DIM1]]#1 : index
// CHECK:               %[[VAL_35:.*]] = arith.divui %[[VAL_32]], %[[ARRAY_DIM1]]#1 : index
// CHECK:               %[[ARRAY_IDX1:.*]] = arith.addi %[[VAL_34]], %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[ARRAY_IDX2:.*]] = arith.addi %[[VAL_35]], %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_38:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_40:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_41:.*]] = arith.subi %[[VAL_38]]#0, %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[ARRAY_IDX0]], %[[VAL_41]] overflow<nuw> : index
// CHECK:               %[[VAL_43:.*]] = arith.subi %[[VAL_39]]#0, %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[ARRAY_IDX1]], %[[VAL_43]] overflow<nuw> : index
// CHECK:               %[[VAL_45:.*]] = arith.subi %[[VAL_40]]#0, %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[ARRAY_IDX2]], %[[VAL_45]] overflow<nuw> : index
// CHECK:               %[[VAL_47:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_42]], %[[VAL_44]], %[[VAL_46]])  : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_48:.*]] = fir.load %[[VAL_47]] : !fir.ref<f32>
// CHECK:               fir.result %[[VAL_48]] : f32
// CHECK:             } else {
// CHECK:               %[[PAD_DIM0:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[PAD_DIM1:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[PAD_DIM2:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[PAD_LINEAR_INDEX:.*]] = arith.subi %[[LINEAR_INDEX]], %[[ARRAY_SIZE]] overflow<nuw> : index
// CHECK:               %[[VAL_51:.*]] = arith.remui %[[PAD_LINEAR_INDEX]], %[[PAD_DIM0]]#1 : index
// CHECK:               %[[VAL_52:.*]] = arith.divui %[[PAD_LINEAR_INDEX]], %[[PAD_DIM0]]#1 : index
// CHECK:               %[[PAD_IDX0:.*]] = arith.addi %[[VAL_51]], %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_54:.*]] = arith.remui %[[VAL_52]], %[[PAD_DIM1]]#1 : index
// CHECK:               %[[VAL_55:.*]] = arith.divui %[[VAL_52]], %[[PAD_DIM1]]#1 : index
// CHECK:               %[[PAD_IDX1:.*]] = arith.addi %[[VAL_54]], %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_56:.*]] = arith.remui %[[VAL_55]], %[[PAD_DIM2]]#1 : index
// CHECK:               %[[PAD_IDX2:.*]] = arith.addi %[[VAL_56]], %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_58:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_59:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_60:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_61:.*]] = arith.subi %[[VAL_58]]#0, %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_62:.*]] = arith.addi %[[PAD_IDX0]], %[[VAL_61]] overflow<nuw> : index
// CHECK:               %[[VAL_63:.*]] = arith.subi %[[VAL_59]]#0, %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_64:.*]] = arith.addi %[[PAD_IDX1]], %[[VAL_63]] overflow<nuw> : index
// CHECK:               %[[VAL_65:.*]] = arith.subi %[[VAL_60]]#0, %[[VAL_4]] overflow<nuw> : index
// CHECK:               %[[VAL_66:.*]] = arith.addi %[[PAD_IDX2]], %[[VAL_65]] overflow<nuw> : index
// CHECK:               %[[VAL_67:.*]] = hlfir.designate %[[VAL_2]] (%[[VAL_62]], %[[VAL_64]], %[[VAL_66]])  : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_68:.*]] = fir.load %[[VAL_67]] : !fir.ref<f32>
// CHECK:               fir.result %[[VAL_68]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_30]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_21]] : !hlfir.expr<?x?xf32>
// CHECK:         }

func.func @reshape_derived_obj(%arg0: !fir.ref<!fir.array<10x!fir.type<whatever>>>, %arg1: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>> {
  %res = hlfir.reshape %arg0 %arg1 : (!fir.ref<!fir.array<10x!fir.type<whatever>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>>
  return %res : !hlfir.expr<?x!fir.type<whatever>>
}
// CHECK-LABEL:   func.func @reshape_derived_obj(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<whatever>>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (i32) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<whatever>> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_7]])  : (!fir.ref<!fir.array<10x!fir.type<whatever>>>, index) -> !fir.ref<!fir.type<whatever>>
// CHECK:             hlfir.yield_element %[[VAL_8]] : !fir.ref<!fir.type<whatever>>
// CHECK:           }
// CHECK:           return %[[VAL_6]] : !hlfir.expr<?x!fir.type<whatever>>
// CHECK:         }

func.func @reshape_derived_expr(%arg0: !hlfir.expr<?x!fir.type<whatever>>, %arg1: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>> {
  %res = hlfir.reshape %arg0 %arg1 : (!hlfir.expr<?x!fir.type<whatever>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>>
  return %res : !hlfir.expr<?x!fir.type<whatever>>
}
// CHECK-LABEL:   func.func @reshape_derived_expr(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !hlfir.expr<?x!fir.type<whatever>>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (i32) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<whatever>> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_7]] : (!hlfir.expr<?x!fir.type<whatever>>, index) -> !hlfir.expr<!fir.type<whatever>>
// CHECK:             hlfir.yield_element %[[VAL_8]] : !hlfir.expr<!fir.type<whatever>>
// CHECK:           }
// CHECK:           return %[[VAL_6]] : !hlfir.expr<?x!fir.type<whatever>>
// CHECK:         }

func.func @reshape_poly_obj(%arg0: !fir.class<!fir.array<?x!fir.type<whatever>>>, %arg1: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>?> {
  %res = hlfir.reshape %arg0 %arg1 : (!fir.class<!fir.array<?x!fir.type<whatever>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>?>
  return %res : !hlfir.expr<?x!fir.type<whatever>?>
}
// CHECK-LABEL:   func.func @reshape_poly_obj(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.class<!fir.array<?x!fir.type<whatever>>>,
// CHECK-SAME:                                %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>?> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (i32) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] mold %[[VAL_0]] unordered : (!fir.shape<1>, !fir.class<!fir.array<?x!fir.type<whatever>>>) -> !hlfir.expr<?x!fir.type<whatever>?> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.class<!fir.array<?x!fir.type<whatever>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]]#0, %[[VAL_2]] overflow<nuw> : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] overflow<nuw> : index
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_11]])  : (!fir.class<!fir.array<?x!fir.type<whatever>>>, index) -> !fir.class<!fir.type<whatever>>
// CHECK:             hlfir.yield_element %[[VAL_12]] : !fir.class<!fir.type<whatever>>
// CHECK:           }
// CHECK:           return %[[VAL_7]] : !hlfir.expr<?x!fir.type<whatever>?>
// CHECK:         }

func.func @reshape_poly_expr(%arg0: !hlfir.expr<?x!fir.type<whatever>?>, %arg1: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>?> {
  %res = hlfir.reshape %arg0 %arg1 : (!hlfir.expr<?x!fir.type<whatever>?>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>?>
  return %res : !hlfir.expr<?x!fir.type<whatever>?>
}
// CHECK-LABEL:   func.func @reshape_poly_expr(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !hlfir.expr<?x!fir.type<whatever>?>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<whatever>?> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (i32) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] mold %[[VAL_0]] unordered : (!fir.shape<1>, !hlfir.expr<?x!fir.type<whatever>?>) -> !hlfir.expr<?x!fir.type<whatever>?> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_7]] : (!hlfir.expr<?x!fir.type<whatever>?>, index) -> !hlfir.expr<!fir.type<whatever>?>
// CHECK:             hlfir.yield_element %[[VAL_8]] : !hlfir.expr<!fir.type<whatever>?>
// CHECK:           }
// CHECK:           return %[[VAL_6]] : !hlfir.expr<?x!fir.type<whatever>?>
// CHECK:         }

func.func @reshape_char(%arg0: !fir.box<!fir.array<?x!fir.char<2,?>>>, %arg1: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.char<2,?>> {
  %res = hlfir.reshape %arg0 %arg1 : (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.char<2,?>>
  return %res : !hlfir.expr<?x!fir.char<2,?>>
}
// CHECK-LABEL:   func.func @reshape_char(
// CHECK-SAME:                            %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<2,?>>>,
// CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>) -> index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_5]], %[[VAL_4]] : index
// CHECK:           %[[VAL_7:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (i32) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] typeparams %[[VAL_6]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           ^bb0(%[[VAL_11:.*]]: index):
// CHECK:             %[[VAL_12:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>) -> index
// CHECK:             %[[VAL_13:.*]] = arith.divsi %[[VAL_12]], %[[VAL_4]] : index
// CHECK:             %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_15:.*]] = arith.subi %[[VAL_14]]#0, %[[VAL_2]] overflow<nuw> : index
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_11]], %[[VAL_15]] overflow<nuw> : index
// CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_16]])  typeparams %[[VAL_13]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index, index) -> !fir.boxchar<2>
// CHECK:             hlfir.yield_element %[[VAL_17]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           return %[[VAL_10]] : !hlfir.expr<?x!fir.char<2,?>>
// CHECK:         }

func.func @reshape_negative_result_array_have_different_types(%arg0: !fir.box<!fir.array<?x!fir.char<2,1>>>, %arg1: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.char<2,2>> {
  %res = hlfir.reshape %arg0 %arg1 : (!fir.box<!fir.array<?x!fir.char<2,1>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.char<2,2>>
  return %res : !hlfir.expr<?x!fir.char<2,2>>
}
// CHECK-LABEL:   func.func @reshape_negative_result_array_have_different_types(
// CHECK:           hlfir.reshape %{{.*}} %{{.*}} : (!fir.box<!fir.array<?x!fir.char<2>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.char<2,2>>

func.func @reshape_negative_array_pad_have_different_types(%arg0: !fir.box<!fir.array<?x!fir.char<2,2>>>, %arg1: !fir.ref<!fir.array<1xi32>>, %arg2: !fir.box<!fir.array<?x!fir.char<2,1>>>) -> !hlfir.expr<?x!fir.char<2,2>> {
  %res = hlfir.reshape %arg0 %arg1 pad %arg2 : (!fir.box<!fir.array<?x!fir.char<2,2>>>, !fir.ref<!fir.array<1xi32>>, !fir.box<!fir.array<?x!fir.char<2,1>>>) -> !hlfir.expr<?x!fir.char<2,2>>
  return %res : !hlfir.expr<?x!fir.char<2,2>>
}
// CHECK-LABEL:   func.func @reshape_negative_array_pad_have_different_types(
// CHECK:           hlfir.reshape %{{.*}} %{{.*}} pad %{{.*}} : (!fir.box<!fir.array<?x!fir.char<2,2>>>, !fir.ref<!fir.array<1xi32>>, !fir.box<!fir.array<?x!fir.char<2>>>) -> !hlfir.expr<?x!fir.char<2,2>>
