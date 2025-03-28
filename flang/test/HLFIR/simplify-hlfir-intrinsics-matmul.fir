// Test hlfir.cshift simplification to hlfir.elemental:
// RUN: fir-opt --simplify-hlfir-intrinsics=allow-new-side-effects=false %s | FileCheck %s --check-prefixes=ALL,NOANSE
// RUN: fir-opt --simplify-hlfir-intrinsics=allow-new-side-effects=true %s | FileCheck %s --check-prefixes=ALL,ANSE
// RUN: fir-opt --simplify-hlfir-intrinsics -flang-inline-matmul-as-elemental %s | FileCheck %s --check-prefixes=ALL,ELEMENTAL

func.func @matmul_matrix_matrix_integer(%arg0: !hlfir.expr<?x?xi16>, %arg1: !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32> {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xi16>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// ALL-LABEL:   func.func @matmul_matrix_matrix_integer(
// ALL-SAME:                                            %[[VAL_0:.*]]: !hlfir.expr<?x?xi16>,
// ALL-SAME:                                            %[[VAL_1:.*]]: !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32> {

// NOANSE: hlfir.matmul

// ANSE:           %[[VAL_2:.*]] = arith.constant 0 : index
// ANSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// ANSE:           %[[VAL_4:.*]] = arith.constant 0 : i32
// ANSE:           %[[VAL_5:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xi16>) -> !fir.shape<2>
// ANSE:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
// ANSE:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_10:.*]] = fir.shape %[[VAL_6]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
// ANSE:           %[[VAL_11:.*]] = hlfir.eval_in_mem shape %[[VAL_10]] : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// ANSE:           ^bb0(%[[VAL_12:.*]]: !fir.ref<!fir.array<?x?xi32>>):
// ANSE:             %[[VAL_13:.*]] = fir.embox %[[VAL_12]](%[[VAL_10]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
// ANSE:             fir.do_loop %[[VAL_14:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered {
// ANSE:               fir.do_loop %[[VAL_15:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:                 %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_18:.*]] = arith.subi %[[VAL_16]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_19:.*]] = arith.addi %[[VAL_15]], %[[VAL_18]] : index
// ANSE:                 %[[VAL_20:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_21:.*]] = arith.addi %[[VAL_14]], %[[VAL_20]] : index
// ANSE:                 %[[VAL_22:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_19]], %[[VAL_21]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// ANSE:                 hlfir.assign %[[VAL_4]] to %[[VAL_22]] : i32, !fir.ref<i32>
// ANSE:               }
// ANSE:             }
// ANSE:             fir.do_loop %[[VAL_23:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_3]] unordered {
// ANSE:               fir.do_loop %[[VAL_24:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered {
// ANSE:                 fir.do_loop %[[VAL_25:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:                   %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_28:.*]] = arith.subi %[[VAL_26]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_29:.*]] = arith.addi %[[VAL_25]], %[[VAL_28]] : index
// ANSE:                   %[[VAL_30:.*]] = arith.subi %[[VAL_27]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_31:.*]] = arith.addi %[[VAL_24]], %[[VAL_30]] : index
// ANSE:                   %[[VAL_32:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_29]], %[[VAL_31]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// ANSE:                   %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<i32>
// ANSE:                   %[[VAL_34:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_25]], %[[VAL_23]] : (!hlfir.expr<?x?xi16>, index, index) -> i16
// ANSE:                   %[[VAL_35:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_23]], %[[VAL_24]] : (!hlfir.expr<?x?xi32>, index, index) -> i32
// ANSE:                   %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (i16) -> i32
// ANSE:                   %[[VAL_37:.*]] = arith.muli %[[VAL_36]], %[[VAL_35]] : i32
// ANSE:                   %[[VAL_38:.*]] = arith.addi %[[VAL_33]], %[[VAL_37]] : i32
// ANSE:                   hlfir.assign %[[VAL_38]] to %[[VAL_32]] : i32, !fir.ref<i32>
// ANSE:                 }
// ANSE:               }
// ANSE:             }
// ANSE:           }
// ANSE:           return %[[VAL_11]] : !hlfir.expr<?x?xi32>
// ANSE:         }

// ELEMENTAL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ELEMENTAL:           %[[VAL_3:.*]] = arith.constant 0 : i32
// ELEMENTAL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xi16>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// ELEMENTAL:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
// ELEMENTAL:             %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_3]]) -> (i32) {
// ELEMENTAL:               %[[VAL_16:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_14]] : (!hlfir.expr<?x?xi16>, index, index) -> i16
// ELEMENTAL:               %[[VAL_17:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_14]], %[[VAL_12]] : (!hlfir.expr<?x?xi32>, index, index) -> i32
// ELEMENTAL:               %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (i16) -> i32
// ELEMENTAL:               %[[VAL_19:.*]] = arith.muli %[[VAL_18]], %[[VAL_17]] : i32
// ELEMENTAL:               %[[VAL_20:.*]] = arith.addi %[[VAL_15]], %[[VAL_19]] : i32
// ELEMENTAL:               fir.result %[[VAL_20]] : i32
// ELEMENTAL:             }
// ELEMENTAL:             hlfir.yield_element %[[VAL_13]] : i32
// ELEMENTAL:           }
// ELEMENTAL:           return %[[VAL_10]] : !hlfir.expr<?x?xi32>
// ELEMENTAL:         }

func.func @matmul_matrix_matrix_real(%arg0: !hlfir.expr<?x?xf32>, %arg1: !hlfir.expr<?x?xf16>) -> !hlfir.expr<?x?xf32> {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xf32>, !hlfir.expr<?x?xf16>) -> !hlfir.expr<?x?xf32>
  return %res : !hlfir.expr<?x?xf32>
}
// ALL-LABEL:   func.func @matmul_matrix_matrix_real(
// ALL-SAME:                                         %[[VAL_0:.*]]: !hlfir.expr<?x?xf32>,
// ALL-SAME:                                         %[[VAL_1:.*]]: !hlfir.expr<?x?xf16>) -> !hlfir.expr<?x?xf32> {

// NOANSE: hlfir.matmul

// ANSE:           %[[VAL_2:.*]] = arith.constant 0 : index
// ANSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// ANSE:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// ANSE:           %[[VAL_5:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf32>) -> !fir.shape<2>
// ANSE:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xf16>) -> !fir.shape<2>
// ANSE:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_10:.*]] = fir.shape %[[VAL_6]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
// ANSE:           %[[VAL_11:.*]] = hlfir.eval_in_mem shape %[[VAL_10]] : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
// ANSE:           ^bb0(%[[VAL_12:.*]]: !fir.ref<!fir.array<?x?xf32>>):
// ANSE:             %[[VAL_13:.*]] = fir.embox %[[VAL_12]](%[[VAL_10]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
// ANSE:             fir.do_loop %[[VAL_14:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered {
// ANSE:               fir.do_loop %[[VAL_15:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:                 %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_18:.*]] = arith.subi %[[VAL_16]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_19:.*]] = arith.addi %[[VAL_15]], %[[VAL_18]] : index
// ANSE:                 %[[VAL_20:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_21:.*]] = arith.addi %[[VAL_14]], %[[VAL_20]] : index
// ANSE:                 %[[VAL_22:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_19]], %[[VAL_21]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// ANSE:                 hlfir.assign %[[VAL_4]] to %[[VAL_22]] : f32, !fir.ref<f32>
// ANSE:               }
// ANSE:             }
// ANSE:             fir.do_loop %[[VAL_23:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_3]] {
// ANSE:               fir.do_loop %[[VAL_24:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] {
// ANSE:                 fir.do_loop %[[VAL_25:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] {
// ANSE:                   %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_28:.*]] = arith.subi %[[VAL_26]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_29:.*]] = arith.addi %[[VAL_25]], %[[VAL_28]] : index
// ANSE:                   %[[VAL_30:.*]] = arith.subi %[[VAL_27]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_31:.*]] = arith.addi %[[VAL_24]], %[[VAL_30]] : index
// ANSE:                   %[[VAL_32:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_29]], %[[VAL_31]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// ANSE:                   %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<f32>
// ANSE:                   %[[VAL_34:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_25]], %[[VAL_23]] : (!hlfir.expr<?x?xf32>, index, index) -> f32
// ANSE:                   %[[VAL_35:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_23]], %[[VAL_24]] : (!hlfir.expr<?x?xf16>, index, index) -> f16
// ANSE:                   %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (f16) -> f32
// ANSE:                   %[[VAL_37:.*]] = arith.mulf %[[VAL_34]], %[[VAL_36]] : f32
// ANSE:                   %[[VAL_38:.*]] = arith.addf %[[VAL_33]], %[[VAL_37]] : f32
// ANSE:                   hlfir.assign %[[VAL_38]] to %[[VAL_32]] : f32, !fir.ref<f32>
// ANSE:                 }
// ANSE:               }
// ANSE:             }
// ANSE:           }
// ANSE:           return %[[VAL_11]] : !hlfir.expr<?x?xf32>
// ANSE:         }

// ELEMENTAL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ELEMENTAL:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// ELEMENTAL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf32>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xf16>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
// ELEMENTAL:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
// ELEMENTAL:             %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_3]]) -> (f32) {
// ELEMENTAL:               %[[VAL_16:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_14]] : (!hlfir.expr<?x?xf32>, index, index) -> f32
// ELEMENTAL:               %[[VAL_17:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_14]], %[[VAL_12]] : (!hlfir.expr<?x?xf16>, index, index) -> f16
// ELEMENTAL:               %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (f16) -> f32
// ELEMENTAL:               %[[VAL_19:.*]] = arith.mulf %[[VAL_16]], %[[VAL_18]] : f32
// ELEMENTAL:               %[[VAL_20:.*]] = arith.addf %[[VAL_15]], %[[VAL_19]] : f32
// ELEMENTAL:               fir.result %[[VAL_20]] : f32
// ELEMENTAL:             }
// ELEMENTAL:             hlfir.yield_element %[[VAL_13]] : f32
// ELEMENTAL:           }
// ELEMENTAL:           return %[[VAL_10]] : !hlfir.expr<?x?xf32>
// ELEMENTAL:         }

func.func @matmul_matrix_matrix_complex(%arg0: !hlfir.expr<?x?xcomplex<f32>>, %arg1: !hlfir.expr<?x?xcomplex<f16>>) -> !hlfir.expr<?x?xcomplex<f32>> {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xcomplex<f32>>, !hlfir.expr<?x?xcomplex<f16>>) -> !hlfir.expr<?x?xcomplex<f32>>
  return %res : !hlfir.expr<?x?xcomplex<f32>>
}
// ALL-LABEL:   func.func @matmul_matrix_matrix_complex(
// ALL-SAME:                                            %[[VAL_0:.*]]: !hlfir.expr<?x?xcomplex<f32>>,
// ALL-SAME:                                            %[[VAL_1:.*]]: !hlfir.expr<?x?xcomplex<f16>>) -> !hlfir.expr<?x?xcomplex<f32>> {

// NOANSE: hlfir.matmul

// ANSE:           %[[VAL_2:.*]] = arith.constant 0 : index
// ANSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// ANSE:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// ANSE:           %[[VAL_5:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xcomplex<f32>>) -> !fir.shape<2>
// ANSE:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xcomplex<f16>>) -> !fir.shape<2>
// ANSE:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_10:.*]] = fir.shape %[[VAL_6]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
// ANSE:           %[[VAL_11:.*]] = hlfir.eval_in_mem shape %[[VAL_10]] : (!fir.shape<2>) -> !hlfir.expr<?x?xcomplex<f32>> {
// ANSE:           ^bb0(%[[VAL_12:.*]]: !fir.ref<!fir.array<?x?xcomplex<f32>>>):
// ANSE:             %[[VAL_13:.*]] = fir.embox %[[VAL_12]](%[[VAL_10]]) : (!fir.ref<!fir.array<?x?xcomplex<f32>>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
// ANSE:             %[[VAL_14:.*]] = fir.undefined complex<f32>
// ANSE:             %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_4]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ANSE:             %[[VAL_16:.*]] = fir.insert_value %[[VAL_15]], %[[VAL_4]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// ANSE:             fir.do_loop %[[VAL_17:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered {
// ANSE:               fir.do_loop %[[VAL_18:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:                 %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_21:.*]] = arith.subi %[[VAL_19]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_22:.*]] = arith.addi %[[VAL_18]], %[[VAL_21]] : index
// ANSE:                 %[[VAL_23:.*]] = arith.subi %[[VAL_20]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_24:.*]] = arith.addi %[[VAL_17]], %[[VAL_23]] : index
// ANSE:                 %[[VAL_25:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_22]], %[[VAL_24]])  : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index, index) -> !fir.ref<complex<f32>>
// ANSE:                 hlfir.assign %[[VAL_16]] to %[[VAL_25]] : complex<f32>, !fir.ref<complex<f32>>
// ANSE:               }
// ANSE:             }
// ANSE:             fir.do_loop %[[VAL_26:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_3]] {
// ANSE:               fir.do_loop %[[VAL_27:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] {
// ANSE:                 fir.do_loop %[[VAL_28:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] {
// ANSE:                   %[[VAL_29:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_30:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_31:.*]] = arith.subi %[[VAL_29]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_32:.*]] = arith.addi %[[VAL_28]], %[[VAL_31]] : index
// ANSE:                   %[[VAL_33:.*]] = arith.subi %[[VAL_30]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_34:.*]] = arith.addi %[[VAL_27]], %[[VAL_33]] : index
// ANSE:                   %[[VAL_35:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_32]], %[[VAL_34]])  : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index, index) -> !fir.ref<complex<f32>>
// ANSE:                   %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<complex<f32>>
// ANSE:                   %[[VAL_37:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_28]], %[[VAL_26]] : (!hlfir.expr<?x?xcomplex<f32>>, index, index) -> complex<f32>
// ANSE:                   %[[VAL_38:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_26]], %[[VAL_27]] : (!hlfir.expr<?x?xcomplex<f16>>, index, index) -> complex<f16>
// ANSE:                   %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (complex<f16>) -> complex<f32>
// ANSE:                   %[[VAL_40:.*]] = fir.mulc %[[VAL_37]], %[[VAL_39]] : complex<f32>
// ANSE:                   %[[VAL_41:.*]] = fir.addc %[[VAL_36]], %[[VAL_40]] : complex<f32>
// ANSE:                   hlfir.assign %[[VAL_41]] to %[[VAL_35]] : complex<f32>, !fir.ref<complex<f32>>
// ANSE:                 }
// ANSE:               }
// ANSE:             }
// ANSE:           }
// ANSE:           return %[[VAL_11]] : !hlfir.expr<?x?xcomplex<f32>>
// ANSE:         }

// ELEMENTAL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ELEMENTAL:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// ELEMENTAL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xcomplex<f32>>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xcomplex<f16>>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xcomplex<f32>> {
// ELEMENTAL:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
// ELEMENTAL:             %[[VAL_13:.*]] = fir.undefined complex<f32>
// ELEMENTAL:             %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_3]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ELEMENTAL:             %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_3]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// ELEMENTAL:             %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (complex<f32>) {
// ELEMENTAL:               %[[VAL_19:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_17]] : (!hlfir.expr<?x?xcomplex<f32>>, index, index) -> complex<f32>
// ELEMENTAL:               %[[VAL_20:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_17]], %[[VAL_12]] : (!hlfir.expr<?x?xcomplex<f16>>, index, index) -> complex<f16>
// ELEMENTAL:               %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (complex<f16>) -> complex<f32>
// ELEMENTAL:               %[[VAL_22:.*]] = fir.mulc %[[VAL_19]], %[[VAL_21]] : complex<f32>
// ELEMENTAL:               %[[VAL_23:.*]] = fir.addc %[[VAL_18]], %[[VAL_22]] : complex<f32>
// ELEMENTAL:               fir.result %[[VAL_23]] : complex<f32>
// ELEMENTAL:             }
// ELEMENTAL:             hlfir.yield_element %[[VAL_16]] : complex<f32>
// ELEMENTAL:           }
// ELEMENTAL:           return %[[VAL_10]] : !hlfir.expr<?x?xcomplex<f32>>
// ELEMENTAL:         }

func.func @matmul_matrix_matrix_complex_real(%arg0: !hlfir.expr<?x?xcomplex<f32>>, %arg1: !hlfir.expr<?x?xf16>) -> !hlfir.expr<?x?xcomplex<f32>> {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xcomplex<f32>>, !hlfir.expr<?x?xf16>) -> !hlfir.expr<?x?xcomplex<f32>>
  return %res : !hlfir.expr<?x?xcomplex<f32>>
}
// ALL-LABEL:   func.func @matmul_matrix_matrix_complex_real(
// ALL-SAME:                                                 %[[VAL_0:.*]]: !hlfir.expr<?x?xcomplex<f32>>,
// ALL-SAME:                                                 %[[VAL_1:.*]]: !hlfir.expr<?x?xf16>) -> !hlfir.expr<?x?xcomplex<f32>> {

// NOANSE: hlfir.matmul

// ANSE:           %[[VAL_2:.*]] = arith.constant 0 : index
// ANSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// ANSE:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// ANSE:           %[[VAL_5:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xcomplex<f32>>) -> !fir.shape<2>
// ANSE:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xf16>) -> !fir.shape<2>
// ANSE:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_10:.*]] = fir.shape %[[VAL_6]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
// ANSE:           %[[VAL_11:.*]] = hlfir.eval_in_mem shape %[[VAL_10]] : (!fir.shape<2>) -> !hlfir.expr<?x?xcomplex<f32>> {
// ANSE:           ^bb0(%[[VAL_12:.*]]: !fir.ref<!fir.array<?x?xcomplex<f32>>>):
// ANSE:             %[[VAL_13:.*]] = fir.embox %[[VAL_12]](%[[VAL_10]]) : (!fir.ref<!fir.array<?x?xcomplex<f32>>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xcomplex<f32>>>
// ANSE:             %[[VAL_14:.*]] = fir.undefined complex<f32>
// ANSE:             %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_4]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ANSE:             %[[VAL_16:.*]] = fir.insert_value %[[VAL_15]], %[[VAL_4]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// ANSE:             fir.do_loop %[[VAL_17:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered {
// ANSE:               fir.do_loop %[[VAL_18:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:                 %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_21:.*]] = arith.subi %[[VAL_19]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_22:.*]] = arith.addi %[[VAL_18]], %[[VAL_21]] : index
// ANSE:                 %[[VAL_23:.*]] = arith.subi %[[VAL_20]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_24:.*]] = arith.addi %[[VAL_17]], %[[VAL_23]] : index
// ANSE:                 %[[VAL_25:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_22]], %[[VAL_24]])  : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index, index) -> !fir.ref<complex<f32>>
// ANSE:                 hlfir.assign %[[VAL_16]] to %[[VAL_25]] : complex<f32>, !fir.ref<complex<f32>>
// ANSE:               }
// ANSE:             }
// ANSE:             fir.do_loop %[[VAL_26:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_3]] {
// ANSE:               fir.do_loop %[[VAL_27:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] {
// ANSE:                 fir.do_loop %[[VAL_28:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] {
// ANSE:                   %[[VAL_29:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_30:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_31:.*]] = arith.subi %[[VAL_29]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_32:.*]] = arith.addi %[[VAL_28]], %[[VAL_31]] : index
// ANSE:                   %[[VAL_33:.*]] = arith.subi %[[VAL_30]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_34:.*]] = arith.addi %[[VAL_27]], %[[VAL_33]] : index
// ANSE:                   %[[VAL_35:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_32]], %[[VAL_34]])  : (!fir.box<!fir.array<?x?xcomplex<f32>>>, index, index) -> !fir.ref<complex<f32>>
// ANSE:                   %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<complex<f32>>
// ANSE:                   %[[VAL_37:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_28]], %[[VAL_26]] : (!hlfir.expr<?x?xcomplex<f32>>, index, index) -> complex<f32>
// ANSE:                   %[[VAL_38:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_26]], %[[VAL_27]] : (!hlfir.expr<?x?xf16>, index, index) -> f16
// ANSE:                   %[[VAL_39:.*]] = fir.undefined complex<f32>
// ANSE:                   %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[VAL_4]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ANSE:                   %[[VAL_41:.*]] = fir.insert_value %[[VAL_40]], %[[VAL_4]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// ANSE:                   %[[VAL_42:.*]] = fir.convert %[[VAL_38]] : (f16) -> f32
// ANSE:                   %[[VAL_43:.*]] = fir.insert_value %[[VAL_41]], %[[VAL_42]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ANSE:                   %[[VAL_44:.*]] = fir.mulc %[[VAL_37]], %[[VAL_43]] : complex<f32>
// ANSE:                   %[[VAL_45:.*]] = fir.addc %[[VAL_36]], %[[VAL_44]] : complex<f32>
// ANSE:                   hlfir.assign %[[VAL_45]] to %[[VAL_35]] : complex<f32>, !fir.ref<complex<f32>>
// ANSE:                 }
// ANSE:               }
// ANSE:             }
// ANSE:           }
// ANSE:           return %[[VAL_11]] : !hlfir.expr<?x?xcomplex<f32>>
// ANSE:         }

// ELEMENTAL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ELEMENTAL:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// ELEMENTAL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xcomplex<f32>>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xf16>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xcomplex<f32>> {
// ELEMENTAL:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
// ELEMENTAL:             %[[VAL_13:.*]] = fir.undefined complex<f32>
// ELEMENTAL:             %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_3]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ELEMENTAL:             %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_3]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// ELEMENTAL:             %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (complex<f32>) {
// ELEMENTAL:               %[[VAL_19:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_17]] : (!hlfir.expr<?x?xcomplex<f32>>, index, index) -> complex<f32>
// ELEMENTAL:               %[[VAL_20:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_17]], %[[VAL_12]] : (!hlfir.expr<?x?xf16>, index, index) -> f16
// ELEMENTAL:               %[[VAL_21:.*]] = fir.undefined complex<f32>
// ELEMENTAL:               %[[VAL_22:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_3]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ELEMENTAL:               %[[VAL_23:.*]] = fir.insert_value %[[VAL_22]], %[[VAL_3]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// ELEMENTAL:               %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (f16) -> f32
// ELEMENTAL:               %[[VAL_25:.*]] = fir.insert_value %[[VAL_23]], %[[VAL_24]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// ELEMENTAL:               %[[VAL_26:.*]] = fir.mulc %[[VAL_19]], %[[VAL_25]] : complex<f32>
// ELEMENTAL:               %[[VAL_27:.*]] = fir.addc %[[VAL_18]], %[[VAL_26]] : complex<f32>
// ELEMENTAL:               fir.result %[[VAL_27]] : complex<f32>
// ELEMENTAL:             }
// ELEMENTAL:             hlfir.yield_element %[[VAL_16]] : complex<f32>
// ELEMENTAL:           }
// ELEMENTAL:           return %[[VAL_10]] : !hlfir.expr<?x?xcomplex<f32>>
// ELEMENTAL:         }

func.func @matmul_matrix_matrix_logical(%arg0: !hlfir.expr<?x?x!fir.logical<1>>, %arg1: !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?x?x!fir.logical<4>> {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?x!fir.logical<1>>, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?x?x!fir.logical<4>>
  return %res : !hlfir.expr<?x?x!fir.logical<4>>
}
// ALL-LABEL:   func.func @matmul_matrix_matrix_logical(
// ALL-SAME:                                            %[[VAL_0:.*]]: !hlfir.expr<?x?x!fir.logical<1>>,
// ALL-SAME:                                            %[[VAL_1:.*]]: !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?x?x!fir.logical<4>> {

// NOANSE: hlfir.matmul

// ANSE:           %[[VAL_2:.*]] = arith.constant 0 : index
// ANSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// ANSE:           %[[VAL_4:.*]] = arith.constant false
// ANSE:           %[[VAL_5:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x!fir.logical<1>>) -> !fir.shape<2>
// ANSE:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?x!fir.logical<4>>) -> !fir.shape<2>
// ANSE:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_10:.*]] = fir.shape %[[VAL_6]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
// ANSE:           %[[VAL_11:.*]] = hlfir.eval_in_mem shape %[[VAL_10]] : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
// ANSE:           ^bb0(%[[VAL_12:.*]]: !fir.ref<!fir.array<?x?x!fir.logical<4>>>):
// ANSE:             %[[VAL_13:.*]] = fir.embox %[[VAL_12]](%[[VAL_10]]) : (!fir.ref<!fir.array<?x?x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?x!fir.logical<4>>>
// ANSE:             %[[VAL_14:.*]] = fir.convert %[[VAL_4]] : (i1) -> !fir.logical<4>
// ANSE:             fir.do_loop %[[VAL_15:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered {
// ANSE:               fir.do_loop %[[VAL_16:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:                 %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_19:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_20:.*]] = arith.addi %[[VAL_16]], %[[VAL_19]] : index
// ANSE:                 %[[VAL_21:.*]] = arith.subi %[[VAL_18]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_22:.*]] = arith.addi %[[VAL_15]], %[[VAL_21]] : index
// ANSE:                 %[[VAL_23:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_20]], %[[VAL_22]])  : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index, index) -> !fir.ref<!fir.logical<4>>
// ANSE:                 hlfir.assign %[[VAL_14]] to %[[VAL_23]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// ANSE:               }
// ANSE:             }
// ANSE:             fir.do_loop %[[VAL_24:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_3]] unordered {
// ANSE:               fir.do_loop %[[VAL_25:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered {
// ANSE:                 fir.do_loop %[[VAL_26:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:                   %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_2]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_28:.*]]:3 = fir.box_dims %[[VAL_13]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// ANSE:                   %[[VAL_29:.*]] = arith.subi %[[VAL_27]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_30:.*]] = arith.addi %[[VAL_26]], %[[VAL_29]] : index
// ANSE:                   %[[VAL_31:.*]] = arith.subi %[[VAL_28]]#0, %[[VAL_3]] : index
// ANSE:                   %[[VAL_32:.*]] = arith.addi %[[VAL_25]], %[[VAL_31]] : index
// ANSE:                   %[[VAL_33:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_30]], %[[VAL_32]])  : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index, index) -> !fir.ref<!fir.logical<4>>
// ANSE:                   %[[VAL_34:.*]] = fir.load %[[VAL_33]] : !fir.ref<!fir.logical<4>>
// ANSE:                   %[[VAL_35:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_26]], %[[VAL_24]] : (!hlfir.expr<?x?x!fir.logical<1>>, index, index) -> !fir.logical<1>
// ANSE:                   %[[VAL_36:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_24]], %[[VAL_25]] : (!hlfir.expr<?x?x!fir.logical<4>>, index, index) -> !fir.logical<4>
// ANSE:                   %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.logical<4>) -> i1
// ANSE:                   %[[VAL_38:.*]] = fir.convert %[[VAL_35]] : (!fir.logical<1>) -> i1
// ANSE:                   %[[VAL_39:.*]] = fir.convert %[[VAL_36]] : (!fir.logical<4>) -> i1
// ANSE:                   %[[VAL_40:.*]] = arith.andi %[[VAL_38]], %[[VAL_39]] : i1
// ANSE:                   %[[VAL_41:.*]] = arith.ori %[[VAL_37]], %[[VAL_40]] : i1
// ANSE:                   %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i1) -> !fir.logical<4>
// ANSE:                   hlfir.assign %[[VAL_42]] to %[[VAL_33]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// ANSE:                 }
// ANSE:               }
// ANSE:             }
// ANSE:           }
// ANSE:           return %[[VAL_11]] : !hlfir.expr<?x?x!fir.logical<4>>
// ANSE:         }

// ELEMENTAL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ELEMENTAL:           %[[VAL_3:.*]] = arith.constant false
// ELEMENTAL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x!fir.logical<1>>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?x!fir.logical<4>>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
// ELEMENTAL:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
// ELEMENTAL:             %[[VAL_13:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
// ELEMENTAL:             %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (!fir.logical<4>) {
// ELEMENTAL:               %[[VAL_17:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_15]] : (!hlfir.expr<?x?x!fir.logical<1>>, index, index) -> !fir.logical<1>
// ELEMENTAL:               %[[VAL_18:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_15]], %[[VAL_12]] : (!hlfir.expr<?x?x!fir.logical<4>>, index, index) -> !fir.logical<4>
// ELEMENTAL:               %[[VAL_19:.*]] = fir.convert %[[VAL_16]] : (!fir.logical<4>) -> i1
// ELEMENTAL:               %[[VAL_20:.*]] = fir.convert %[[VAL_17]] : (!fir.logical<1>) -> i1
// ELEMENTAL:               %[[VAL_21:.*]] = fir.convert %[[VAL_18]] : (!fir.logical<4>) -> i1
// ELEMENTAL:               %[[VAL_22:.*]] = arith.andi %[[VAL_20]], %[[VAL_21]] : i1
// ELEMENTAL:               %[[VAL_23:.*]] = arith.ori %[[VAL_19]], %[[VAL_22]] : i1
// ELEMENTAL:               %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i1) -> !fir.logical<4>
// ELEMENTAL:               fir.result %[[VAL_24]] : !fir.logical<4>
// ELEMENTAL:             }
// ELEMENTAL:             hlfir.yield_element %[[VAL_14]] : !fir.logical<4>
// ELEMENTAL:           }
// ELEMENTAL:           return %[[VAL_10]] : !hlfir.expr<?x?x!fir.logical<4>>
// ELEMENTAL:         }

func.func @matmul_matrix_vector_real(%arg0: !hlfir.expr<?x?xf32>, %arg1: !hlfir.expr<?xf16>) -> !hlfir.expr<?xf32> {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xf32>, !hlfir.expr<?xf16>) -> !hlfir.expr<?xf32>
  return %res : !hlfir.expr<?xf32>
}
// ALL-LABEL:   func.func @matmul_matrix_vector_real(
// ALL-SAME:                                         %[[VAL_0:.*]]: !hlfir.expr<?x?xf32>,
// ALL-SAME:                                         %[[VAL_1:.*]]: !hlfir.expr<?xf16>) -> !hlfir.expr<?xf32> {

// NOANSE: hlfir.matmul

// ANSE:           %[[VAL_2:.*]] = arith.constant 0 : index
// ANSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// ANSE:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// ANSE:           %[[VAL_5:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf32>) -> !fir.shape<2>
// ANSE:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// ANSE:           %[[VAL_9:.*]] = hlfir.eval_in_mem shape %[[VAL_8]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// ANSE:           ^bb0(%[[VAL_10:.*]]: !fir.ref<!fir.array<?xf32>>):
// ANSE:             %[[VAL_11:.*]] = fir.embox %[[VAL_10]](%[[VAL_8]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// ANSE:             fir.do_loop %[[VAL_12:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] unordered {
// ANSE:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_11]], %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// ANSE:               %[[VAL_14:.*]] = arith.subi %[[VAL_13]]#0, %[[VAL_3]] : index
// ANSE:               %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_14]] : index
// ANSE:               %[[VAL_16:.*]] = hlfir.designate %[[VAL_11]] (%[[VAL_15]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// ANSE:               hlfir.assign %[[VAL_4]] to %[[VAL_16]] : f32, !fir.ref<f32>
// ANSE:             }
// ANSE:             fir.do_loop %[[VAL_17:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_3]] {
// ANSE:               fir.do_loop %[[VAL_18:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] {
// ANSE:                 %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_11]], %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_20:.*]] = arith.subi %[[VAL_19]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_21:.*]] = arith.addi %[[VAL_18]], %[[VAL_20]] : index
// ANSE:                 %[[VAL_22:.*]] = hlfir.designate %[[VAL_11]] (%[[VAL_21]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// ANSE:                 %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<f32>
// ANSE:                 %[[VAL_24:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_18]], %[[VAL_17]] : (!hlfir.expr<?x?xf32>, index, index) -> f32
// ANSE:                 %[[VAL_25:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_17]] : (!hlfir.expr<?xf16>, index) -> f16
// ANSE:                 %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (f16) -> f32
// ANSE:                 %[[VAL_27:.*]] = arith.mulf %[[VAL_24]], %[[VAL_26]] : f32
// ANSE:                 %[[VAL_28:.*]] = arith.addf %[[VAL_23]], %[[VAL_27]] : f32
// ANSE:                 hlfir.assign %[[VAL_28]] to %[[VAL_22]] : f32, !fir.ref<f32>
// ANSE:               }
// ANSE:             }
// ANSE:           }
// ANSE:           return %[[VAL_9]] : !hlfir.expr<?xf32>
// ANSE:         }

// ELEMENTAL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ELEMENTAL:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// ELEMENTAL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf32>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_7:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// ELEMENTAL:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// ELEMENTAL:           ^bb0(%[[VAL_9:.*]]: index):
// ELEMENTAL:             %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] iter_args(%[[VAL_12:.*]] = %[[VAL_3]]) -> (f32) {
// ELEMENTAL:               %[[VAL_13:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_9]], %[[VAL_11]] : (!hlfir.expr<?x?xf32>, index, index) -> f32
// ELEMENTAL:               %[[VAL_14:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_11]] : (!hlfir.expr<?xf16>, index) -> f16
// ELEMENTAL:               %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (f16) -> f32
// ELEMENTAL:               %[[VAL_16:.*]] = arith.mulf %[[VAL_13]], %[[VAL_15]] : f32
// ELEMENTAL:               %[[VAL_17:.*]] = arith.addf %[[VAL_12]], %[[VAL_16]] : f32
// ELEMENTAL:               fir.result %[[VAL_17]] : f32
// ELEMENTAL:             }
// ELEMENTAL:             hlfir.yield_element %[[VAL_10]] : f32
// ELEMENTAL:           }
// ELEMENTAL:           return %[[VAL_8]] : !hlfir.expr<?xf32>
// ELEMENTAL:         }

func.func @matmul_vector_matrix_real(%arg0: !hlfir.expr<?xf32>, %arg1: !hlfir.expr<?x?xf16>) -> !hlfir.expr<?xf32> {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?xf32>, !hlfir.expr<?x?xf16>) -> !hlfir.expr<?xf32>
  return %res : !hlfir.expr<?xf32>
}
// ALL-LABEL:   func.func @matmul_vector_matrix_real(
// ALL-SAME:                                         %[[VAL_0:.*]]: !hlfir.expr<?xf32>,
// ALL-SAME:                                         %[[VAL_1:.*]]: !hlfir.expr<?x?xf16>) -> !hlfir.expr<?xf32> {

// NOANSE: hlfir.matmul

// ANSE:           %[[VAL_2:.*]] = arith.constant 0 : index
// ANSE:           %[[VAL_3:.*]] = arith.constant 1 : index
// ANSE:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// ANSE:           %[[VAL_5:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xf32>) -> !fir.shape<1>
// ANSE:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_5]] {dim = 0 : index} : (!fir.shape<1>) -> index
// ANSE:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xf16>) -> !fir.shape<2>
// ANSE:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ANSE:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
// ANSE:           %[[VAL_10:.*]] = hlfir.eval_in_mem shape %[[VAL_9]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// ANSE:           ^bb0(%[[VAL_11:.*]]: !fir.ref<!fir.array<?xf32>>):
// ANSE:             %[[VAL_12:.*]] = fir.embox %[[VAL_11]](%[[VAL_9]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// ANSE:             fir.do_loop %[[VAL_13:.*]] = %[[VAL_3]] to %[[VAL_8]] step %[[VAL_3]] unordered {
// ANSE:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_12]], %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// ANSE:               %[[VAL_15:.*]] = arith.subi %[[VAL_14]]#0, %[[VAL_3]] : index
// ANSE:               %[[VAL_16:.*]] = arith.addi %[[VAL_13]], %[[VAL_15]] : index
// ANSE:               %[[VAL_17:.*]] = hlfir.designate %[[VAL_12]] (%[[VAL_16]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// ANSE:               hlfir.assign %[[VAL_4]] to %[[VAL_17]] : f32, !fir.ref<f32>
// ANSE:             }
// ANSE:             fir.do_loop %[[VAL_18:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_3]] {
// ANSE:               fir.do_loop %[[VAL_19:.*]] = %[[VAL_3]] to %[[VAL_8]] step %[[VAL_3]] {
// ANSE:                 %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_12]], %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// ANSE:                 %[[VAL_21:.*]] = arith.subi %[[VAL_20]]#0, %[[VAL_3]] : index
// ANSE:                 %[[VAL_22:.*]] = arith.addi %[[VAL_19]], %[[VAL_21]] : index
// ANSE:                 %[[VAL_23:.*]] = hlfir.designate %[[VAL_12]] (%[[VAL_22]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// ANSE:                 %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<f32>
// ANSE:                 %[[VAL_25:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_18]] : (!hlfir.expr<?xf32>, index) -> f32
// ANSE:                 %[[VAL_26:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_18]], %[[VAL_19]] : (!hlfir.expr<?x?xf16>, index, index) -> f16
// ANSE:                 %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (f16) -> f32
// ANSE:                 %[[VAL_28:.*]] = arith.mulf %[[VAL_25]], %[[VAL_27]] : f32
// ANSE:                 %[[VAL_29:.*]] = arith.addf %[[VAL_24]], %[[VAL_28]] : f32
// ANSE:                 hlfir.assign %[[VAL_29]] to %[[VAL_23]] : f32, !fir.ref<f32>
// ANSE:               }
// ANSE:             }
// ANSE:           }
// ANSE:           return %[[VAL_10]] : !hlfir.expr<?xf32>
// ANSE:         }

// ELEMENTAL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ELEMENTAL:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// ELEMENTAL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xf32>) -> !fir.shape<1>
// ELEMENTAL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<1>) -> index
// ELEMENTAL:           %[[VAL_6:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xf16>) -> !fir.shape<2>
// ELEMENTAL:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_6]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ELEMENTAL:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
// ELEMENTAL:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// ELEMENTAL:           ^bb0(%[[VAL_10:.*]]: index):
// ELEMENTAL:             %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_13:.*]] = %[[VAL_3]]) -> (f32) {
// ELEMENTAL:               %[[VAL_14:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_12]] : (!hlfir.expr<?xf32>, index) -> f32
// ELEMENTAL:               %[[VAL_15:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_12]], %[[VAL_10]] : (!hlfir.expr<?x?xf16>, index, index) -> f16
// ELEMENTAL:               %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (f16) -> f32
// ELEMENTAL:               %[[VAL_17:.*]] = arith.mulf %[[VAL_14]], %[[VAL_16]] : f32
// ELEMENTAL:               %[[VAL_18:.*]] = arith.addf %[[VAL_13]], %[[VAL_17]] : f32
// ELEMENTAL:               fir.result %[[VAL_18]] : f32
// ELEMENTAL:             }
// ELEMENTAL:             hlfir.yield_element %[[VAL_11]] : f32
// ELEMENTAL:           }
// ELEMENTAL:           return %[[VAL_9]] : !hlfir.expr<?xf32>
// ELEMENTAL:         }

func.func @matmul_transpose_matrix_matrix_integer(%arg0: !hlfir.expr<?x?xi16>, %arg1: !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32> {
  %res = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<?x?xi16>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return %res : !hlfir.expr<?x?xi32>
}
// ALL-LABEL:   func.func @matmul_transpose_matrix_matrix_integer(
// ALL-SAME:                                                      %[[VAL_0:.*]]: !hlfir.expr<?x?xi16>,
// ALL-SAME:                                                      %[[VAL_1:.*]]: !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32> {
// ALL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ALL:           %[[VAL_3:.*]] = arith.constant 0 : i32
// ALL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xi16>) -> !fir.shape<2>
// ALL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ALL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ALL:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
// ALL:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ALL:           %[[VAL_9:.*]] = fir.shape %[[VAL_6]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
// ALL:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// ALL:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
// ALL:             %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_3]]) -> (i32) {
// ALL:               %[[VAL_16:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_14]], %[[VAL_11]] : (!hlfir.expr<?x?xi16>, index, index) -> i16
// ALL:               %[[VAL_17:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_14]], %[[VAL_12]] : (!hlfir.expr<?x?xi32>, index, index) -> i32
// ALL:               %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (i16) -> i32
// ALL:               %[[VAL_19:.*]] = arith.muli %[[VAL_18]], %[[VAL_17]] : i32
// ALL:               %[[VAL_20:.*]] = arith.addi %[[VAL_15]], %[[VAL_19]] : i32
// ALL:               fir.result %[[VAL_20]] : i32
// ALL:             }
// ALL:             hlfir.yield_element %[[VAL_13]] : i32
// ALL:           }
// ALL:           return %[[VAL_10]] : !hlfir.expr<?x?xi32>
// ALL:         }

func.func @matmul_transpose_matrix_vector_real(%arg0: !hlfir.expr<?x?xf32>, %arg1: !hlfir.expr<?xf16>) -> !hlfir.expr<?xf32> {
  %res = hlfir.matmul_transpose %arg0 %arg1 : (!hlfir.expr<?x?xf32>, !hlfir.expr<?xf16>) -> !hlfir.expr<?xf32>
  return %res : !hlfir.expr<?xf32>
}
// ALL-LABEL:   func.func @matmul_transpose_matrix_vector_real(
// ALL-SAME:                                                   %[[VAL_0:.*]]: !hlfir.expr<?x?xf32>,
// ALL-SAME:                                                   %[[VAL_1:.*]]: !hlfir.expr<?xf16>) -> !hlfir.expr<?xf32> {
// ALL:           %[[VAL_2:.*]] = arith.constant 1 : index
// ALL:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// ALL:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf32>) -> !fir.shape<2>
// ALL:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// ALL:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// ALL:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// ALL:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// ALL:           ^bb0(%[[VAL_9:.*]]: index):
// ALL:             %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_12:.*]] = %[[VAL_3]]) -> (f32) {
// ALL:               %[[VAL_13:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_9]] : (!hlfir.expr<?x?xf32>, index, index) -> f32
// ALL:               %[[VAL_14:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_11]] : (!hlfir.expr<?xf16>, index) -> f16
// ALL:               %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (f16) -> f32
// ALL:               %[[VAL_16:.*]] = arith.mulf %[[VAL_13]], %[[VAL_15]] : f32
// ALL:               %[[VAL_17:.*]] = arith.addf %[[VAL_12]], %[[VAL_16]] : f32
// ALL:               fir.result %[[VAL_17]] : f32
// ALL:             }
// ALL:             hlfir.yield_element %[[VAL_10]] : f32
// ALL:           }
// ALL:           return %[[VAL_8]] : !hlfir.expr<?xf32>
// ALL:         }

// Check that the inner-product loop uses the best known extent
// of the input matrices:
func.func @matmul_matrix_matrix_deduce_bounds(%arg0: !hlfir.expr<?x10xi16>, %arg1: !hlfir.expr<?x?xi32>, %arg2: !hlfir.expr<10x?xi16>) -> (!hlfir.expr<?x?xi32>, !hlfir.expr<?x?xi32>) {
  %res1 = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x10xi16>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  %res2 = hlfir.matmul %arg1 %arg2 : (!hlfir.expr<?x?xi32>, !hlfir.expr<10x?xi16>) -> !hlfir.expr<?x?xi32>
  return %res1, %res2 : !hlfir.expr<?x?xi32>, !hlfir.expr<?x?xi32>
}
// ALL-LABEL:   func.func @matmul_matrix_matrix_deduce_bounds(

// ANSE:           %[[VAL_6:.*]] = arith.constant 10 : index
// ANSE:           hlfir.eval_in_mem shape {{.*}}
// ANSE:             fir.do_loop
// ANSE:               fir.do_loop
// ANSE:             fir.do_loop %{{.*}} = %{{.*}} to %[[VAL_6]]
// ANSE:               fir.do_loop
// ANSE:                 fir.do_loop
// ANSE:           hlfir.eval_in_mem shape {{.*}}
// ANSE:             fir.do_loop
// ANSE:               fir.do_loop
// ANSE:             fir.do_loop %{{.*}} = %{{.*}} to %[[VAL_6]]
// ANSE:               fir.do_loop
// ANSE:                 fir.do_loop

// ELEMENTAL:           %[[VAL_5:.*]] = arith.constant 10 : index
// ELEMENTAL:           hlfir.elemental %{{.*}}
// ELEMENTAL:             fir.do_loop %{{.*}} = %{{.*}} to %[[VAL_5]]
// ELEMENTAL:           hlfir.elemental %{{.*}}
// ELEMENTAL:             fir.do_loop %{{.*}} = %{{.*}} to %[[VAL_5]]
