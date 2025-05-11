// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

func.func @test_total_expr(%input: !hlfir.expr<?x?xf32>, %mask: !hlfir.expr<?x?x!fir.logical<4>>) -> f32 {
  %0 = hlfir.maxval %input mask %mask {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?x?xf32>, !hlfir.expr<?x?x!fir.logical<4>>) -> f32
  return %0 : f32
}
// CHECK-LABEL:   func.func @test_total_expr(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?xf32>,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x!fir.logical<4>>) -> f32 {
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant -3.40282347E+38 : f32
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf32>) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_7:.*]]:2 = fir.do_loop %[[VAL_8:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] iter_args(%[[VAL_9:.*]] = %[[VAL_3]], %[[FIRST1:.*]] = %[[TRUE]]) -> (f32, i1) {
// CHECK:             %[[VAL_10:.*]]:2 = fir.do_loop %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]], %[[FIRST2:.*]] = %[[FIRST1]]) -> (f32, i1) {
// CHECK:               %[[VAL_13:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_11]], %[[VAL_8]] : (!hlfir.expr<?x?x!fir.logical<4>>, index, index) -> !fir.logical<4>
// CHECK:               %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.logical<4>) -> i1
// CHECK:               %[[VAL_15:.*]]:2 = fir.if %[[VAL_14]] -> (f32, i1) {
// CHECK:                 %[[VAL_16:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_8]] : (!hlfir.expr<?x?xf32>, index, index) -> f32
// CHECK:                 %[[VAL_17:.*]] = arith.cmpf ogt, %[[VAL_16]], %[[VAL_12]] fastmath<contract> : f32
// CHECK:                 %[[VAL_18:.*]] = arith.cmpf une, %[[VAL_12]], %[[VAL_12]] fastmath<contract> : f32
// CHECK:                 %[[VAL_19:.*]] = arith.cmpf oeq, %[[VAL_16]], %[[VAL_16]] fastmath<contract> : f32
// CHECK:                 %[[VAL_20:.*]] = arith.andi %[[VAL_18]], %[[VAL_19]] : i1
// CHECK:                 %[[VAL_21:.*]] = arith.ori %[[VAL_17]], %[[VAL_20]] : i1
// CHECK:                 %[[IS_FIRST:.*]] = arith.ori %[[VAL_21]], %[[FIRST2]] : i1
// CHECK:                 %[[VAL_22:.*]] = arith.select %[[IS_FIRST]], %[[VAL_16]], %[[VAL_12]] : f32
// CHECK:                 fir.result %[[VAL_22]], %[[FALSE]] : f32, i1
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_12]], %[[FIRST2]] : f32, i1
// CHECK:               }
// CHECK:               fir.result %[[VAL_15]]#0, %[[VAL_15]]#1 : f32, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1 : f32, i1
// CHECK:           }
// CHECK:           return %[[VAL_7]]#0 : f32
// CHECK:         }

func.func @test_partial_expr(%input: !hlfir.expr<?x?xf64>, %mask: !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?xf64> {
  %dim = arith.constant 1 : i32
  %0 = hlfir.maxval %input dim %dim mask %mask {fastmath = #arith.fastmath<reassoc>} : (!hlfir.expr<?x?xf64>, i32, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?xf64>
  return %0 : !hlfir.expr<?xf64>
}
// CHECK-LABEL:   func.func @test_partial_expr(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?xf64>,
// CHECK-SAME:                                 %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?xf64> {
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant -1.7976931348623157E+308 : f64
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf64>) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
// CHECK:           ^bb0(%[[VAL_9:.*]]: index):
// CHECK:             %[[VAL_10:.*]]:2 = fir.do_loop %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_3]], %[[FIRST:.*]] = %[[TRUE]]) -> (f64, i1) {
// CHECK:               %[[VAL_13:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_11]], %[[VAL_9]] : (!hlfir.expr<?x?x!fir.logical<4>>, index, index) -> !fir.logical<4>
// CHECK:               %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.logical<4>) -> i1
// CHECK:               %[[VAL_15:.*]]:2 = fir.if %[[VAL_14]] -> (f64, i1) {
// CHECK:                 %[[VAL_16:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_9]] : (!hlfir.expr<?x?xf64>, index, index) -> f64
// CHECK:                 %[[VAL_17:.*]] = arith.cmpf ogt, %[[VAL_16]], %[[VAL_12]] fastmath<reassoc> : f64
// CHECK:                 %[[VAL_18:.*]] = arith.cmpf une, %[[VAL_12]], %[[VAL_12]] fastmath<reassoc> : f64
// CHECK:                 %[[VAL_19:.*]] = arith.cmpf oeq, %[[VAL_16]], %[[VAL_16]] fastmath<reassoc> : f64
// CHECK:                 %[[VAL_20:.*]] = arith.andi %[[VAL_18]], %[[VAL_19]] : i1
// CHECK:                 %[[VAL_21:.*]] = arith.ori %[[VAL_17]], %[[VAL_20]] : i1
// CHECK:                 %[[IS_FIRST:.*]] = arith.ori %[[VAL_21]], %[[FIRST]] : i1
// CHECK:                 %[[VAL_22:.*]] = arith.select %[[IS_FIRST]], %[[VAL_16]], %[[VAL_12]] : f64
// CHECK:                 fir.result %[[VAL_22]], %[[FALSE]] : f64, i1
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_12]], %[[FIRST]] : f64, i1
// CHECK:               }
// CHECK:               fir.result %[[VAL_15]]#0, %[[VAL_15]]#1 : f64, i1
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_10]]#0 : f64
// CHECK:           }
// CHECK:           return %[[VAL_8]] : !hlfir.expr<?xf64>
// CHECK:         }

func.func @test_total_var(%input: !fir.box<!fir.array<?x?xf16>>, %mask: !fir.ref<!fir.array<2x2x!fir.logical<1>>>) -> f16 {
  %0 = hlfir.maxval %input mask %mask {fastmath = #arith.fastmath<reassoc>} : (!fir.box<!fir.array<?x?xf16>>, !fir.ref<!fir.array<2x2x!fir.logical<1>>>) -> f16
  return %0 : f16
}
// CHECK-LABEL:   func.func @test_total_var(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf16>>,
// CHECK-SAME:                              %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.array<2x2x!fir.logical<1>>>) -> f16 {
// CHECK:           %[[VAL_2:.*]] = arith.constant -6.550400e+04 : f16
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]]:2 = fir.do_loop %[[VAL_8:.*]] = %[[VAL_3]] to %[[VAL_6]]#1 step %[[VAL_3]] unordered iter_args(%[[VAL_9:.*]] = %[[VAL_2]], %[[FIRST1:.*]] = %[[TRUE]]) -> (f16, i1) {
// CHECK:             %[[VAL_10:.*]]:2 = fir.do_loop %[[VAL_11:.*]] = %[[VAL_3]] to %[[VAL_5]]#1 step %[[VAL_3]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_9]], %[[FIRST2:.*]] = %[[FIRST1]]) -> (f16, i1) {
// CHECK:               %[[VAL_13:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_11]], %[[VAL_8]])  : (!fir.ref<!fir.array<2x2x!fir.logical<1>>>, index, index) -> !fir.ref<!fir.logical<1>>
// CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<!fir.logical<1>>
// CHECK:               %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_16:.*]]:2 = fir.if %[[VAL_15]] -> (f16, i1) {
// CHECK:                 %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_19:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_3]] : index
// CHECK:                 %[[VAL_20:.*]] = arith.addi %[[VAL_11]], %[[VAL_19]] : index
// CHECK:                 %[[VAL_21:.*]] = arith.subi %[[VAL_18]]#0, %[[VAL_3]] : index
// CHECK:                 %[[VAL_22:.*]] = arith.addi %[[VAL_8]], %[[VAL_21]] : index
// CHECK:                 %[[VAL_23:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_20]], %[[VAL_22]])  : (!fir.box<!fir.array<?x?xf16>>, index, index) -> !fir.ref<f16>
// CHECK:                 %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<f16>
// CHECK:                 %[[VAL_25:.*]] = arith.cmpf ogt, %[[VAL_24]], %[[VAL_12]] fastmath<reassoc> : f16
// CHECK:                 %[[VAL_26:.*]] = arith.cmpf une, %[[VAL_12]], %[[VAL_12]] fastmath<reassoc> : f16
// CHECK:                 %[[VAL_27:.*]] = arith.cmpf oeq, %[[VAL_24]], %[[VAL_24]] fastmath<reassoc> : f16
// CHECK:                 %[[VAL_28:.*]] = arith.andi %[[VAL_26]], %[[VAL_27]] : i1
// CHECK:                 %[[VAL_29:.*]] = arith.ori %[[VAL_25]], %[[VAL_28]] : i1
// CHECK:                 %[[IS_FIRST:.*]] = arith.ori %[[VAL_29]], %[[FIRST2]] : i1
// CHECK:                 %[[VAL_30:.*]] = arith.select %[[IS_FIRST]], %[[VAL_24]], %[[VAL_12]] : f16
// CHECK:                 fir.result %[[VAL_30]], %[[FALSE]] : f16, i1
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_12]], %[[FIRST2]] : f16, i1
// CHECK:               }
// CHECK:               fir.result %[[VAL_16]]#0, %[[VAL_16]]#1 : f16, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]]#0, %[[VAL_10]]#1 : f16, i1
// CHECK:           }
// CHECK:           return %[[VAL_7]]#0 : f16
// CHECK:         }

func.func @test_partial_var(%input: !fir.box<!fir.array<?x?xf16>>, %mask: !fir.box<!fir.array<2x2x!fir.logical<1>>>) -> !hlfir.expr<?xf16> {
  %dim = arith.constant 2 : i32
  %0 = hlfir.maxval %input dim %dim mask %mask {fastmath = #arith.fastmath<reassoc>} : (!fir.box<!fir.array<?x?xf16>>, i32, !fir.box<!fir.array<2x2x!fir.logical<1>>>) -> !hlfir.expr<?xf16>
  return %0 : !hlfir.expr<?xf16>
}
// CHECK-LABEL:   func.func @test_partial_var(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf16>>,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<2x2x!fir.logical<1>>>) -> !hlfir.expr<?xf16> {
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant -6.550400e+04 : f16
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<2x2x!fir.logical<1>>>) -> i1
// CHECK:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf16> {
// CHECK:           ^bb0(%[[VAL_11:.*]]: index):
// CHECK:             %[[VAL_12:.*]]:2 = fir.do_loop %[[VAL_13:.*]] = %[[VAL_4]] to %[[VAL_7]]#1 step %[[VAL_4]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_3]], %[[FIRST:.*]] = %[[TRUE]]) -> (f16, i1) {
// CHECK:               %[[VAL_15:.*]] = fir.if %[[VAL_9]] -> (!fir.logical<1>) {
// CHECK:                 %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_5]] : (!fir.box<!fir.array<2x2x!fir.logical<1>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<2x2x!fir.logical<1>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_18:.*]] = arith.subi %[[VAL_16]]#0, %[[VAL_4]] : index
// CHECK:                 %[[VAL_19:.*]] = arith.addi %[[VAL_11]], %[[VAL_18]] : index
// CHECK:                 %[[VAL_20:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_4]] : index
// CHECK:                 %[[VAL_21:.*]] = arith.addi %[[VAL_13]], %[[VAL_20]] : index
// CHECK:                 %[[VAL_22:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_19]], %[[VAL_21]])  : (!fir.box<!fir.array<2x2x!fir.logical<1>>>, index, index) -> !fir.ref<!fir.logical<1>>
// CHECK:                 %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.logical<1>>
// CHECK:                 fir.result %[[VAL_23]] : !fir.logical<1>
// CHECK:               } else {
// CHECK:                 %[[VAL_24:.*]] = fir.convert %[[VAL_2]] : (i1) -> !fir.logical<1>
// CHECK:                 fir.result %[[VAL_24]] : !fir.logical<1>
// CHECK:               }
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_15]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_26:.*]]:2 = fir.if %[[VAL_25]] -> (f16, i1) {
// CHECK:                 %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_28:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_29:.*]] = arith.subi %[[VAL_27]]#0, %[[VAL_4]] : index
// CHECK:                 %[[VAL_30:.*]] = arith.addi %[[VAL_11]], %[[VAL_29]] : index
// CHECK:                 %[[VAL_31:.*]] = arith.subi %[[VAL_28]]#0, %[[VAL_4]] : index
// CHECK:                 %[[VAL_32:.*]] = arith.addi %[[VAL_13]], %[[VAL_31]] : index
// CHECK:                 %[[VAL_33:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_30]], %[[VAL_32]])  : (!fir.box<!fir.array<?x?xf16>>, index, index) -> !fir.ref<f16>
// CHECK:                 %[[VAL_34:.*]] = fir.load %[[VAL_33]] : !fir.ref<f16>
// CHECK:                 %[[VAL_35:.*]] = arith.cmpf ogt, %[[VAL_34]], %[[VAL_14]] fastmath<reassoc> : f16
// CHECK:                 %[[VAL_36:.*]] = arith.cmpf une, %[[VAL_14]], %[[VAL_14]] fastmath<reassoc> : f16
// CHECK:                 %[[VAL_37:.*]] = arith.cmpf oeq, %[[VAL_34]], %[[VAL_34]] fastmath<reassoc> : f16
// CHECK:                 %[[VAL_38:.*]] = arith.andi %[[VAL_36]], %[[VAL_37]] : i1
// CHECK:                 %[[VAL_39:.*]] = arith.ori %[[VAL_35]], %[[VAL_38]] : i1
// CHECK:                 %[[IS_FIRST:.*]] = arith.ori %[[VAL_39]], %[[FIRST]] : i1
// CHECK:                 %[[VAL_40:.*]] = arith.select %[[IS_FIRST]], %[[VAL_34]], %[[VAL_14]] : f16
// CHECK:                 fir.result %[[VAL_40]], %[[FALSE]] : f16, i1
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_14]], %[[FIRST]] : f16, i1
// CHECK:               }
// CHECK:               fir.result %[[VAL_26]]#0, %[[VAL_26]]#1 : f16, i1
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_12]]#0 : f16
// CHECK:           }
// CHECK:           return %[[VAL_10]] : !hlfir.expr<?xf16>
// CHECK:         }

func.func @test_partial_expr_nomask(%input: !hlfir.expr<?x?xf64>) -> !hlfir.expr<?xf64> {
  %dim = arith.constant 1 : i32
  %0 = hlfir.maxval %input dim %dim {fastmath = #arith.fastmath<reassoc>} : (!hlfir.expr<?x?xf64>, i32) -> !hlfir.expr<?xf64>
  return %0 : !hlfir.expr<?xf64>
}
// CHECK-LABEL:   func.func @test_partial_expr_nomask(
// CHECK-SAME:                                        %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?xf64>) -> !hlfir.expr<?xf64> {
// CHECK:           %[[VAL_1:.*]] = arith.constant -1.7976931348623157E+308 : f64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf64>) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
// CHECK:           ^bb0(%[[VAL_9:.*]]: index):
// CHECK:             %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_3]] : index
// CHECK:             %[[VAL_11:.*]] = fir.if %[[VAL_10]] -> (f64) {
// CHECK:               %[[VAL_12:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_2]], %[[VAL_9]] : (!hlfir.expr<?x?xf64>, index, index) -> f64
// CHECK:               fir.result %[[VAL_12]] : f64
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_1]] : f64
// CHECK:             }
// CHECK:             %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_11]]) -> (f64) {
// CHECK:               %[[VAL_16:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_14]], %[[VAL_9]] : (!hlfir.expr<?x?xf64>, index, index) -> f64
// CHECK:               %[[VAL_17:.*]] = arith.cmpf ogt, %[[VAL_16]], %[[VAL_15]] fastmath<reassoc> : f64
// CHECK:               %[[VAL_18:.*]] = arith.cmpf une, %[[VAL_15]], %[[VAL_15]] fastmath<reassoc> : f64
// CHECK:               %[[VAL_19:.*]] = arith.cmpf oeq, %[[VAL_16]], %[[VAL_16]] fastmath<reassoc> : f64
// CHECK:               %[[VAL_20:.*]] = arith.andi %[[VAL_18]], %[[VAL_19]] : i1
// CHECK:               %[[VAL_21:.*]] = arith.ori %[[VAL_17]], %[[VAL_20]] : i1
// CHECK:               %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_16]], %[[VAL_15]] : f64
// CHECK:               fir.result %[[VAL_22]] : f64
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_13]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_8]] : !hlfir.expr<?xf64>
// CHECK:         }

func.func @test_total_var_nomask(%input: !fir.box<!fir.array<?x?xf16>>) -> f16 {
  %0 = hlfir.maxval %input {fastmath = #arith.fastmath<reassoc>} : (!fir.box<!fir.array<?x?xf16>>) -> f16
  return %0 : f16
}
// CHECK-LABEL:   func.func @test_total_var_nomask(
// CHECK-SAME:                                     %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf16>>) -> f16 {
// CHECK:           %[[VAL_1:.*]] = arith.constant -6.550400e+04 : f16
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]]#1, %[[VAL_3]] : index
// CHECK:           %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]]#1, %[[VAL_3]] : index
// CHECK:           %[[VAL_8:.*]] = arith.andi %[[VAL_6]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (f16) {
// CHECK:             %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:             %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_10]]#0, %[[VAL_11]]#0)  : (!fir.box<!fir.array<?x?xf16>>, index, index) -> !fir.ref<f16>
// CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<f16>
// CHECK:             fir.result %[[VAL_13]] : f16
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_1]] : f16
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_2]] to %[[VAL_5]]#1 step %[[VAL_2]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_9]]) -> (f16) {
// CHECK:             %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_2]] to %[[VAL_4]]#1 step %[[VAL_2]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_16]]) -> (f16) {
// CHECK:               %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:               %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?xf16>>, index) -> (index, index, index)
// CHECK:               %[[VAL_22:.*]] = arith.subi %[[VAL_20]]#0, %[[VAL_2]] : index
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_18]], %[[VAL_22]] : index
// CHECK:               %[[VAL_24:.*]] = arith.subi %[[VAL_21]]#0, %[[VAL_2]] : index
// CHECK:               %[[VAL_25:.*]] = arith.addi %[[VAL_15]], %[[VAL_24]] : index
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_23]], %[[VAL_25]])  : (!fir.box<!fir.array<?x?xf16>>, index, index) -> !fir.ref<f16>
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<f16>
// CHECK:               %[[VAL_28:.*]] = arith.cmpf ogt, %[[VAL_27]], %[[VAL_19]] fastmath<reassoc> : f16
// CHECK:               %[[VAL_29:.*]] = arith.cmpf une, %[[VAL_19]], %[[VAL_19]] fastmath<reassoc> : f16
// CHECK:               %[[VAL_30:.*]] = arith.cmpf oeq, %[[VAL_27]], %[[VAL_27]] fastmath<reassoc> : f16
// CHECK:               %[[VAL_31:.*]] = arith.andi %[[VAL_29]], %[[VAL_30]] : i1
// CHECK:               %[[VAL_32:.*]] = arith.ori %[[VAL_28]], %[[VAL_31]] : i1
// CHECK:               %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_27]], %[[VAL_19]] : f16
// CHECK:               fir.result %[[VAL_33]] : f16
// CHECK:             }
// CHECK:             fir.result %[[VAL_17]] : f16
// CHECK:           }
// CHECK:           return %[[VAL_14]] : f16
// CHECK:         }

// Test that 'nnan' allows using -LARGEST value as the reduction init.
func.func @test_partial_expr_nnan(%input: !hlfir.expr<?x?xf64>) -> !hlfir.expr<?xf64> {
  %dim = arith.constant 1 : i32
  %0 = hlfir.maxval %input dim %dim {fastmath = #arith.fastmath<nnan>} : (!hlfir.expr<?x?xf64>, i32) -> !hlfir.expr<?xf64>
  return %0 : !hlfir.expr<?xf64>
}
// CHECK-LABEL:   func.func @test_partial_expr_nnan(
// CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?xf64>) -> !hlfir.expr<?xf64> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant -1.7976931348623157E+308 : f64
// CHECK:           %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xf64>) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_1]] iter_args(%[[VAL_11:.*]] = %[[VAL_2]]) -> (f64) {
// CHECK:               %[[VAL_12:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]], %[[VAL_8]] : (!hlfir.expr<?x?xf64>, index, index) -> f64
// CHECK:               %[[VAL_13:.*]] = arith.cmpf ogt, %[[VAL_12]], %[[VAL_11]] fastmath<nnan> : f64
// CHECK:               %[[VAL_14:.*]] = arith.cmpf une, %[[VAL_11]], %[[VAL_11]] fastmath<nnan> : f64
// CHECK:               %[[VAL_15:.*]] = arith.cmpf oeq, %[[VAL_12]], %[[VAL_12]] fastmath<nnan> : f64
// CHECK:               %[[VAL_16:.*]] = arith.andi %[[VAL_14]], %[[VAL_15]] : i1
// CHECK:               %[[VAL_17:.*]] = arith.ori %[[VAL_13]], %[[VAL_16]] : i1
// CHECK:               %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_12]], %[[VAL_11]] : f64
// CHECK:               fir.result %[[VAL_18]] : f64
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_9]] : f64
// CHECK:           }
// CHECK:           return %[[VAL_7]] : !hlfir.expr<?xf64>
// CHECK:         }
