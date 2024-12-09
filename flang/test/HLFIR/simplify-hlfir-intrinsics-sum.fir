// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

// box with known extents
func.func @sum_box_known_extents(%arg0: !fir.box<!fir.array<2x3xi32>>) {
  %cst = arith.constant 2 : i32
  %res = hlfir.sum %arg0 dim %cst : (!fir.box<!fir.array<2x3xi32>>, i32) -> !hlfir.expr<2xi32>
  return
}
// CHECK-LABEL:   func.func @sum_box_known_extents(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.box<!fir.array<2x3xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<2xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_3]] step %[[VAL_7]] unordered iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (i32) {
// CHECK:               %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_12]] : (!fir.box<!fir.array<2x3xi32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_14]] : (!fir.box<!fir.array<2x3xi32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_17:.*]] = arith.subi %[[VAL_13]]#0, %[[VAL_16]] : index
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_6]], %[[VAL_17]] : index
// CHECK:               %[[VAL_19:.*]] = arith.subi %[[VAL_15]]#0, %[[VAL_16]] : index
// CHECK:               %[[VAL_20:.*]] = arith.addi %[[VAL_10]], %[[VAL_19]] : index
// CHECK:               %[[VAL_21:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_18]], %[[VAL_20]])  : (!fir.box<!fir.array<2x3xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_11]], %[[VAL_22]] : i32
// CHECK:               fir.result %[[VAL_23]] : i32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// expr with known extents
func.func @sum_expr_known_extents(%arg0: !hlfir.expr<2x3xi32>) {
  %cst = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %cst : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
  return
}
// CHECK-LABEL:   func.func @sum_expr_known_extents(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !hlfir.expr<2x3xi32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_2]] step %[[VAL_7]] unordered iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (i32) {
// CHECK:               %[[VAL_12:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]], %[[VAL_6]] : (!hlfir.expr<2x3xi32>, index, index) -> i32
// CHECK:               %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:               fir.result %[[VAL_13]] : i32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// box with unknown extent
func.func @sum_box_unknown_extent1(%arg0: !fir.box<!fir.array<?x3xcomplex<f64>>>) {
  %cst = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %cst : (!fir.box<!fir.array<?x3xcomplex<f64>>>, i32) -> !hlfir.expr<3xcomplex<f64>>
  return
}
// CHECK-LABEL:   func.func @sum_box_unknown_extent1(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.box<!fir.array<?x3xcomplex<f64>>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xcomplex<f64>> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:             %[[VAL_10:.*]] = fir.undefined complex<f64>
// CHECK:             %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_9]], [0 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_9]], [1 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_8]] to %[[VAL_3]]#1 step %[[VAL_8]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (complex<f64>) {
// CHECK:               %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_16]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_18]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_20:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_21:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_20]] : index
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_14]], %[[VAL_21]] : index
// CHECK:               %[[VAL_23:.*]] = arith.subi %[[VAL_19]]#0, %[[VAL_20]] : index
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_7]], %[[VAL_23]] : index
// CHECK:               %[[VAL_25:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_22]], %[[VAL_24]])  : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index, index) -> !fir.ref<complex<f64>>
// CHECK:               %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<complex<f64>>
// CHECK:               %[[VAL_27:.*]] = fir.addc %[[VAL_15]], %[[VAL_26]] : complex<f64>
// CHECK:               fir.result %[[VAL_27]] : complex<f64>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_13]] : complex<f64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @sum_box_unknown_extent2(%arg0: !fir.box<!fir.array<?x3xcomplex<f64>>>) {
  %cst = arith.constant 2 : i32
  %res = hlfir.sum %arg0 dim %cst : (!fir.box<!fir.array<?x3xcomplex<f64>>>, i32) -> !hlfir.expr<?xcomplex<f64>>
  return
}
// CHECK-LABEL:   func.func @sum_box_unknown_extent2(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.box<!fir.array<?x3xcomplex<f64>>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xcomplex<f64>> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:             %[[VAL_10:.*]] = fir.undefined complex<f64>
// CHECK:             %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_9]], [0 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_9]], [1 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_8]] to %[[VAL_4]] step %[[VAL_8]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (complex<f64>) {
// CHECK:               %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_16]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_18]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_20:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_21:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_20]] : index
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_7]], %[[VAL_21]] : index
// CHECK:               %[[VAL_23:.*]] = arith.subi %[[VAL_19]]#0, %[[VAL_20]] : index
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_14]], %[[VAL_23]] : index
// CHECK:               %[[VAL_25:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_22]], %[[VAL_24]])  : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index, index) -> !fir.ref<complex<f64>>
// CHECK:               %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<complex<f64>>
// CHECK:               %[[VAL_27:.*]] = fir.addc %[[VAL_15]], %[[VAL_26]] : complex<f64>
// CHECK:               fir.result %[[VAL_27]] : complex<f64>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_13]] : complex<f64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// expr with unknown extent
func.func @sum_expr_unknown_extent1(%arg0: !hlfir.expr<?x3xf32>) {
  %cst = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %cst : (!hlfir.expr<?x3xf32>, i32) -> !hlfir.expr<3xf32>
  return
}
// CHECK-LABEL:   func.func @sum_expr_unknown_extent1(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !hlfir.expr<?x3xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[VAL_3:.*]] = hlfir.get_extent %[[VAL_2]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_3]] step %[[VAL_8]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (f32) {
// CHECK:               %[[VAL_13:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_7]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:               %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:               fir.result %[[VAL_14]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_10]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @sum_expr_unknown_extent2(%arg0: !hlfir.expr<?x3xf32>) {
  %cst = arith.constant 2 : i32
  %res = hlfir.sum %arg0 dim %cst : (!hlfir.expr<?x3xf32>, i32) -> !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @sum_expr_unknown_extent2(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !hlfir.expr<?x3xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[VAL_3:.*]] = hlfir.get_extent %[[VAL_2]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_4]] step %[[VAL_8]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (f32) {
// CHECK:               %[[VAL_13:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_7]], %[[VAL_11]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:               %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:               fir.result %[[VAL_14]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_10]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// scalar mask
func.func @sum_scalar_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !fir.ref<!fir.logical<1>>) {
  %cst = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !fir.ref<!fir.logical<1>>) -> !hlfir.expr<3xf32>
  return
}
// CHECK-LABEL:   func.func @sum_scalar_mask(
// CHECK-SAME:                               %[[VAL_0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:                               %[[VAL_1:.*]]: !fir.ref<!fir.logical<1>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<1>>
// CHECK:             %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_9]] to %[[VAL_4]] step %[[VAL_9]] iter_args(%[[VAL_14:.*]] = %[[VAL_10]]) -> (f32) {
// CHECK:               %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_16:.*]] = fir.if %[[VAL_15]] -> (f32) {
// CHECK:                 %[[VAL_17:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_13]], %[[VAL_8]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[VAL_18:.*]] = arith.addf %[[VAL_14]], %[[VAL_17]] : f32
// CHECK:                 fir.result %[[VAL_18]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_14]] : f32
// CHECK:               }
// CHECK:               fir.result %[[VAL_16]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_12]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// scalar boxed mask
func.func @sum_scalar_boxed_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !fir.box<!fir.logical<1>>) {
  %cst = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !fir.box<!fir.logical<1>>) -> !hlfir.expr<3xf32>
  return
}
// CHECK-LABEL:   func.func @sum_scalar_boxed_mask(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !fir.box<!fir.logical<1>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_11:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.logical<1>>) -> i1
// CHECK:             %[[VAL_12:.*]] = fir.if %[[VAL_11]] -> (!fir.logical<1>) {
// CHECK:               %[[VAL_13:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.logical<1>>) -> !fir.ref<!fir.logical<1>>
// CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<!fir.logical<1>>
// CHECK:               fir.result %[[VAL_14]] : !fir.logical<1>
// CHECK:             } else {
// CHECK:               %[[VAL_15:.*]] = arith.constant true
// CHECK:               %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i1) -> !fir.logical<1>
// CHECK:               fir.result %[[VAL_16]] : !fir.logical<1>
// CHECK:             }
// CHECK:             %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_9]] to %[[VAL_4]] step %[[VAL_9]] iter_args(%[[VAL_19:.*]] = %[[VAL_10]]) -> (f32) {
// CHECK:               %[[VAL_20:.*]] = fir.convert %[[VAL_12]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_21:.*]] = fir.if %[[VAL_20]] -> (f32) {
// CHECK:                 %[[VAL_22:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_18]], %[[VAL_8]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[VAL_23:.*]] = arith.addf %[[VAL_19]], %[[VAL_22]] : f32
// CHECK:                 fir.result %[[VAL_23]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_19]] : f32
// CHECK:               }
// CHECK:               fir.result %[[VAL_21]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_17]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// array mask
func.func @sum_array_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !fir.box<!fir.array<?x3x!fir.logical<1>>>) {
  %cst = arith.constant 2 : i32
  %res = hlfir.sum %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !fir.box<!fir.array<?x3x!fir.logical<1>>>) -> !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @sum_array_mask(
// CHECK-SAME:                              %[[VAL_0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: !fir.box<!fir.array<?x3x!fir.logical<1>>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_11:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x3x!fir.logical<1>>>) -> i1
// CHECK:             %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_9]] to %[[VAL_5]] step %[[VAL_9]] iter_args(%[[VAL_14:.*]] = %[[VAL_10]]) -> (f32) {
// CHECK:               %[[VAL_15:.*]] = fir.if %[[VAL_11]] -> (!fir.logical<1>) {
// CHECK:                 %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:                 %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_16]] : (!fir.box<!fir.array<?x3x!fir.logical<1>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:                 %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_18]] : (!fir.box<!fir.array<?x3x!fir.logical<1>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_20:.*]] = arith.constant 1 : index
// CHECK:                 %[[VAL_21:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_20]] : index
// CHECK:                 %[[VAL_22:.*]] = arith.addi %[[VAL_8]], %[[VAL_21]] : index
// CHECK:                 %[[VAL_23:.*]] = arith.subi %[[VAL_19]]#0, %[[VAL_20]] : index
// CHECK:                 %[[VAL_24:.*]] = arith.addi %[[VAL_13]], %[[VAL_23]] : index
// CHECK:                 %[[VAL_25:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_22]], %[[VAL_24]])  : (!fir.box<!fir.array<?x3x!fir.logical<1>>>, index, index) -> !fir.ref<!fir.logical<1>>
// CHECK:                 %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<!fir.logical<1>>
// CHECK:                 fir.result %[[VAL_26]] : !fir.logical<1>
// CHECK:               } else {
// CHECK:                 %[[VAL_27:.*]] = arith.constant true
// CHECK:                 %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i1) -> !fir.logical<1>
// CHECK:                 fir.result %[[VAL_28]] : !fir.logical<1>
// CHECK:               }
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_15]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_30:.*]] = fir.if %[[VAL_29]] -> (f32) {
// CHECK:                 %[[VAL_31:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_8]], %[[VAL_13]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[VAL_32:.*]] = arith.addf %[[VAL_14]], %[[VAL_31]] : f32
// CHECK:                 fir.result %[[VAL_32]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_14]] : f32
// CHECK:               }
// CHECK:               fir.result %[[VAL_30]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_12]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// array expr mask
func.func @sum_array_expr_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !hlfir.expr<?x3x!fir.logical<1>>) {
  %cst = arith.constant 2 : i32
  %res = hlfir.sum %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !hlfir.expr<?x3x!fir.logical<1>>) -> !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @sum_array_expr_mask(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !hlfir.expr<?x3x!fir.logical<1>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_5]] step %[[VAL_9]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (f32) {
// CHECK:               %[[VAL_14:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_8]], %[[VAL_12]] : (!hlfir.expr<?x3x!fir.logical<1>>, index, index) -> !fir.logical<1>
// CHECK:               %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_16:.*]] = fir.if %[[VAL_15]] -> (f32) {
// CHECK:                 %[[VAL_17:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_8]], %[[VAL_12]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[VAL_18:.*]] = arith.addf %[[VAL_13]], %[[VAL_17]] : f32
// CHECK:                 fir.result %[[VAL_18]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_13]] : f32
// CHECK:               }
// CHECK:               fir.result %[[VAL_16]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_11]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// unordered floating point reduction
func.func @sum_unordered_reduction(%arg0: !hlfir.expr<2x3xf32>) {
  %cst = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %cst {fastmath = #arith.fastmath<reassoc>} : (!hlfir.expr<2x3xf32>, i32) -> !hlfir.expr<3xf32>
  return
}
// CHECK-LABEL:   func.func @sum_unordered_reduction(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !hlfir.expr<2x3xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_2]] step %[[VAL_7]] unordered iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (f32) {
// CHECK:               %[[VAL_12:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]], %[[VAL_6]] : (!hlfir.expr<2x3xf32>, index, index) -> f32
// CHECK:               %[[VAL_13:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] fastmath<reassoc> : f32
// CHECK:               fir.result %[[VAL_13]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_9]] : f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// negative: total reduction
func.func @sum_total_reduction(%arg0: !fir.box<!fir.array<3xi32>>) {
  %cst = arith.constant 1 : i32
  %res = hlfir.sum %arg0 dim %cst : (!fir.box<!fir.array<3xi32>>, i32) -> i32
  return
}
// CHECK-LABEL:   func.func @sum_total_reduction(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<3xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = hlfir.sum %[[VAL_0]] dim %[[VAL_1]] : (!fir.box<!fir.array<3xi32>>, i32) -> i32
// CHECK:           return
// CHECK:         }

// negative: non-const dim
func.func @sum_non_const_dim(%arg0: !fir.box<!fir.array<3xi32>>, %dim: i32) {
  %res = hlfir.sum %arg0 dim %dim : (!fir.box<!fir.array<3xi32>>, i32) -> i32
  return
}
// CHECK-LABEL:   func.func @sum_non_const_dim(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.box<!fir.array<3xi32>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: i32) {
// CHECK:           %[[VAL_2:.*]] = hlfir.sum %[[VAL_0]] dim %[[VAL_1]] : (!fir.box<!fir.array<3xi32>>, i32) -> i32
// CHECK:           return
// CHECK:         }

// negative: invalid dim==0
func.func @sum_invalid_dim0(%arg0: !hlfir.expr<2x3xi32>) {
  %cst = arith.constant 0 : i32
  %res = hlfir.sum %arg0 dim %cst : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
  return
}
// CHECK-LABEL:   func.func @sum_invalid_dim0(
// CHECK:           hlfir.sum %{{.*}} dim %{{.*}} : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>

// negative: invalid dim>rank
func.func @sum_invalid_dim_big(%arg0: !hlfir.expr<2x3xi32>) {
  %cst = arith.constant 3 : i32
  %res = hlfir.sum %arg0 dim %cst : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
  return
}
// CHECK-LABEL:   func.func @sum_invalid_dim_big(
// CHECK:           hlfir.sum %{{.*}} dim %{{.*}} : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
