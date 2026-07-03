// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

// box with known extents
func.func @product_box_known_extents(%arg0: !fir.box<!fir.array<2x3xi32>>) -> !hlfir.expr<2xi32> {
  %cst = arith.constant 2 : i32
  %res = hlfir.product %arg0 dim %cst : (!fir.box<!fir.array<2x3xi32>>, i32) -> !hlfir.expr<2xi32>
  return %res : !hlfir.expr<2xi32>
}
// CHECK-LABEL:   func.func @product_box_known_extents(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<2x3xi32>>) -> !hlfir.expr<2xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<2xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] unordered iter_args(%[[VAL_2:.*]] = %[[CONSTANT_2]]) -> (i32) {
// CHECK:               %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<2x3xi32>>, index) -> (index, index, index)
// CHECK:               %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<2x3xi32>>, index) -> (index, index, index)
// CHECK:               %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_0]]#0, %[[CONSTANT_1]] : index
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[SUBI_0]] : index
// CHECK:               %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_1]]#0, %[[CONSTANT_1]] : index
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_1]], %[[SUBI_1]] : index
// CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[ARG0]] (%[[ADDI_0]], %[[ADDI_1]])  : (!fir.box<!fir.array<2x3xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
// CHECK:               %[[MULI_0:.*]] = arith.muli %[[VAL_2]], %[[LOAD_0]] : i32
// CHECK:               fir.result %[[MULI_0]] : i32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : i32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<2xi32>
// CHECK:         }

// expr with known extents
func.func @product_expr_known_extents(%arg0: !hlfir.expr<2x3xi32>) -> !hlfir.expr<3xi32> {
  %cst = arith.constant 1 : i32
  %res = hlfir.product %arg0 dim %cst : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
  return %res : !hlfir.expr<3xi32>
}
// CHECK-LABEL:   func.func @product_expr_known_extents(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<2x3xi32>) -> !hlfir.expr<3xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (i32) {
// CHECK:               %[[APPLY_0:.*]] = hlfir.apply %[[ARG0]], %[[VAL_1]], %[[VAL_0]] : (!hlfir.expr<2x3xi32>, index, index) -> i32
// CHECK:               %[[MULI_0:.*]] = arith.muli %[[VAL_2]], %[[APPLY_0]] : i32
// CHECK:               fir.result %[[MULI_0]] : i32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : i32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<3xi32>
// CHECK:         }


// box with unknown extent
func.func @product_box_unknown_extent1(%arg0: !fir.box<!fir.array<?x3xcomplex<f64>>>) -> !hlfir.expr<3xcomplex<f64>> {
  %cst = arith.constant 1 : i32
  %res = hlfir.product %arg0 dim %cst : (!fir.box<!fir.array<?x3xcomplex<f64>>>, i32) -> !hlfir.expr<3xcomplex<f64>>
  return %res : !hlfir.expr<3xcomplex<f64>>
}
// CHECK-LABEL:   func.func @product_box_unknown_extent1(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x3xcomplex<f64>>>) -> !hlfir.expr<3xcomplex<f64>> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_4]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xcomplex<f64>> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[UNDEFINED_0:.*]] = fir.undefined complex<f64>
// CHECK:             %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_2]], [0 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[CONSTANT_1]], [1 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[BOX_DIMS_0]]#1 step %[[CONSTANT_0]] iter_args(%[[VAL_2:.*]] = %[[INSERT_VALUE_1]]) -> (complex<f64>) {
// CHECK:               %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_4]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_1]]#0, %[[CONSTANT_0]] : index
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[SUBI_0]] : index
// CHECK:               %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_2]]#0, %[[CONSTANT_0]] : index
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_0]], %[[SUBI_1]] : index
// CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[ARG0]] (%[[ADDI_0]], %[[ADDI_1]])  : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index, index) -> !fir.ref<complex<f64>>
// CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<complex<f64>>
// CHECK:               %[[MULC_0:.*]] = fir.mulc %[[VAL_2]], %[[LOAD_0]] : complex<f64>
// CHECK:               fir.result %[[MULC_0]] : complex<f64>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : complex<f64>
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<3xcomplex<f64>>
// CHECK:         }


func.func @product_box_unknown_extent2(%arg0: !fir.box<!fir.array<?x3xcomplex<f64>>>) -> !hlfir.expr<?xcomplex<f64>> {
  %cst = arith.constant 2 : i32
  %res = hlfir.product %arg0 dim %cst : (!fir.box<!fir.array<?x3xcomplex<f64>>>, i32) -> !hlfir.expr<?xcomplex<f64>>
  return %res : !hlfir.expr<?xcomplex<f64>>
}
// CHECK-LABEL:   func.func @product_box_unknown_extent2(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x3xcomplex<f64>>>) -> !hlfir.expr<?xcomplex<f64>> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_4]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xcomplex<f64>> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[UNDEFINED_0:.*]] = fir.undefined complex<f64>
// CHECK:             %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_2]], [0 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[CONSTANT_1]], [1 : index] : (complex<f64>, f64) -> complex<f64>
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_0]] iter_args(%[[VAL_2:.*]] = %[[INSERT_VALUE_1]]) -> (complex<f64>) {
// CHECK:               %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_4]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index) -> (index, index, index)
// CHECK:               %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_1]]#0, %[[CONSTANT_0]] : index
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[SUBI_0]] : index
// CHECK:               %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_2]]#0, %[[CONSTANT_0]] : index
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_1]], %[[SUBI_1]] : index
// CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[ARG0]] (%[[ADDI_0]], %[[ADDI_1]])  : (!fir.box<!fir.array<?x3xcomplex<f64>>>, index, index) -> !fir.ref<complex<f64>>
// CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<complex<f64>>
// CHECK:               %[[MULC_0:.*]] = fir.mulc %[[VAL_2]], %[[LOAD_0]] : complex<f64>
// CHECK:               fir.result %[[MULC_0]] : complex<f64>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : complex<f64>
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<?xcomplex<f64>>
// CHECK:         }


// expr with unknown extent
func.func @product_expr_unknown_extent1(%arg0: !hlfir.expr<?x3xf32>) -> !hlfir.expr<3xf32> {
  %cst = arith.constant 1 : i32
  %res = hlfir.product %arg0 dim %cst : (!hlfir.expr<?x3xf32>, i32) -> !hlfir.expr<3xf32>
  return %res : !hlfir.expr<3xf32>
}
// CHECK-LABEL:   func.func @product_expr_unknown_extent1(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<?x3xf32>) -> !hlfir.expr<3xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_OF_0:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[GET_EXTENT_0:.*]] = hlfir.get_extent %[[SHAPE_OF_0]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[GET_EXTENT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (f32) {
// CHECK:               %[[APPLY_0:.*]] = hlfir.apply %[[ARG0]], %[[VAL_1]], %[[VAL_0]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[VAL_2]], %[[APPLY_0]] : f32
// CHECK:               fir.result %[[MULF_0]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : f32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<3xf32>
// CHECK:         }


func.func @product_expr_unknown_extent2(%arg0: !hlfir.expr<?x3xf32>) -> !hlfir.expr<?xf32> {
  %cst = arith.constant 2 : i32
  %res = hlfir.product %arg0 dim %cst : (!hlfir.expr<?x3xf32>, i32) -> !hlfir.expr<?xf32>
  return %res : !hlfir.expr<?xf32>
}
// CHECK-LABEL:   func.func @product_expr_unknown_extent2(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<?x3xf32>) -> !hlfir.expr<?xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_OF_0:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[GET_EXTENT_0:.*]] = hlfir.get_extent %[[SHAPE_OF_0]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[GET_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (f32) {
// CHECK:               %[[APPLY_0:.*]] = hlfir.apply %[[ARG0]], %[[VAL_0]], %[[VAL_1]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[VAL_2]], %[[APPLY_0]] : f32
// CHECK:               fir.result %[[MULF_0]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : f32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<?xf32>
// CHECK:         }

// scalar mask
func.func @product_scalar_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !fir.ref<!fir.logical<1>>) -> !hlfir.expr<3xf32> {
  %cst = arith.constant 1 : i32
  %res = hlfir.product %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !fir.ref<!fir.logical<1>>) -> !hlfir.expr<3xf32>
  return %res : !hlfir.expr<3xf32>
}
// CHECK-LABEL:   func.func @product_scalar_mask(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.logical<1>>) -> !hlfir.expr<3xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_OF_0:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[GET_EXTENT_0:.*]] = hlfir.get_extent %[[SHAPE_OF_0]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.logical<1>>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[GET_EXTENT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (f32) {
// CHECK:               %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (!fir.logical<1>) -> i1
// CHECK:               %[[IF_0:.*]] = fir.if %[[CONVERT_0]] -> (f32) {
// CHECK:                 %[[APPLY_0:.*]] = hlfir.apply %[[ARG0]], %[[VAL_1]], %[[VAL_0]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[MULF_0:.*]] = arith.mulf %[[VAL_2]], %[[APPLY_0]] : f32
// CHECK:                 fir.result %[[MULF_0]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_2]] : f32
// CHECK:               }
// CHECK:               fir.result %[[IF_0]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : f32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<3xf32>
// CHECK:         }

// scalar boxed mask
func.func @product_scalar_boxed_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !fir.box<!fir.logical<1>>) -> !hlfir.expr<3xf32> {
  %cst = arith.constant 1 : i32
  %res = hlfir.product %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !fir.box<!fir.logical<1>>) -> !hlfir.expr<3xf32>
  return %res : !hlfir.expr<3xf32>
}
// CHECK-LABEL:   func.func @product_scalar_boxed_mask(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.logical<1>>) -> !hlfir.expr<3xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_OF_0:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[GET_EXTENT_0:.*]] = hlfir.get_extent %[[SHAPE_OF_0]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[IS_PRESENT_0:.*]] = fir.is_present %[[ARG1]] : (!fir.box<!fir.logical<1>>) -> i1
// CHECK:           %[[IF_0:.*]] = fir.if %[[IS_PRESENT_0]] -> (!fir.logical<1>) {
// CHECK:             %[[BOX_ADDR_0:.*]] = fir.box_addr %[[ARG1]] : (!fir.box<!fir.logical<1>>) -> !fir.ref<!fir.logical<1>>
// CHECK:             %[[LOAD_0:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.ref<!fir.logical<1>>
// CHECK:             fir.result %[[LOAD_0]] : !fir.logical<1>
// CHECK:           } else {
// CHECK:             %[[CONVERT_0:.*]] = fir.convert %[[CONSTANT_2]] : (i1) -> !fir.logical<1>
// CHECK:             fir.result %[[CONVERT_0]] : !fir.logical<1>
// CHECK:           }
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[GET_EXTENT_0]] step %[[CONSTANT_0]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (f32) {
// CHECK:               %[[CONVERT_1:.*]] = fir.convert %[[IF_0]] : (!fir.logical<1>) -> i1
// CHECK:               %[[IF_1:.*]] = fir.if %[[CONVERT_1]] -> (f32) {
// CHECK:                 %[[APPLY_0:.*]] = hlfir.apply %[[ARG0]], %[[VAL_1]], %[[VAL_0]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[MULF_0:.*]] = arith.mulf %[[VAL_2]], %[[APPLY_0]] : f32
// CHECK:                 fir.result %[[MULF_0]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_2]] : f32
// CHECK:               }
// CHECK:               fir.result %[[IF_1]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : f32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<3xf32>
// CHECK:         }


// array mask
func.func @product_array_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !fir.box<!fir.array<?x3x!fir.logical<1>>>) -> !hlfir.expr<?xf32> {
  %cst = arith.constant 2 : i32
  %res = hlfir.product %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !fir.box<!fir.array<?x3x!fir.logical<1>>>) -> !hlfir.expr<?xf32>
  return %res : !hlfir.expr<?xf32>
}
// CHECK-LABEL:   func.func @product_array_mask(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?x3x!fir.logical<1>>>) -> !hlfir.expr<?xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_OF_0:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[GET_EXTENT_0:.*]] = hlfir.get_extent %[[SHAPE_OF_0]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[GET_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[IS_PRESENT_0:.*]] = fir.is_present %[[ARG1]] : (!fir.box<!fir.array<?x3x!fir.logical<1>>>) -> i1
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_3]]) -> (f32) {
// CHECK:               %[[IF_0:.*]] = fir.if %[[IS_PRESENT_0]] -> (!fir.logical<1>) {
// CHECK:                 %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[ARG1]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?x3x!fir.logical<1>>>, index) -> (index, index, index)
// CHECK:                 %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[ARG1]], %[[CONSTANT_2]] : (!fir.box<!fir.array<?x3x!fir.logical<1>>>, index) -> (index, index, index)
// CHECK:                 %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_0]]#0, %[[CONSTANT_2]] : index
// CHECK:                 %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[SUBI_0]] : index
// CHECK:                 %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_1]]#0, %[[CONSTANT_2]] : index
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[VAL_1]], %[[SUBI_1]] : index
// CHECK:                 %[[DESIGNATE_0:.*]] = hlfir.designate %[[ARG1]] (%[[ADDI_0]], %[[ADDI_1]])  : (!fir.box<!fir.array<?x3x!fir.logical<1>>>, index, index) -> !fir.ref<!fir.logical<1>>
// CHECK:                 %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<!fir.logical<1>>
// CHECK:                 fir.result %[[LOAD_0]] : !fir.logical<1>
// CHECK:               } else {
// CHECK:                 %[[CONVERT_0:.*]] = fir.convert %[[CONSTANT_0]] : (i1) -> !fir.logical<1>
// CHECK:                 fir.result %[[CONVERT_0]] : !fir.logical<1>
// CHECK:               }
// CHECK:               %[[CONVERT_1:.*]] = fir.convert %[[IF_0]] : (!fir.logical<1>) -> i1
// CHECK:               %[[IF_1:.*]] = fir.if %[[CONVERT_1]] -> (f32) {
// CHECK:                 %[[APPLY_0:.*]] = hlfir.apply %[[ARG0]], %[[VAL_0]], %[[VAL_1]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[MULF_0:.*]] = arith.mulf %[[VAL_2]], %[[APPLY_0]] : f32
// CHECK:                 fir.result %[[MULF_0]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_2]] : f32
// CHECK:               }
// CHECK:               fir.result %[[IF_1]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : f32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<?xf32>
// CHECK:         }


// array expr mask
func.func @product_array_expr_mask(%arg0: !hlfir.expr<?x3xf32>, %mask: !hlfir.expr<?x3x!fir.logical<1>>) -> !hlfir.expr<?xf32> {
  %cst = arith.constant 2 : i32
  %res = hlfir.product %arg0 dim %cst mask %mask : (!hlfir.expr<?x3xf32>, i32, !hlfir.expr<?x3x!fir.logical<1>>) -> !hlfir.expr<?xf32>
  return %res : !hlfir.expr<?xf32>
}
// CHECK-LABEL:   func.func @product_array_expr_mask(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<?x3xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: !hlfir.expr<?x3x!fir.logical<1>>) -> !hlfir.expr<?xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_OF_0:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x3xf32>) -> !fir.shape<2>
// CHECK:           %[[GET_EXTENT_0:.*]] = hlfir.get_extent %[[SHAPE_OF_0]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[GET_EXTENT_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] iter_args(%[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (f32) {
// CHECK:               %[[APPLY_0:.*]] = hlfir.apply %[[ARG1]], %[[VAL_0]], %[[VAL_1]] : (!hlfir.expr<?x3x!fir.logical<1>>, index, index) -> !fir.logical<1>
// CHECK:               %[[CONVERT_0:.*]] = fir.convert %[[APPLY_0]] : (!fir.logical<1>) -> i1
// CHECK:               %[[IF_0:.*]] = fir.if %[[CONVERT_0]] -> (f32) {
// CHECK:                 %[[APPLY_1:.*]] = hlfir.apply %[[ARG0]], %[[VAL_0]], %[[VAL_1]] : (!hlfir.expr<?x3xf32>, index, index) -> f32
// CHECK:                 %[[MULF_0:.*]] = arith.mulf %[[VAL_2]], %[[APPLY_1]] : f32
// CHECK:                 fir.result %[[MULF_0]] : f32
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_2]] : f32
// CHECK:               }
// CHECK:               fir.result %[[IF_0]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : f32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<?xf32>
// CHECK:         }

// unordered floating point reduction
func.func @product_unordered_reduction(%arg0: !hlfir.expr<2x3xf32>) -> !hlfir.expr<3xf32> {
  %cst = arith.constant 1 : i32
  %res = hlfir.product %arg0 dim %cst {fastmath = #arith.fastmath<reassoc>} : (!hlfir.expr<2x3xf32>, i32) -> !hlfir.expr<3xf32>
  return %res : !hlfir.expr<3xf32>
}
// CHECK-LABEL:   func.func @product_unordered_reduction(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<2x3xf32>) -> !hlfir.expr<3xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 3 : index
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL_0:.*]] = hlfir.elemental %[[SHAPE_0]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (f32) {
// CHECK:               %[[APPLY_0:.*]] = hlfir.apply %[[ARG0]], %[[VAL_1]], %[[VAL_0]] : (!hlfir.expr<2x3xf32>, index, index) -> f32
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[VAL_2]], %[[APPLY_0]] fastmath<reassoc> : f32
// CHECK:               fir.result %[[MULF_0]] : f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[DO_LOOP_0]] : f32
// CHECK:           }
// CHECK:           return %[[ELEMENTAL_0]] : !hlfir.expr<3xf32>
// CHECK:         }


// total 1d reduction
func.func @product_total_1d_reduction(%arg0: !fir.box<!fir.array<3xi32>>) -> i32 {
  %cst = arith.constant 1 : i32
  %res = hlfir.product %arg0 dim %cst : (!fir.box<!fir.array<3xi32>>, i32) -> i32
  return %res : i32
}
// CHECK-LABEL:   func.func @product_total_1d_reduction(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<3xi32>>) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_1]] step %[[CONSTANT_3]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_2]]) -> (i32) {
// CHECK:             %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<3xi32>>, index) -> (index, index, index)
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_0]]#0, %[[CONSTANT_3]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[SUBI_0]] : index
// CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[ARG0]] (%[[ADDI_0]])  : (!fir.box<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[LOAD_0]] : i32
// CHECK:             fir.result %[[MULI_0]] : i32
// CHECK:           }
// CHECK:           return %[[DO_LOOP_0]] : i32
// CHECK:         }

// total 2d reduction
func.func @product_total_2d_reduction(%arg0: !fir.box<!fir.array<?x3xi32>>) -> i32 {
  %res = hlfir.product %arg0 : (!fir.box<!fir.array<?x3xi32>>) -> i32
  return %res : i32
}
// CHECK-LABEL:   func.func @product_total_2d_reduction(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x3xi32>>) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_3]] : (!fir.box<!fir.array<?x3xi32>>, index) -> (index, index, index)
// CHECK:           %[[DO_LOOP_0:.*]] = fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] unordered iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (i32) {
// CHECK:             %[[DO_LOOP_1:.*]] = fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[BOX_DIMS_0]]#1 step %[[CONSTANT_0]] unordered iter_args(%[[VAL_3:.*]] = %[[VAL_1]]) -> (i32) {
// CHECK:               %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_3]] : (!fir.box<!fir.array<?x3xi32>>, index) -> (index, index, index)
// CHECK:               %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[ARG0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?x3xi32>>, index) -> (index, index, index)
// CHECK:               %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_1]]#0, %[[CONSTANT_0]] : index
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[SUBI_0]] : index
// CHECK:               %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_2]]#0, %[[CONSTANT_0]] : index
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_0]], %[[SUBI_1]] : index
// CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[ARG0]] (%[[ADDI_0]], %[[ADDI_1]])  : (!fir.box<!fir.array<?x3xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
// CHECK:               %[[MULI_0:.*]] = arith.muli %[[VAL_3]], %[[LOAD_0]] : i32
// CHECK:               fir.result %[[MULI_0]] : i32
// CHECK:             }
// CHECK:             fir.result %[[DO_LOOP_1]] : i32
// CHECK:           }
// CHECK:           return %[[DO_LOOP_0]] : i32
// CHECK:         }


// negative: invalid dim==0
func.func @product_invalid_dim0(%arg0: !hlfir.expr<2x3xi32>) -> !hlfir.expr<3xi32> {
  %cst = arith.constant 0 : i32
  %res = hlfir.product %arg0 dim %cst : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
  return %res : !hlfir.expr<3xi32>
}
// CHECK-LABEL:   func.func @product_invalid_dim0(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<2x3xi32>) -> !hlfir.expr<3xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[PRODUCT_0:.*]] = hlfir.product %[[ARG0]] dim %[[CONSTANT_0]] : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
// CHECK:           return %[[PRODUCT_0]] : !hlfir.expr<3xi32>
// CHECK:         }

// negative: invalid dim>rank
func.func @product_invalid_dim_big(%arg0: !hlfir.expr<2x3xi32>) -> !hlfir.expr<3xi32> {
  %cst = arith.constant 3 : i32
  %res = hlfir.product %arg0 dim %cst : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
  return %res : !hlfir.expr<3xi32>
}
// CHECK-LABEL:   func.func @product_invalid_dim_big(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<2x3xi32>) -> !hlfir.expr<3xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : i32
// CHECK:           %[[PRODUCT_0:.*]] = hlfir.product %[[ARG0]] dim %[[CONSTANT_0]] : (!hlfir.expr<2x3xi32>, i32) -> !hlfir.expr<3xi32>
// CHECK:           return %[[PRODUCT_0]] : !hlfir.expr<3xi32>
// CHECK:         }