// Test hlfir.dot_product simplification to a reduction loop:
// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

func.func @dot_product_integer(%arg0: !hlfir.expr<?xi16>, %arg1: !hlfir.expr<?xi32>) -> i32 {
  %res = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<?xi16>, !hlfir.expr<?xi32>) -> i32
  return %res : i32
}
// CHECK-LABEL:   func.func @dot_product_integer(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !hlfir.expr<?xi16>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !hlfir.expr<?xi32>) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xi16>) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_6:.*]] = fir.do_loop %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] unordered iter_args(%[[VAL_8:.*]] = %[[VAL_3]]) -> (i32) {
// CHECK:             %[[VAL_9:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_7]] : (!hlfir.expr<?xi16>, index) -> i16
// CHECK:             %[[VAL_10:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_7]] : (!hlfir.expr<?xi32>, index) -> i32
// CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_9]] : (i16) -> i32
// CHECK:             %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_10]] : i32
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_8]], %[[VAL_12]] : i32
// CHECK:             fir.result %[[VAL_13]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_6]] : i32
// CHECK:         }

func.func @dot_product_real(%arg0: !hlfir.expr<?xf32>, %arg1: !hlfir.expr<?xf16>) -> f32 {
  %res = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<?xf32>, !hlfir.expr<?xf16>) -> f32
  return %res : f32
}
// CHECK-LABEL:   func.func @dot_product_real(
// CHECK-SAME:                                %[[VAL_0:.*]]: !hlfir.expr<?xf32>,
// CHECK-SAME:                                %[[VAL_1:.*]]: !hlfir.expr<?xf16>) -> f32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xf32>) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_6:.*]] = fir.do_loop %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_8:.*]] = %[[VAL_3]]) -> (f32) {
// CHECK:             %[[VAL_9:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_7]] : (!hlfir.expr<?xf32>, index) -> f32
// CHECK:             %[[VAL_10:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_7]] : (!hlfir.expr<?xf16>, index) -> f16
// CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (f16) -> f32
// CHECK:             %[[VAL_12:.*]] = arith.mulf %[[VAL_9]], %[[VAL_11]] : f32
// CHECK:             %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_12]] : f32
// CHECK:             fir.result %[[VAL_13]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_6]] : f32
// CHECK:         }

func.func @dot_product_complex(%arg0: !hlfir.expr<?xcomplex<f32>>, %arg1: !hlfir.expr<?xcomplex<f16>>) -> complex<f32> {
  %res = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<?xcomplex<f32>>, !hlfir.expr<?xcomplex<f16>>) -> complex<f32>
  return %res : complex<f32>
}
// CHECK-LABEL:   func.func @dot_product_complex(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !hlfir.expr<?xcomplex<f32>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !hlfir.expr<?xcomplex<f16>>) -> complex<f32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xcomplex<f32>>) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_6:.*]] = fir.undefined complex<f32>
// CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_3]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_3]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:           %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (complex<f32>) {
// CHECK:             %[[VAL_12:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]] : (!hlfir.expr<?xcomplex<f32>>, index) -> complex<f32>
// CHECK:             %[[VAL_13:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_10]] : (!hlfir.expr<?xcomplex<f16>>, index) -> complex<f16>
// CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (complex<f16>) -> complex<f32>
// CHECK:             %[[VAL_15:.*]] = fir.extract_value %[[VAL_12]], [1 : index] : (complex<f32>) -> f32
// CHECK:             %[[VAL_16:.*]] = arith.negf %[[VAL_15]] : f32
// CHECK:             %[[VAL_17:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_16]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:             %[[VAL_18:.*]] = fir.mulc %[[VAL_17]], %[[VAL_14]] : complex<f32>
// CHECK:             %[[VAL_19:.*]] = fir.addc %[[VAL_11]], %[[VAL_18]] : complex<f32>
// CHECK:             fir.result %[[VAL_19]] : complex<f32>
// CHECK:           }
// CHECK:           return %[[VAL_9]] : complex<f32>
// CHECK:         }

func.func @dot_product_real_complex(%arg0: !hlfir.expr<?xf32>, %arg1: !hlfir.expr<?xcomplex<f16>>) -> complex<f32> {
  %res = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<?xf32>, !hlfir.expr<?xcomplex<f16>>) -> complex<f32>
  return %res : complex<f32>
}
// CHECK-LABEL:   func.func @dot_product_real_complex(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !hlfir.expr<?xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !hlfir.expr<?xcomplex<f16>>) -> complex<f32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xf32>) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_6:.*]] = fir.undefined complex<f32>
// CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_3]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_3]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:           %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (complex<f32>) {
// CHECK:             %[[VAL_12:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]] : (!hlfir.expr<?xf32>, index) -> f32
// CHECK:             %[[VAL_13:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_10]] : (!hlfir.expr<?xcomplex<f16>>, index) -> complex<f16>
// CHECK:             %[[VAL_14:.*]] = fir.undefined complex<f32>
// CHECK:             %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_3]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:             %[[VAL_16:.*]] = fir.insert_value %[[VAL_15]], %[[VAL_3]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:             %[[VAL_17:.*]] = fir.insert_value %[[VAL_16]], %[[VAL_12]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_13]] : (complex<f16>) -> complex<f32>
// CHECK:             %[[VAL_19:.*]] = fir.extract_value %[[VAL_17]], [1 : index] : (complex<f32>) -> f32
// CHECK:             %[[VAL_20:.*]] = arith.negf %[[VAL_19]] : f32
// CHECK:             %[[VAL_21:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_20]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:             %[[VAL_22:.*]] = fir.mulc %[[VAL_21]], %[[VAL_18]] : complex<f32>
// CHECK:             %[[VAL_23:.*]] = fir.addc %[[VAL_11]], %[[VAL_22]] : complex<f32>
// CHECK:             fir.result %[[VAL_23]] : complex<f32>
// CHECK:           }
// CHECK:           return %[[VAL_9]] : complex<f32>
// CHECK:         }

func.func @dot_product_logical(%arg0: !hlfir.expr<?x!fir.logical<1>>, %arg1: !hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4> {
  %res = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<?x!fir.logical<1>>, !hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
  return %res : !fir.logical<4>
}
// CHECK-LABEL:   func.func @dot_product_logical(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !hlfir.expr<?x!fir.logical<1>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant false
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x!fir.logical<1>>) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
// CHECK:           %[[VAL_7:.*]] = fir.do_loop %[[VAL_8:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] unordered iter_args(%[[VAL_9:.*]] = %[[VAL_6]]) -> (!fir.logical<4>) {
// CHECK:             %[[VAL_10:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_8]] : (!hlfir.expr<?x!fir.logical<1>>, index) -> !fir.logical<1>
// CHECK:             %[[VAL_11:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_8]] : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_9]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (!fir.logical<1>) -> i1
// CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_11]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_15:.*]] = arith.andi %[[VAL_13]], %[[VAL_14]] : i1
// CHECK:             %[[VAL_16:.*]] = arith.ori %[[VAL_12]], %[[VAL_15]] : i1
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i1) -> !fir.logical<4>
// CHECK:             fir.result %[[VAL_17]] : !fir.logical<4>
// CHECK:           }
// CHECK:           return %[[VAL_7]] : !fir.logical<4>
// CHECK:         }

func.func @dot_product_known_dim(%arg0: !hlfir.expr<10xf32>, %arg1: !hlfir.expr<?xi16>) -> f32 {
  %res1 = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<10xf32>, !hlfir.expr<?xi16>) -> f32
  %res2 = hlfir.dot_product %arg1 %arg0 : (!hlfir.expr<?xi16>, !hlfir.expr<10xf32>) -> f32
  %res = arith.addf %res1, %res2 : f32
  return %res : f32
}
// CHECK-LABEL:   func.func @dot_product_known_dim(
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           fir.do_loop %{{.*}} = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_2]]
// CHECK:           fir.do_loop %{{.*}} = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_2]]
