// RUN: fir-opt --split-input-file --simplify-intrinsics %s | FileCheck %s

// Call to SUM with 1D I32 array is replaced.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_1d_array_int(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> i32 {
    %c10 = arith.constant 10 : index
    %0 = fir.alloca i32 {bindc_name = "test_sum_2", uniq_name = "_QFtest_sum_2Etest_sum_2"}
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F6973756D5F322E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranASumInteger4(%5, %6, %c5_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
    fir.store %9 to %0 : !fir.ref<i32>
    %10 = fir.load %0 : !fir.ref<i32>
    return %10 : i32
  }
  func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F322E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./isum_2.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}


// CHECK-LABEL:   func.func @sum_1d_array_int(
// CHECK-SAME:                             %[[A:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> i32 {
// CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_I32:.*]] = fir.embox %[[A]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_I32]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           %[[RES:.*]] = fir.call @_FortranASumInteger4x1_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> i32
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           return %{{.*}} : i32
// CHECK:         }
// CHECK:         func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}

// CHECK-LABEL:   func.func private @_FortranASumInteger4x1_simplified(
// CHECK-SAME:                                                       %[[ARR:.*]]: !fir.box<none>) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[CI32_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[SUM:.*]] = %[[CI32_0]]) -> (i32) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:             %[[NEW_SUM:.*]] = arith.addi %[[ITEM_VAL]], %[[SUM]] : i32
// CHECK:             fir.result %[[NEW_SUM]] : i32
// CHECK:           }
// CHECK:           return %[[RES]] : i32
// CHECK:         }

// -----

// Call to SUM with 2D I32 arrays is replaced.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_2d_array_int(%arg0: !fir.ref<!fir.array<10x10xi32>> {fir.bindc_name = "a"}) -> i32 {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    %0 = fir.alloca i32 {bindc_name = "test_sum_3", uniq_name = "_QFtest_sum_3Etest_sum_3"}
    %1 = fir.shape %c10, %c10_0 : (index, index) -> !fir.shape<2>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F6973756D5F332E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranASumInteger4(%5, %6, %c5_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
    fir.store %9 to %0 : !fir.ref<i32>
    %10 = fir.load %0 : !fir.ref<i32>
    return %10 : i32
  }
  func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F332E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./isum_3.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}

// CHECK-LABEL:   func.func @sum_2d_array_int({{.*}} !fir.ref<!fir.array<10x10xi32>> {fir.bindc_name = "a"}) -> i32 {
// CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index, index) -> !fir.shape<2>
// CHECK:           %[[A_BOX_I32:.*]] = fir.embox %[[A]](%[[SHAPE]]) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_I32]] : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<none>
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           %[[RES:.*]] = fir.call @_FortranASumInteger4x2_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> i32
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           return %{{.*}} : i32
// CHECK:         }
// CHECK:         func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}

// CHECK-LABEL:   func.func private @_FortranASumInteger4x2_simplified(
// CHECK-SAME:                                                       %[[ARR:.*]]: !fir.box<none>) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK:           %[[CI32_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS_0:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT_0:.*]] = arith.subi %[[DIMS_0]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[DIMIDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMS_1:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[DIMIDX_1]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT_1:.*]] = arith.subi %[[DIMS_1]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES_1:.*]] = fir.do_loop %[[ITER_1:.*]] = %[[CINDEX_0]] to %[[EXTENT_1]] step %[[CINDEX_1]] iter_args(%[[SUM_1:.*]] = %[[CI32_0]]) -> (i32) {
// CHECK:             %[[RES_0:.*]] = fir.do_loop %[[ITER_0:.*]] = %[[CINDEX_0]] to %[[EXTENT_0]] step %[[CINDEX_1]] iter_args(%[[SUM_0:.*]] = %[[SUM_1]]) -> (i32) {
// CHECK:               %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER_0]], %[[ITER_1]] : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:               %[[NEW_SUM:.*]] = arith.addi %[[ITEM_VAL]], %[[SUM_0]] : i32
// CHECK:               fir.result %[[NEW_SUM]] : i32
// CHECK:             }
// CHECK:             fir.result %[[RES_0]]
// CHECK:           }
// CHECK:           return %[[RES_1]] : i32
// CHECK:         }

// -----

// Call to SUM with 1D F64 is replaced.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_1d_real(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}) -> f64 {
    %c10 = arith.constant 10 : index
    %0 = fir.alloca f64 {bindc_name = "sum_1d_real", uniq_name = "_QFsum_1d_realEsum_1d_real"}
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F6973756D5F352E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<10xf64>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranASumReal8(%5, %6, %c5_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f64
    fir.store %9 to %0 : !fir.ref<f64>
    %10 = fir.load %0 : !fir.ref<f64>
    return %10 : f64
  }
  func.func private @_FortranASumReal8(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f64 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F352E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./isum_5.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}


// CHECK-LABEL:   func.func @sum_1d_real(
// CHECK-SAME:                           %[[A:.*]]: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}) -> f64 {
// CHECK:           %[[CINDEX_10:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[CINDEX_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_F64:.*]] = fir.embox %[[A]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_F64]] : (!fir.box<!fir.array<10xf64>>) -> !fir.box<none>
// CHECK-NOT:       fir.call @_FortranASumReal8({{.*}})
// CHECK:           %[[RES:.*]] = fir.call @_FortranASumReal8x1_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> f64
// CHECK-NOT:       fir.call @_FortranASumReal8({{.*}})
// CHECK:           return %{{.*}} : f64
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranASumReal8x1_simplified(
// CHECK-SAME:                                                    %[[ARR:.*]]: !fir.box<none>) -> f64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_F64:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_F64]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[SUM]] = %[[ZERO]]) -> (f64) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_F64]], %[[ITER]] : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<f64>
// CHECK:             %[[NEW_SUM:.*]] = arith.addf %[[ITEM_VAL]], %[[SUM]] : f64
// CHECK:             fir.result %[[NEW_SUM]] : f64
// CHECK:           }
// CHECK:           return %[[RES]] : f64
// CHECK:         }

// -----

// Call to SUM with 1D F32 is replaced.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_1d_real(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "a"}) -> f32 {
    %c10 = arith.constant 10 : index
    %0 = fir.alloca f32 {bindc_name = "sum_1d_real", uniq_name = "_QFsum_1d_realEsum_1d_real"}
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F6973756D5F352E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranASumReal4(%5, %6, %c5_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f32
    fir.store %9 to %0 : !fir.ref<f32>
    %10 = fir.load %0 : !fir.ref<f32>
    return %10 : f32
  }
  func.func private @_FortranASumReal4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f32 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F352E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./isum_5.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}


// CHECK-LABEL:   func.func @sum_1d_real(
// CHECK-SAME:                           %[[A:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "a"}) -> f32 {
// CHECK:           %[[CINDEX_10:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[CINDEX_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_F32:.*]] = fir.embox %[[A]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_F32]] : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
// CHECK-NOT:       fir.call @_FortranASumReal4({{.*}})
// CHECK:           %[[RES:.*]] = fir.call @_FortranASumReal4x1_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> f32
// CHECK-NOT:       fir.call @_FortranASumReal4({{.*}})
// CHECK:           return %{{.*}} : f32
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranASumReal4x1_simplified(
// CHECK-SAME:                                                    %[[ARR:.*]]: !fir.box<none>) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_F32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_F32]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[SUM]] = %[[ZERO]]) -> (f32) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_F32]], %[[ITER]] : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<f32>
// CHECK:             %[[NEW_SUM:.*]] = arith.addf %[[ITEM_VAL]], %[[SUM]] : f32
// CHECK:             fir.result %[[NEW_SUM]] : f32
// CHECK:           }
// CHECK:           return %[[RES]] : f32
// CHECK:         }

// -----

// Call to SUM with 1D COMPLEX array is not replaced.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_1d_complex(%arg0: !fir.ref<!fir.array<10x!fir.complex<4>>> {fir.bindc_name = "a"}) -> !fir.complex<4> {
    %0 = fir.alloca !fir.complex<4>
    %c10 = arith.constant 10 : index
    %1 = fir.alloca !fir.complex<4> {bindc_name = "sum_1d_complex", uniq_name = "_QFsum_1d_complexEsum_1d_complex"}
    %2 = fir.shape %c10 : (index) -> !fir.shape<1>
    %3 = fir.embox %arg0(%2) : (!fir.ref<!fir.array<10x!fir.complex<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.complex<4>>>
    %4 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %5 = fir.address_of(@_QQcl.2E2F6973756D5F362E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %6 = fir.convert %0 : (!fir.ref<!fir.complex<4>>) -> !fir.ref<complex<f32>>
    %7 = fir.convert %3 : (!fir.box<!fir.array<10x!fir.complex<4>>>) -> !fir.box<none>
    %8 = fir.convert %5 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %9 = fir.convert %c0 : (index) -> i32
    %10 = fir.convert %4 : (!fir.box<i1>) -> !fir.box<none>
    %11 = fir.call @_FortranACppSumComplex4(%6, %7, %8, %c5_i32, %9, %10) : (!fir.ref<complex<f32>>, !fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> none
    %12 = fir.load %0 : !fir.ref<!fir.complex<4>>
    fir.store %12 to %1 : !fir.ref<!fir.complex<4>>
    %13 = fir.load %1 : !fir.ref<!fir.complex<4>>
    return %13 : !fir.complex<4>
  }
  func.func private @_FortranACppSumComplex4(!fir.ref<complex<f32>>, !fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> none attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F362E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./isum_6.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}

// CHECK-LABEL:   func.func @sum_1d_complex(%{{.*}}: !fir.ref<!fir.array<10x!fir.complex<4>>> {fir.bindc_name = "a"}) -> !fir.complex<4> {
// CHECK-NOT:       fir.call @_FortranACppSumComplex4x1_simplified({{.*}})
// CHECK:           fir.call @_FortranACppSumComplex4({{.*}}) : (!fir.ref<complex<f32>>, !fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> none
// CHECK-NOT:       fir.call @_FortranACppSumComplex4x1_simplified({{.*}})

// -----

// Test that two functions calling the same SUM function
// generates only ONE function declaration (and that both
// calls are converted)
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_1d_calla(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> i32 {
    %c10 = arith.constant 10 : index
    %0 = fir.alloca i32 {bindc_name = "sum_1d_calla", uniq_name = "_QFsum_1d_callaEsum_1d_calla"}
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F6973756D5F372E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranASumInteger4(%5, %6, %c5_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
    fir.store %9 to %0 : !fir.ref<i32>
    %10 = fir.load %0 : !fir.ref<i32>
    return %10 : i32
  }
  func.func @sum_1d_callb(%arg0: !fir.ref<!fir.array<20xi32>> {fir.bindc_name = "a"}) -> i32 {
    %c20 = arith.constant 20 : index
    %0 = fir.alloca i32 {bindc_name = "sum_1d_callb", uniq_name = "_QFsum_1d_callbEsum_1d_callb"}
    %1 = fir.shape %c20 : (index) -> !fir.shape<1>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<20xi32>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F6973756D5F372E66393000) : !fir.ref<!fir.char<1,13>>
    %c12_i32 = arith.constant 12 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<20xi32>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranASumInteger4(%5, %6, %c12_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
    fir.store %9 to %0 : !fir.ref<i32>
    %10 = fir.load %0 : !fir.ref<i32>
    return %10 : i32
  }
  func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F372E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./isum_7.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}

// CHECK-LABEL:   func.func @sum_1d_calla(%{{.*}}) -> i32 {
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           fir.call @_FortranASumInteger4x1_simplified(%{{.*}})
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:         }

// CHECK-LABEL:   func.func @sum_1d_callb(%{{.*}}) -> i32 {
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           fir.call @_FortranASumInteger4x1_simplified(%{{.*}})
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranASumInteger4x1_simplified({{.*}}) -> i32 {{.*}} {
// CHECK:           return %{{.*}} : i32
// CHECK:         }
// CHECK-NOT:   func.func private @_FortranASumInteger4x1_simplified({{.*}})

// -----

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_1d_stride(%arg0: !fir.ref<!fir.array<20xi32>> {fir.bindc_name = "a"}) -> i32 {
    %c20 = arith.constant 20 : index
    %0 = fir.alloca i32 {bindc_name = "sum_1d_stride", uniq_name = "_QFsum_1d_strideEsum_1d_stride"}
    %c1 = arith.constant 1 : index
    %c2_i64 = arith.constant 2 : i64
    %1 = fir.convert %c2_i64 : (i64) -> index
    %2 = arith.addi %c1, %c20 : index
    %3 = arith.subi %2, %c1 : index
    %4 = fir.shape %c20 : (index) -> !fir.shape<1>
    %5 = fir.slice %c1, %3, %1 : (index, index, index) -> !fir.slice<1>
    %6 = fir.embox %arg0(%4) [%5] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
    %7 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %8 = fir.address_of(@_QQcl.2E2F6973756D5F382E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %9 = fir.convert %6 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
    %10 = fir.convert %8 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %11 = fir.convert %c0 : (index) -> i32
    %12 = fir.convert %7 : (!fir.box<i1>) -> !fir.box<none>
    %13 = fir.call @_FortranASumInteger4(%9, %10, %c5_i32, %11, %12) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
    fir.store %13 to %0 : !fir.ref<i32>
    %14 = fir.load %0 : !fir.ref<i32>
    return %14 : i32
  }
  func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F382E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./isum_8.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}

// CHECK-LABEL:   func.func @sum_1d_stride(%{{.*}} -> i32 {
// CHECK:           %[[CI64_2:.*]] = arith.constant 2 : i64
// CHECK:           %[[CINDEX_2:.*]] = fir.convert %[[CI64_2]] : (i64) -> index
// CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}}
// CHECK:           %[[SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %[[CINDEX_2]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[A_BOX_I32:.*]] = fir.embox %{{.*}}(%[[SHAPE]]) {{\[}}%[[SLICE]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_I32]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %{{.*}} = fir.call @_FortranASumInteger4x1_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> i32
// CHECK:           return %{{.*}} : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranASumInteger4x1_simplified(%{{.*}}) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %{{.*}} : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %{{.*}} to %[[EXTENT]] step %[[CINDEX_1]] iter_args({{.*}}) -> (i32) {
// CHECK:             %{{.*}} = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           }
// CHECK:           return %[[RES]] : i32
// CHECK:         }

// -----

// Check that the compiler accepts unknown size arrays.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_dim() {
    %arr = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
    %var = fir.alloca !fir.array<8x8xi32>  
    %size = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %lineno = arith.constant 12 : i32
    %shape = fir.shape %size, %size : (index, index) -> !fir.shape<2>
    %slice = fir.slice %c1, %size, %c1, %c1, %size, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
    %box_array = fir.embox %var(%shape) [%slice] : (!fir.ref<!fir.array<8x8xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xi32>>
    %box_none = fir.convert %arr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
    %box_none2 = fir.convert %box_array : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
    %absent = fir.absent !fir.box<i1>
    %file = fir.address_of(@filename) : !fir.ref<!fir.char<1,16>>
    %file_ref = fir.convert %file : (!fir.ref<!fir.char<1,16>>) -> !fir.ref<i8>
    %absent_none = fir.convert %absent : (!fir.box<i1>) -> !fir.box<none>
    %res = fir.call @_FortranASumDim(%box_none, %box_none2, %c1_i32, %file_ref, %lineno, %absent_none) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>) -> none
    func.return
  }
}

// Just check that SOMETHING is being output.
// CHECK-LABEL @sum_dim() {
// CHECK: return


// -----

// Using an unknown size.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @sum_1d_unknown(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) -> i32 {
    %0 = fir.alloca i32 {bindc_name = "test_sum_1", uniq_name = "_QFtest_sum_1Etest_sum_1"}
    %1 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %2 = fir.address_of(@_QQcl.2E2F696D61785F312E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %3 = fir.convert %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
    %4 = fir.convert %2 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %5 = fir.convert %c0 : (index) -> i32
    %6 = fir.convert %1 : (!fir.box<i1>) -> !fir.box<none>
    %7 = fir.call @_FortranASumlInteger4(%3, %4, %c5_i32, %5, %6) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
    fir.store %7 to %0 : !fir.ref<i32>
    %8 = fir.load %0 : !fir.ref<i32>
    return %8 : i32
  }
  func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F696D61785F312E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./imax_1.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}

// Just check that SOMETHING is being output.
// CHECK-LABEL @sum_1d_unknown() {
// CHECK: return

// -----

func.func @dot_f32(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "b"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductReal4(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  fir.store %5 to %0 : !fir.ref<f32>
  %6 = fir.load %0 : !fir.ref<f32>
  return %6 : f32
}
// CHECK-LABEL:   func.func @dot_f32(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "b"}) -> f32 {
// CHECK:           %[[RESLOC:.*]] = fir.alloca f32 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
// CHECK:           %[[ACAST:.*]] = fir.convert %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[BCAST:.*]] = fir.convert %[[B]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranADotProductReal4_f32_f32_simplified(%[[ACAST]], %[[BCAST]]) : (!fir.box<none>, !fir.box<none>) -> f32
// CHECK:           fir.store %[[RES]] to %[[RESLOC]] : !fir.ref<f32>
// CHECK:           %[[RET:.*]] = fir.load %[[RESLOC]] : !fir.ref<f32>
// CHECK:           return %[[RET]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranADotProductReal4_f32_f32_simplified(
// CHECK-SAME:      %[[A:.*]]: !fir.box<none>,
// CHECK-SAME:      %[[B:.*]]: !fir.box<none>) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[FZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[IZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[ACAST:.*]] = fir.convert %[[A]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[BCAST:.*]] = fir.convert %[[B]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ACAST]], %[[IZERO]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:           %[[IONE:.*]] = arith.constant 1 : index
// CHECK:           %[[LEN:.*]] = arith.subi %[[DIMS]]#1, %[[IONE]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[IDX:.*]] = %[[IZERO]] to %[[LEN]] step %[[IONE]] iter_args(%[[SUM:.*]] = %[[FZERO]]) -> (f32) {
// CHECK:             %[[ALOC:.*]] = fir.coordinate_of %[[ACAST]], %[[IDX]] : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[AVAL:.*]] = fir.load %[[ALOC]] : !fir.ref<f32>
// CHECK:             %[[AVALCAST:.*]] = fir.convert %[[AVAL]] : (f32) -> f32
// CHECK:             %[[BLOC:.*]] = fir.coordinate_of %[[BCAST]], %[[IDX]] : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[BVAL:.*]] = fir.load %[[BLOC]] : !fir.ref<f32>
// CHECK:             %[[BVALCAST:.*]] = fir.convert %[[BVAL]] : (f32) -> f32
// CHECK:             %[[MUL:.*]] = arith.mulf %[[AVALCAST]], %[[BVALCAST]] : f32
// CHECK:             %[[NEWSUM:.*]] = arith.addf %[[MUL]], %[[SUM]] : f32
// CHECK:             fir.result %[[NEWSUM]] : f32
// CHECK:           }
// CHECK:           return %[[RES]] : f32
// CHECK:         }

// -----

func.func @dot_f64(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "b"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductReal8(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f64
  fir.store %5 to %0 : !fir.ref<f64>
  %6 = fir.load %0 : !fir.ref<f64>
  return %6 : f64
}
func.func private @_FortranADotProductReal8(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f64 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// The same code handles all FP types, so just check that there is no
// call to runtime:
// CHECK-LABEL:   func.func @dot_f64(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "b"}) -> f64 {
// CHECK-NOT: call{{.*}}_FortranADotProductReal8(

// -----

func.func @dot_f80(%arg0: !fir.box<!fir.array<?xf80>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf80>> {fir.bindc_name = "b"}) -> f80 {
  %0 = fir.alloca f80 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xf80>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xf80>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductReal10(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f80
  fir.store %5 to %0 : !fir.ref<f80>
  %6 = fir.load %0 : !fir.ref<f80>
  return %6 : f80
}
func.func private @_FortranADotProductReal10(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f80 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// The same code handles all FP types, so just check that there is no
// call to runtime:
// CHECK-LABEL:   func.func @dot_f80(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xf80>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xf80>> {fir.bindc_name = "b"}) -> f80 {
// CHECK-NOT: call{{.*}}_FortranADotProductReal10(

// -----

func.func @dot_f128(%arg0: !fir.box<!fir.array<?xf128>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf128>> {fir.bindc_name = "b"}) -> f128 {
  %0 = fir.alloca f128 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xf128>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xf128>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductReal16(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f128
  fir.store %5 to %0 : !fir.ref<f128>
  %6 = fir.load %0 : !fir.ref<f128>
  return %6 : f128
}
func.func private @_FortranADotProductReal16(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f128 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// The same code handles all FP types, so just check that there is no
// call to runtime:
// CHECK-LABEL:   func.func @dot_f128(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xf128>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xf128>> {fir.bindc_name = "b"}) -> f128 {
// CHECK-NOT: call{{.*}}_FortranADotProductReal16(

// -----

func.func @dot_i32(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) -> i32 {
  %0 = fir.alloca i32 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductInteger4(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  fir.store %5 to %0 : !fir.ref<i32>
  %6 = fir.load %0 : !fir.ref<i32>
  return %6 : i32
}
func.func private @_FortranADotProductInteger4(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// CHECK-LABEL:   func.func @dot_i32(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}) -> i32 {
// CHECK:           %[[RESLOC:.*]] = fir.alloca i32 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
// CHECK:           %[[ACAST:.*]] = fir.convert %[[A]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[BCAST:.*]] = fir.convert %[[B]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranADotProductInteger4_i32_i32_simplified(%[[ACAST]], %[[BCAST]]) : (!fir.box<none>, !fir.box<none>) -> i32
// CHECK:           fir.store %[[RES]] to %[[RESLOC]] : !fir.ref<i32>
// CHECK:           %[[RET:.*]] = fir.load %[[RESLOC]] : !fir.ref<i32>
// CHECK:           return %[[RET]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranADotProductInteger4_i32_i32_simplified(
// CHECK-SAME:      %[[A:.*]]: !fir.box<none>,
// CHECK-SAME:      %[[B:.*]]: !fir.box<none>) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[I32ZERO:.*]] = arith.constant 0 : i32
// CHECK:           %[[IZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[ACAST:.*]] = fir.convert %[[A]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[BCAST:.*]] = fir.convert %[[B]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ACAST]], %[[IZERO]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[IONE:.*]] = arith.constant 1 : index
// CHECK:           %[[LEN:.*]] = arith.subi %[[DIMS]]#1, %[[IONE]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[IDX:.*]] = %[[IZERO]] to %[[LEN]] step %[[IONE]] iter_args(%[[SUM:.*]] = %[[I32ZERO]]) -> (i32) {
// CHECK:             %[[ALOC:.*]] = fir.coordinate_of %[[ACAST]], %[[IDX]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[AVAL:.*]] = fir.load %[[ALOC]] : !fir.ref<i32>
// CHECK:             %[[AVALCAST:.*]] = fir.convert %[[AVAL]] : (i32) -> i32
// CHECK:             %[[BLOC:.*]] = fir.coordinate_of %[[BCAST]], %[[IDX]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[BVAL:.*]] = fir.load %[[BLOC]] : !fir.ref<i32>
// CHECK:             %[[BVALCAST:.*]] = fir.convert %[[BVAL]] : (i32) -> i32
// CHECK:             %[[MUL:.*]] = arith.muli %[[AVALCAST]], %[[BVALCAST]] : i32
// CHECK:             %[[NEWSUM:.*]] = arith.addi %[[MUL]], %[[SUM]] : i32
// CHECK:             fir.result %[[NEWSUM]] : i32
// CHECK:           }
// CHECK:           return %[[RES]] : i32
// CHECK:         }

// -----

func.func @dot_i8(%arg0: !fir.box<!fir.array<?xi8>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi8>> {fir.bindc_name = "b"}) -> i8 {
  %0 = fir.alloca i8 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductInteger1(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i8
  fir.store %5 to %0 : !fir.ref<i8>
  %6 = fir.load %0 : !fir.ref<i8>
  return %6 : i8
}
func.func private @_FortranADotProductInteger1(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i8 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// The same code handles all integer types, so just check that there is no
// call to runtime:
// CHECK-LABEL:   func.func @dot_i8(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xi8>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xi8>> {fir.bindc_name = "b"}) -> i8 {
// CHECK-NOT: call{{.*}}_FortranADotProductInteger1(

// -----

func.func @dot_i16(%arg0: !fir.box<!fir.array<?xi16>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi16>> {fir.bindc_name = "b"}) -> i16 {
  %0 = fir.alloca i16 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xi16>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xi16>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductInteger2(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i16
  fir.store %5 to %0 : !fir.ref<i16>
  %6 = fir.load %0 : !fir.ref<i16>
  return %6 : i16
}
func.func private @_FortranADotProductInteger2(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i16 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// The same code handles all integer types, so just check that there is no
// call to runtime:
// CHECK-LABEL:   func.func @dot_i16(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xi16>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xi16>> {fir.bindc_name = "b"}) -> i16 {
// CHECK-NOT: call{{.*}}_FortranADotProductInteger2(

// -----

func.func @dot_i64(%arg0: !fir.box<!fir.array<?xi64>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi64>> {fir.bindc_name = "b"}) -> i64 {
  %0 = fir.alloca i64 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductInteger8(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i64
  fir.store %5 to %0 : !fir.ref<i64>
  %6 = fir.load %0 : !fir.ref<i64>
  return %6 : i64
}
func.func private @_FortranADotProductInteger8(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i64 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// The same code handles all integer types, so just check that there is no
// call to runtime:
// CHECK-LABEL:   func.func @dot_i64(
// CHECK-SAME:                      %[[A:.*]]: !fir.box<!fir.array<?xi64>> {fir.bindc_name = "a"},
// CHECK-SAME:                      %[[B:.*]]: !fir.box<!fir.array<?xi64>> {fir.bindc_name = "b"}) -> i64 {
// CHECK-NOT: call{{.*}}_FortranADotProductInteger8(

// -----

// Test mixed types, e.g. when _FortranADotProductReal8 is called
// with <?xf64> and <?xf32> arguments. The loaded elements must be converted
// to the result type REAL(8) before the computations.

func.func @dot_f64_f32(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "b"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductReal8(%2, %3, %4, %c3_i32) : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f64
  fir.store %5 to %0 : !fir.ref<f64>
  %6 = fir.load %0 : !fir.ref<f64>
  return %6 : f64
}
func.func private @_FortranADotProductReal4(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// CHECK-LABEL:   func.func @dot_f64_f32(
// CHECK-SAME:      %[[A:.*]]: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"},
// CHECK-SAME:      %[[B:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "b"}) -> f64 {
// CHECK:           %[[RESLOC:.*]] = fir.alloca f64 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
// CHECK:           %[[ACAST:.*]] = fir.convert %[[A]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:           %[[BCAST:.*]] = fir.convert %[[B]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranADotProductReal8_f64_f32_simplified(%[[ACAST]], %[[BCAST]]) : (!fir.box<none>, !fir.box<none>) -> f64
// CHECK:           fir.store %[[RES]] to %[[RESLOC]] : !fir.ref<f64>
// CHECK:           %[[RET:.*]] = fir.load %[[RESLOC]] : !fir.ref<f64>
// CHECK:           return %[[RET]] : f64
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranADotProductReal8_f64_f32_simplified(
// CHECK-SAME:      %[[A:.*]]: !fir.box<none>,
// CHECK-SAME:      %[[B:.*]]: !fir.box<none>) -> f64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[FZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[IZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[ACAST:.*]] = fir.convert %[[A]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[BCAST:.*]] = fir.convert %[[B]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ACAST]], %[[IZERO]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[IONE:.*]] = arith.constant 1 : index
// CHECK:           %[[LEN:.*]] = arith.subi %[[DIMS]]#1, %[[IONE]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[IDX:.*]] = %[[IZERO]] to %[[LEN]] step %[[IONE]] iter_args(%[[SUM:.*]] = %[[FZERO]]) -> (f64) {
// CHECK:             %[[ALOC:.*]] = fir.coordinate_of %[[ACAST]], %[[IDX]] : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:             %[[AVAL:.*]] = fir.load %[[ALOC]] : !fir.ref<f64>
// CHECK:             %[[AVALCAST:.*]] = fir.convert %[[AVAL]] : (f64) -> f64
// CHECK:             %[[BLOC:.*]] = fir.coordinate_of %[[BCAST]], %[[IDX]] : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[BVAL:.*]] = fir.load %[[BLOC]] : !fir.ref<f32>
// CHECK:             %[[BVALCAST:.*]] = fir.convert %[[BVAL]] : (f32) -> f64
// CHECK:             %[[MUL:.*]] = arith.mulf %[[AVALCAST]], %[[BVALCAST]] : f64
// CHECK:             %[[NEWSUM:.*]] = arith.addf %[[MUL]], %[[SUM]] : f64
// CHECK:             fir.result %[[NEWSUM]] : f64
// CHECK:           }
// CHECK:           return %[[RES]] : f64
// CHECK:         }


// -----

// Call to MAXVAL with 1D I32 array is replaced.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @maxval_1d_array_int(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> i32 {
    %c10 = arith.constant 10 : index
    %0 = fir.alloca i32 {bindc_name = "test_max_2", uniq_name = "_QFtest_max_2Etest_max_2"}
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F696D61785F322E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranAMaxvalInteger4(%5, %6, %c5_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
    fir.store %9 to %0 : !fir.ref<i32>
    %10 = fir.load %0 : !fir.ref<i32>
    return %10 : i32
  }
  func.func private @_FortranAMaxvalInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F696D61785F322E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./imax_2.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}

// CHECK-LABEL:   func.func @maxval_1d_array_int(
// CHECK-SAME:                             %[[A:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> i32 {
// CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_I32:.*]] = fir.embox %[[A]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_I32]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranAMaxvalInteger4x1_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> i32
// CHECK:           return %{{.*}} : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAMaxvalInteger4x1_simplified(
// CHECK-SAME:                                                       %[[ARR:.*]]: !fir.box<none>) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[CI32_MININT:.*]] = arith.constant -2147483648 : i32
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[MAX:.*]] = %[[CI32_MININT]]) -> (i32) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:             %[[NEW_MAX:.*]] = arith.maxsi %[[ITEM_VAL]], %[[MAX]] : i32
// CHECK:             fir.result %[[NEW_MAX]] : i32
// CHECK:           }
// CHECK:           return %[[RES]] : i32
// CHECK:         }

// -----

// Call to MAXVAL with 1D F64 is replaced.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @maxval_1d_real(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}) -> f64 {
    %c10 = arith.constant 10 : index
    %0 = fir.alloca f64 {bindc_name = "maxval_1d_real", uniq_name = "_QFmaxval_1d_realEmaxval_1d_real"}
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
    %3 = fir.absent !fir.box<i1>
    %c0 = arith.constant 0 : index
    %4 = fir.address_of(@_QQcl.2E2F6973756D5F352E66393000) : !fir.ref<!fir.char<1,13>>
    %c5_i32 = arith.constant 5 : i32
    %5 = fir.convert %2 : (!fir.box<!fir.array<10xf64>>) -> !fir.box<none>
    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
    %7 = fir.convert %c0 : (index) -> i32
    %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
    %9 = fir.call @_FortranAMaxvalReal8(%5, %6, %c5_i32, %7, %8) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f64
    fir.store %9 to %0 : !fir.ref<f64>
    %10 = fir.load %0 : !fir.ref<f64>
    return %10 : f64
  }
  func.func private @_FortranAMaxvalReal8(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f64 attributes {fir.runtime}
  fir.global linkonce @_QQcl.2E2F6973756D5F352E66393000 constant : !fir.char<1,13> {
    %0 = fir.string_lit "./imaxval_5.f90\00"(13) : !fir.char<1,13>
    fir.has_value %0 : !fir.char<1,13>
  }
}


// CHECK-LABEL:   func.func @maxval_1d_real(
// CHECK-SAME:                           %[[A:.*]]: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}) -> f64 {
// CHECK:           %[[CINDEX_10:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[CINDEX_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_F64:.*]] = fir.embox %[[A]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_F64]] : (!fir.box<!fir.array<10xf64>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranAMaxvalReal8x1_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> f64
// CHECK:           return %{{.*}} : f64
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAMaxvalReal8x1_simplified(
// CHECK-SAME:                                                    %[[ARR:.*]]: !fir.box<none>) -> f64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_F64:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[NEG_DBL_MAX:.*]] = arith.constant -1.7976931348623157E+308 : f64
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_F64]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[MAX]] = %[[NEG_DBL_MAX]]) -> (f64) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_F64]], %[[ITER]] : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<f64>
// CHECK:             %[[CMP:.*]] = arith.cmpf ogt, %[[ITEM_VAL]], %[[MAX]] : f64
// CHECK:             %[[NEW_MAX:.*]] = arith.select %[[CMP]], %[[ITEM_VAL]], %[[MAX]] : f64
// CHECK:             fir.result %[[NEW_MAX]] : f64
// CHECK:           }
// CHECK:           return %[[RES]] : f64
// CHECK:         }

// -----

// SUM reduction of sliced explicit-shape array is replaced with
// 2D simplified implementation.
func.func @sum_sliced_embox_i64(%arg0: !fir.ref<!fir.array<10x10x10xi64>> {fir.bindc_name = "a"}) -> f32 {
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  %c10_1 = arith.constant 10 : index
  %0 = fir.alloca f32 {bindc_name = "sum_sliced_embox_i64", uniq_name = "_QFsum_sliced_embox_i64Esum_sliced_embox_i64"}
  %1 = fir.alloca i64 {bindc_name = "sum_sliced_i64", uniq_name = "_QFsum_sliced_embox_i64Esum_sliced_i64"}
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %2 = fir.convert %c1_i64 : (i64) -> index
  %3 = arith.addi %c1, %c10 : index
  %4 = arith.subi %3, %c1 : index
  %c1_i64_2 = arith.constant 1 : i64
  %5 = fir.convert %c1_i64_2 : (i64) -> index
  %6 = arith.addi %c1, %c10_0 : index
  %7 = arith.subi %6, %c1 : index
  %c1_i64_3 = arith.constant 1 : i64
  %8 = fir.undefined index
  %9 = fir.shape %c10, %c10_0, %c10_1 : (index, index, index) -> !fir.shape<3>
  %10 = fir.slice %c1, %4, %2, %c1, %7, %5, %c1_i64_3, %8, %8 : (index, index, index, index, index, index, i64, index, index) -> !fir.slice<3>
  %11 = fir.embox %arg0(%9) [%10] : (!fir.ref<!fir.array<10x10x10xi64>>, !fir.shape<3>, !fir.slice<3>) -> !fir.box<!fir.array<?x?xi64>>
  %12 = fir.absent !fir.box<i1>
  %c0 = arith.constant 0 : index
  %13 = fir.address_of(@_QQcl.2E2F746573742E66393000) : !fir.ref<!fir.char<1,11>>
  %c3_i32 = arith.constant 3 : i32
  %14 = fir.convert %11 : (!fir.box<!fir.array<?x?xi64>>) -> !fir.box<none>
  %15 = fir.convert %13 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
  %16 = fir.convert %c0 : (index) -> i32
  %17 = fir.convert %12 : (!fir.box<i1>) -> !fir.box<none>
  %18 = fir.call @_FortranASumInteger8(%14, %15, %c3_i32, %16, %17) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i64
  fir.store %18 to %1 : !fir.ref<i64>
  %19 = fir.load %0 : !fir.ref<f32>
  return %19 : f32
}
func.func private @_FortranASumInteger8(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i64 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F746573742E66393000 constant : !fir.char<1,11> {
  %0 = fir.string_lit "./test.f90\00"(11) : !fir.char<1,11>
  fir.has_value %0 : !fir.char<1,11>
}

// CHECK-NOT: call{{.*}}_FortranASumInteger8(
// CHECK: call @_FortranASumInteger8x2_simplified(
// CHECK-NOT: call{{.*}}_FortranASumInteger8(

// -----

// SUM reduction of sliced assumed-shape array is replaced with
// 2D simplified implementation.
func.func @_QPsum_sliced_rebox_i64(%arg0: !fir.box<!fir.array<?x?x?xi64>> {fir.bindc_name = "a"}) -> f32 {
  %0 = fir.alloca i64 {bindc_name = "sum_sliced_i64", uniq_name = "_QFsum_sliced_rebox_i64Esum_sliced_i64"}
  %1 = fir.alloca f32 {bindc_name = "sum_sliced_rebox_i64", uniq_name = "_QFsum_sliced_rebox_i64Esum_sliced_rebox_i64"}
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %2 = fir.convert %c1_i64 : (i64) -> index
  %c0 = arith.constant 0 : index
  %3:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x?x?xi64>>, index) -> (index, index, index)
  %4 = arith.addi %c1, %3#1 : index
  %5 = arith.subi %4, %c1 : index
  %c1_i64_0 = arith.constant 1 : i64
  %6 = fir.convert %c1_i64_0 : (i64) -> index
  %c1_1 = arith.constant 1 : index
  %7:3 = fir.box_dims %arg0, %c1_1 : (!fir.box<!fir.array<?x?x?xi64>>, index) -> (index, index, index)
  %8 = arith.addi %c1, %7#1 : index
  %9 = arith.subi %8, %c1 : index
  %c1_i64_2 = arith.constant 1 : i64
  %10 = fir.undefined index
  %11 = fir.slice %c1, %5, %2, %c1, %9, %6, %c1_i64_2, %10, %10 : (index, index, index, index, index, index, i64, index, index) -> !fir.slice<3>
  %12 = fir.rebox %arg0 [%11] : (!fir.box<!fir.array<?x?x?xi64>>, !fir.slice<3>) -> !fir.box<!fir.array<?x?xi64>>
  %13 = fir.absent !fir.box<i1>
  %c0_3 = arith.constant 0 : index
  %14 = fir.address_of(@_QQcl.2E2F746573742E66393000) : !fir.ref<!fir.char<1,11>>
  %c8_i32 = arith.constant 8 : i32
  %15 = fir.convert %12 : (!fir.box<!fir.array<?x?xi64>>) -> !fir.box<none>
  %16 = fir.convert %14 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
  %17 = fir.convert %c0_3 : (index) -> i32
  %18 = fir.convert %13 : (!fir.box<i1>) -> !fir.box<none>
  %19 = fir.call @_FortranASumInteger8(%15, %16, %c8_i32, %17, %18) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i64
  fir.store %19 to %0 : !fir.ref<i64>
  %20 = fir.load %1 : !fir.ref<f32>
  return %20 : f32
}
func.func private @_FortranASumInteger8(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i64 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F746573742E66393000 constant : !fir.char<1,11> {
  %0 = fir.string_lit "./test.f90\00"(11) : !fir.char<1,11>
  fir.has_value %0 : !fir.char<1,11>
}

// CHECK-NOT: call{{.*}}_FortranASumInteger8(
// CHECK: call @_FortranASumInteger8x2_simplified(
// CHECK-NOT: call{{.*}}_FortranASumInteger8(

// -----

func.func @dot_f32_contract_reassoc(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "b"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductReal4(%2, %3, %4, %c3_i32) fastmath<contract,reassoc> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  fir.store %5 to %0 : !fir.ref<f32>
  %6 = fir.load %0 : !fir.ref<f32>
  return %6 : f32
}

func.func @dot_f32_fast(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "b"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "dot", uniq_name = "_QFdotEdot"}
  %1 = fir.address_of(@_QQcl.2E2F646F742E66393000) : !fir.ref<!fir.char<1,10>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  %3 = fir.convert %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  %4 = fir.convert %1 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  %5 = fir.call @_FortranADotProductReal4(%2, %3, %4, %c3_i32) fastmath<fast> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  fir.store %5 to %0 : !fir.ref<f32>
  %6 = fir.load %0 : !fir.ref<f32>
  return %6 : f32
}

func.func private @_FortranADotProductReal4(!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F646F742E66393000 constant : !fir.char<1,10> {
  %0 = fir.string_lit "./dot.f90\00"(10) : !fir.char<1,10>
  fir.has_value %0 : !fir.char<1,10>
}

// CHECK-LABEL: @dot_f32_contract_reassoc
// CHECK: fir.call @_FortranADotProductReal4_reassoc_contract_f32_f32_simplified(%2, %3) fastmath<reassoc,contract>
// CHECK-LABEL: @dot_f32_fast
// CHECK: fir.call @_FortranADotProductReal4_fast_f32_f32_simplified(%2, %3) fastmath<fast>
// CHECK-LABEL: func.func private @_FortranADotProductReal4_reassoc_contract_f32_f32_simplified
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f32
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f32
// CHECK-LABEL: func.func private @_FortranADotProductReal4_fast_f32_f32_simplified
// CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f32
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f32

// -----

func.func @sum_1d_real_contract_reassoc(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}) -> f64 {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca f64 {bindc_name = "sum_1d_real", uniq_name = "_QFsum_1d_realEsum_1d_real"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
  %3 = fir.absent !fir.box<i1>
  %c0 = arith.constant 0 : index
  %4 = fir.address_of(@_QQcl.2E2F6973756D5F352E66393000) : !fir.ref<!fir.char<1,13>>
  %c5_i32 = arith.constant 5 : i32
  %5 = fir.convert %2 : (!fir.box<!fir.array<10xf64>>) -> !fir.box<none>
  %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
  %7 = fir.convert %c0 : (index) -> i32
  %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
  %9 = fir.call @_FortranASumReal8(%5, %6, %c5_i32, %7, %8) fastmath<contract,reassoc> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f64
  fir.store %9 to %0 : !fir.ref<f64>
  %10 = fir.load %0 : !fir.ref<f64>
  return %10 : f64
}

func.func @sum_1d_real_fast(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}) -> f64 {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca f64 {bindc_name = "sum_1d_real", uniq_name = "_QFsum_1d_realEsum_1d_real"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
  %3 = fir.absent !fir.box<i1>
  %c0 = arith.constant 0 : index
  %4 = fir.address_of(@_QQcl.2E2F6973756D5F352E66393000) : !fir.ref<!fir.char<1,13>>
  %c5_i32 = arith.constant 5 : i32
  %5 = fir.convert %2 : (!fir.box<!fir.array<10xf64>>) -> !fir.box<none>
  %6 = fir.convert %4 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
  %7 = fir.convert %c0 : (index) -> i32
  %8 = fir.convert %3 : (!fir.box<i1>) -> !fir.box<none>
  %9 = fir.call @_FortranASumReal8(%5, %6, %c5_i32, %7, %8) fastmath<fast> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f64
  fir.store %9 to %0 : !fir.ref<f64>
  %10 = fir.load %0 : !fir.ref<f64>
  return %10 : f64
}

func.func private @_FortranASumReal8(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f64 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F6973756D5F352E66393000 constant : !fir.char<1,13> {
  %0 = fir.string_lit "./isum_5.f90\00"(13) : !fir.char<1,13>
  fir.has_value %0 : !fir.char<1,13>
}

// CHECK-LABEL: @sum_1d_real_contract_reassoc
// CHECK: fir.call @_FortranASumReal8x1_reassoc_contract_simplified(%5) fastmath<reassoc,contract>
// CHECK-LABEL: @sum_1d_real_fast
// CHECK: fir.call @_FortranASumReal8x1_fast_simplified(%5) fastmath<fast>
// CHECK-LABEL: func.func private @_FortranASumReal8x1_reassoc_contract_simplified
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc,contract> : f64
// CHECK-LABEL: func.func private @_FortranASumReal8x1_fast_simplified
// CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64

// -----
// Ensure count is simplified in valid case

func.func @_QMtestPcount_generate_mask(%arg0: !fir.ref<f32> {fir.bindc_name = "a"}) -> i32 {
  %0 = fir.alloca i32 {bindc_name = "count_generate_mask", uniq_name = "_QMtestFcount_generate_maskEcount_generate_mask"}
  %c10 = arith.constant 10 : index
  %1 = fir.alloca !fir.array<10x!fir.logical<4>> {bindc_name = "mask", uniq_name = "_QMtestFcount_generate_maskEmask"}
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
  %c0 = arith.constant 0 : index
  %4 = fir.address_of(@_QQcl.2E2F746573746661696C2E66393000) : !fir.ref<!fir.char<1,15>>
  %c10_i32 = arith.constant 10 : i32
  %5 = fir.convert %3 : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
  %6 = fir.convert %4 : (!fir.ref<!fir.char<1,15>>) -> !fir.ref<i8>
  %7 = fir.convert %c0 : (index) -> i32
  %8 = fir.call @_FortranACount(%5, %6, %c10_i32, %7) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64
  %9 = fir.convert %8 : (i64) -> i32
  fir.store %9 to %0 : !fir.ref<i32>
  %10 = fir.load %0 : !fir.ref<i32>
  return %10 : i32
}
func.func private @_FortranACount(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64 attributes {fir.runtime}
fir.global linkonce @_QQcl.2E2F746573746661696C2E66393000 constant : !fir.char<1,15> {
  %0 = fir.string_lit "./test.f90\00"(15) : !fir.char<1,15>
  fir.has_value %0 : !fir.char<1,15>
}

// CHECK-LABEL:   func.func @_QMtestPcount_generate_mask(
// CHECK-SAME:                                           %[[A:.*]]: !fir.ref<f32> {fir.bindc_name = "a"}) -> i32 {
// CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_LOGICAL:.*]] = fir.embox %{{.*}}(%[[SHAPE]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_LOGICAL]] : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
// CHECK-NOT:       fir.call @_FortranACount({{.*}})
// CHECK:           %[[RES:.*]] = fir.call @_FortranACountLogical4x1_simplified(%[[A_BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> i64
// CHECK-NOT:       fir.call @_FortranACount({{.*}})
// CHECK:           return %{{.*}} : i32
// CHECK:         }
// CHECK:         func.func private @_FortranACount(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64 attributes {fir.runtime}

// CHECK-LABEL:   func.func private @_FortranACountLogical4x1_simplified(
// CHECK-SAME:                                                            %[[ARR:.*]]: !fir.box<none>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[IZERO:.*]] = arith.constant 0 : i64
// CHECK:           %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[C_INDEX0]] to %[[EXTENT]] step %[[C_INDEX1]] iter_args(%[[COUNT:.*]] = %[[IZERO]]) -> (i64) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:             %[[I32_0:.*]] = arith.constant 0 : i32
// CHECK:             %[[I64_0:.*]] = arith.constant 0 : i64
// CHECK:             %[[I64_1:.*]] = arith.constant 1 : i64
// CHECK:             %[[CMP:.*]] = arith.cmpi eq, %[[ITEM_VAL]], %[[I32_0]] : i32
// CHECK:             %[[SELECT:.*]] = arith.select %[[CMP]], %[[I64_0]], %[[I64_1]] : i64
// CHECK:             %[[NEW_COUNT:.*]] = arith.addi %[[SELECT]], %[[COUNT]] : i64
// CHECK:             fir.result %[[NEW_COUNT]] : i64
// CHECK:           }
// CHECK:           return %[[RES:.*]] : i64
// CHECK:         }

// -----
// Ensure count is properly simplified for different mask kind

func.func @_QPdiffkind(%arg0: !fir.ref<!fir.array<10x!fir.logical<2>>> {fir.bindc_name = "mask"}) -> i32 {
  %0 = fir.alloca i32 {bindc_name = "diffkind", uniq_name = "_QFdiffkindEdiffkind"}
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10x!fir.logical<2>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<2>>>
  %c0 = arith.constant 0 : index
  %3 = fir.address_of(@_QQcl.916d74b25894ddf7881ff7f913a677f5) : !fir.ref<!fir.char<1,52>>
  %c5_i32 = arith.constant 5 : i32
  %4 = fir.convert %2 : (!fir.box<!fir.array<10x!fir.logical<2>>>) -> !fir.box<none>
  %5 = fir.convert %3 : (!fir.ref<!fir.char<1,52>>) -> !fir.ref<i8>
  %6 = fir.convert %c0 : (index) -> i32
  %7 = fir.call @_FortranACount(%4, %5, %c5_i32, %6) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64
  %8 = fir.convert %7 : (i64) -> i32
  fir.store %8 to %0 : !fir.ref<i32>
  %9 = fir.load %0 : !fir.ref<i32>
  return %9 : i32
}

// CHECK-LABEL:   func.func @_QPdiffkind(
// CHECK-SAME:                           %[[A:.*]]: !fir.ref<!fir.array<10x!fir.logical<2>>> {fir.bindc_name = "mask"}) -> i32 {
// CHECK:           %[[res:.*]] = fir.call @_FortranACountLogical2x1_simplified({{.*}}) fastmath<contract> : (!fir.box<none>) -> i64

// CHECK-LABEL:   func.func private @_FortranACountLogical2x1_simplified(
// CHECK-SAME:                                                           %[[ARR:.*]]: !fir.box<none>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I16:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi16>>
// CHECK:           %[[IZERO:.*]] = arith.constant 0 : i64
// CHECK:           %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMIDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I16]], %[[DIMIDX_0]] : (!fir.box<!fir.array<?xi16>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[C_INDEX0]] to %[[EXTENT]] step %[[C_INDEX1]] iter_args(%[[COUNT:.*]] = %[[IZERO]]) -> (i64) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_I16]], %[[ITER]] : (!fir.box<!fir.array<?xi16>>, index) -> !fir.ref<i16>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i16>
// CHECK:             %[[I16_0:.*]] = arith.constant 0 : i16
// CHECK:             %[[I64_0:.*]] = arith.constant 0 : i64
// CHECK:             %[[I64_1:.*]] = arith.constant 1 : i64
// CHECK:             %[[CMP:.*]] = arith.cmpi eq, %[[ITEM_VAL]], %[[I16_0]] : i16
// CHECK:             %[[SELECT:.*]] = arith.select %[[CMP]], %[[I64_0]], %[[I64_1]] : i64
// CHECK:             %[[NEW_COUNT:.*]] = arith.addi %[[SELECT]], %[[COUNT]] : i64
// CHECK:             fir.result %[[NEW_COUNT]] : i64
// CHECK:           }
// CHECK:           return %[[RES:.*]] : i64
// CHECK:         }

// -----
// Ensure count isn't simplified when given dim argument

func.func @_QMtestPcount_generate_mask(%arg0: !fir.ref<!fir.array<10x10x!fir.logical<4>>> {fir.bindc_name = "mask"}) -> !fir.array<10xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  %c10_1 = arith.constant 10 : index
  %1 = fir.alloca !fir.array<10xi32> {bindc_name = "res", uniq_name = "_QMtestFcount_generate_maskEres"}
  %2 = fir.shape %c10_1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
  %c2_i32 = arith.constant 2 : i32
  %4 = fir.shape %c10, %c10_0 : (index, index) -> !fir.shape<2>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10x10x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10x!fir.logical<4>>>
  %c4 = arith.constant 4 : index
  %6 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %7 = fir.shape %c0 : (index) -> !fir.shape<1>
  %8 = fir.embox %6(%7) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %8 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %9 = fir.address_of(@_QQcl.2E2F746573746661696C2E66393000) : !fir.ref<!fir.char<1,15>>
  %c11_i32 = arith.constant 11 : i32
  %10 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %11 = fir.convert %5 : (!fir.box<!fir.array<10x10x!fir.logical<4>>>) -> !fir.box<none>
  %12 = fir.convert %c4 : (index) -> i32
  %13 = fir.convert %9 : (!fir.ref<!fir.char<1,15>>) -> !fir.ref<i8>
  %14 = fir.call @_FortranACountDim(%10, %11, %c2_i32, %12, %13, %c11_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none
  %15 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_2 = arith.constant 0 : index
  %16:3 = fir.box_dims %15, %c0_2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %17 = fir.box_addr %15 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %18 = fir.shape_shift %16#0, %16#1 : (index, index) -> !fir.shapeshift<1>
  %19 = fir.array_load %17(%18) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  %c1 = arith.constant 1 : index
  %c0_3 = arith.constant 0 : index
  %20 = arith.subi %c10_1, %c1 : index
  %21 = fir.do_loop %arg1 = %c0_3 to %20 step %c1 unordered iter_args(%arg2 = %3) -> (!fir.array<10xi32>) {
    %23 = fir.array_fetch %19, %arg1 : (!fir.array<?xi32>, index) -> i32
    %24 = fir.array_update %arg2, %23, %arg1 : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
    fir.result %24 : !fir.array<10xi32>
  }
  fir.array_merge_store %3, %21 to %1 : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
  fir.freemem %17 : !fir.heap<!fir.array<?xi32>>
  %22 = fir.load %1 : !fir.ref<!fir.array<10xi32>>
  return %22 : !fir.array<10xi32>
}
func.func private @_FortranACountDim(!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QMtestPcount_generate_mask(
// CHECK-SAME:                                           %[[A:.*]]: !fir.ref<!fir.array<10x10x!fir.logical<4>>> {fir.bindc_name = "mask"}) -> !fir.array<10xi32> {
// CHECK-NOT        fir.call @_FortranACountDimLogical4_simplified({{.*}})
// CHECK:           %[[RES:.*]] = fir.call @_FortranACountDim({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none
// CHECK-NOT        fir.call @_FortranACountDimLogical4_simplified({{.*}})

// -----
// Ensure count isn't simplified for unknown dimension arrays

func.func @_QPmc(%arg0: !fir.box<!fir.array<?x?x?x!fir.logical<4>>> {fir.bindc_name = "m"}) -> i32 {
  %0 = fir.alloca i32 {bindc_name = "mc", uniq_name = "_QFmcEmc"}
  %c0 = arith.constant 0 : index
  %1 = fir.address_of(@_QQcl.95529933117f47914fc21e220c1a0896) : !fir.ref<!fir.char<1,58>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>) -> !fir.box<none>
  %3 = fir.convert %1 : (!fir.ref<!fir.char<1,58>>) -> !fir.ref<i8>
  %4 = fir.convert %c0 : (index) -> i32
  %5 = fir.call @_FortranACount(%2, %3, %c3_i32, %4) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64
  %6 = fir.convert %5 : (i64) -> i32
  fir.store %6 to %0 : !fir.ref<i32>
  %7 = fir.load %0 : !fir.ref<i32>
  return %7 : i32
}
func.func private @_FortranACount(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64 attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPmc(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?x!fir.logical<4>>> {fir.bindc_name = "m"}) -> i32 {
// CHECK-NOT        fir.call @_FortranACountLogical4_simplified({{.*}})
// CHECK:           %[[RES:.*]] = fir.call @_FortranACount({{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64
// CHECK-NOT        fir.call @_FortranACountLogical4_simplified({{.*}})

// -----
// Ensure Any is simplified in correct usage

func.func @_QPtestAny_NoDimArg(%arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.logical<4> {bindc_name = "testAny_NoDimArg", uniq_name = "_QFtestAny_NoDimArgEtestAny_NoDimArg"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
  %c1 = arith.constant 1 : index
  %3 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
  %c3_i32 = arith.constant 3 : i32
  %4 = fir.convert %2 : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
  %5 = fir.convert %3 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %6 = fir.convert %c1 : (index) -> i32
  %7 = fir.call @_FortranAAny(%4, %5, %c3_i32, %6) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
  %8 = fir.convert %7 : (i1) -> !fir.logical<4>
  fir.store %8 to %0 : !fir.ref<!fir.logical<4>>
  %9 = fir.load %0 : !fir.ref<!fir.logical<4>>
  return %9 : !fir.logical<4>
}
func.func private @_FortranAAny(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1 attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPtestAny_NoDimArg(
// CHECK-SAME:                          %[[ARR:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
// CHECK:           %[[SIZE:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[SIZE]] : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_LOGICAL:.*]] = fir.embox %[[ARR]](%[[SHAPE]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_LOGICAL]] : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranAAnyLogical4x1_simplified(%[[A_BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> i1
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAAnyLogical4x1_simplified(
// CHECK-SAME:                                                 %[[ARR:.*]]: !fir.box<none>) -> i1 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[INIT_COND:.*]] = arith.constant true
// CHECK:           %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[A_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_IDX0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[A_BOX_I32]], %[[DIM_IDX0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[RES:.*]]:2 = fir.iterate_while (%[[ITER:.*]] = %[[C_INDEX0]] to %[[EXTENT]] step %[[C_INDEX1]]) and (%[[OK:.*]] = %[[INIT_COND]]) iter_args(%[[INIT:.*]] = %[[FALSE]]) -> (i1) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[A_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:             %[[I32_0:.*]] = arith.constant 0 : i32
// CHECK:             %[[CMP:.*]] = arith.cmpi ne, %[[ITEM_VAL]], %[[I32_0]] : i32
// CHECK:             %[[I1_1:.*]] = arith.constant true
// CHECK:             %[[CONTINUE:.*]] = arith.xori %[[CMP]], %[[I1_1]] : i1
// CHECK:             fir.result %[[CONTINUE]], %[[CMP]] : i1, i1
// CHECK:           }
// CHECK:           return %[[RES:.*]]#1 : i1
// CHECK:         }

// -----
// Ensure Any is simpified correctly for different kind logical

func.func @_QPtestAny_NoDimArgLogical8(%arg0: !fir.ref<!fir.array<10x!fir.logical<8>>> {fir.bindc_name = "a"}) -> !fir.logical<8> {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.logical<8> {bindc_name = "testAny_NoDimArgLogical8", uniq_name = "_QFtestAny_NoDimArgLogical8EtestAny_NoDimArgLogical8"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10x!fir.logical<8>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<8>>>
  %c1 = arith.constant 1 : index
  %3 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
  %c3_i32 = arith.constant 3 : i32
  %4 = fir.convert %2 : (!fir.box<!fir.array<10x!fir.logical<8>>>) -> !fir.box<none>
  %5 = fir.convert %3 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %6 = fir.convert %c1 : (index) -> i32
  %7 = fir.call @_FortranAAny(%4, %5, %c3_i32, %6) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
  %8 = fir.convert %7 : (i1) -> !fir.logical<8>
  fir.store %8 to %0 : !fir.ref<!fir.logical<8>>
  %9 = fir.load %0 : !fir.ref<!fir.logical<8>>
  return %9 : !fir.logical<8>
}
func.func private @_FortranAAny(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1 attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPtestAny_NoDimArgLogical8(
// CHECK-SAME:                          %[[ARR:.*]]: !fir.ref<!fir.array<10x!fir.logical<8>>> {fir.bindc_name = "a"}) -> !fir.logical<8> {
// CHECK:           %[[SIZE:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[SIZE]] : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_LOGICAL:.*]] = fir.embox %[[ARR]](%[[SHAPE]]) : (!fir.ref<!fir.array<10x!fir.logical<8>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<8>>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_LOGICAL]] : (!fir.box<!fir.array<10x!fir.logical<8>>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranAAnyLogical8x1_simplified(%[[A_BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> i1
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAAnyLogical8x1_simplified(
// CHECK-SAME:                                                 %[[ARR:.*]]: !fir.box<none>) -> i1 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[INIT_COND:.*]] = arith.constant true
// CHECK:           %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[A_BOX_I64:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi64>>
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_IDX0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[A_BOX_I64]], %[[DIM_IDX0]] : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[RES:.*]]:2 = fir.iterate_while (%[[ITER:.*]] = %[[C_INDEX0]] to %[[EXTENT]] step %[[C_INDEX1]]) and (%[[OK:.*]] = %[[INIT_COND]]) iter_args(%[[INIT:.*]] = %[[FALSE]]) -> (i1) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[A_BOX_I64]], %[[ITER]] : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i64>
// CHECK:             %[[I64_0:.*]] = arith.constant 0 : i64
// CHECK:             %[[CMP:.*]] = arith.cmpi ne, %[[ITEM_VAL]], %[[I64_0]] : i64
// CHECK:             %[[I1_1:.*]] = arith.constant true
// CHECK:             %[[CONTINUE:.*]] = arith.xori %[[CMP]], %[[I1_1]] : i1
// CHECK:             fir.result %[[CONTINUE]], %[[CMP]] : i1, i1
// CHECK:           }
// CHECK:           return %[[RES:.*]]#1 : i1
// CHECK:         }

// -----
// Ensure Any is not simplified when call ends in 'Dim'

func.func @_QPtestAny_DimArg(%arg0: !fir.ref<!fir.array<10x10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.array<10x!fir.logical<4>> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  %c10_1 = arith.constant 10 : index
  %1 = fir.alloca !fir.array<10x!fir.logical<4>> {bindc_name = "testAny_DimArg", uniq_name = "_QFtestAny_DimArgEtestAny_DimArg"}
  %2 = fir.shape %c10_1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  %c2_i32 = arith.constant 2 : i32
  %4 = fir.shape %c10, %c10_0 : (index, index) -> !fir.shape<2>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10x10x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10x!fir.logical<4>>>
  %6 = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
  %c0 = arith.constant 0 : index
  %7 = fir.shape %c0 : (index) -> !fir.shape<1>
  %8 = fir.embox %6(%7) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  fir.store %8 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
  %9 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
  %c3_i32 = arith.constant 3 : i32
  %10 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
  %11 = fir.convert %5 : (!fir.box<!fir.array<10x10x!fir.logical<4>>>) -> !fir.box<none>
  %12 = fir.convert %9 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %13 = fir.call @_FortranAAnyDim(%10, %11, %c2_i32, %12, %c3_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  %14 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
  %c0_2 = arith.constant 0 : index
  %15:3 = fir.box_dims %14, %c0_2 : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
  %16 = fir.box_addr %14 : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
  %17 = fir.shape_shift %15#0, %15#1 : (index, index) -> !fir.shapeshift<1>
  %18 = fir.array_load %16(%17) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<4>>
  %c1 = arith.constant 1 : index
  %c0_3 = arith.constant 0 : index
  %19 = arith.subi %c10_1, %c1 : index
  %20 = fir.do_loop %arg1 = %c0_3 to %19 step %c1 unordered iter_args(%arg2 = %3) -> (!fir.array<10x!fir.logical<4>>) {
    %22 = fir.array_fetch %18, %arg1 : (!fir.array<?x!fir.logical<4>>, index) -> !fir.logical<4>
    %23 = fir.array_update %arg2, %22, %arg1 : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
    fir.result %23 : !fir.array<10x!fir.logical<4>>
  }
  fir.array_merge_store %3, %20 to %1 : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
  fir.freemem %16 : !fir.heap<!fir.array<?x!fir.logical<4>>>
  %21 = fir.load %1 : !fir.ref<!fir.array<10x!fir.logical<4>>>
  return %21 : !fir.array<10x!fir.logical<4>>
}
func.func private @_FortranAAnyDim(!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPtestAny_DimArg(
// CHECK-SAME:                          %[[ARR:.*]]: !fir.ref<!fir.array<10x10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.array<10x!fir.logical<4>> {
// CHECK-NOT        fir.call @_FortranAAnyDimLogical4x1_simplified({{.*}})
// CHECK:           fir.call @_FortranAAnyDim({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
// CHECK-NOT        fir.call @_FortranAAnyDimLogical4x1_simplified({{.*}})

// -----
// Ensure Any is not simplified for unknown dimension arrays

func.func @_QPtestAny_UnknownDim(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
  %0 = fir.alloca !fir.logical<4> {bindc_name = "testAny_UnknownDim", uniq_name = "_QFtestAny_UnknownDimEtestAny_UnknownDim"}
  %c1 = arith.constant 1 : index
  %1 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
  %c3_i32 = arith.constant 3 : i32
  %2 = fir.convert %arg0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  %3 = fir.convert %1 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %4 = fir.convert %c1 : (index) -> i32
  %5 = fir.call @_FortranAAny(%2, %3, %c3_i32, %4) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
  %6 = fir.convert %5 : (i1) -> !fir.logical<4>
  fir.store %6 to %0 : !fir.ref<!fir.logical<4>>
  %7 = fir.load %0 : !fir.ref<!fir.logical<4>>
  return %7 : !fir.logical<4>
}
func.func private @_FortranAAny(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1 attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPtestAny_UnknownDim(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
// CHECK-NOT        fir.call @_FortranAAnyLogical4x1_simplified({{.*}})
// CHECK:           fir.call @_FortranAAny({{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
// CHECK-NOT        fir.call @_FortranAAnyLogical4x1_simplified({{.*}})

// -----
// Check that multi-rank Any cases are properly simplified

func.func @_QPtestAny_2D(%arg0: !fir.ref<!fir.array<10x0x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %0 = fir.alloca !fir.logical<4> {bindc_name = "testAny_2D", uniq_name = "_QFtestAny_2DEtestAny_2D"}
    %1 = fir.shape %c10, %c0 : (index, index) -> !fir.shape<2>
    %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10x0x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<10x0x!fir.logical<4>>>
    %c1 = arith.constant 1 : index
    %3 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
    %c3_i32 = arith.constant 3 : i32
    %4 = fir.convert %2 : (!fir.box<!fir.array<10x0x!fir.logical<4>>>) -> !fir.box<none>
    %5 = fir.convert %3 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
    %6 = fir.convert %c1 : (index) -> i32
    %7 = fir.call @_FortranAAny(%4, %5, %c3_i32, %6) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
    %8 = fir.convert %7 : (i1) -> !fir.logical<4>
    fir.store %8 to %0 : !fir.ref<!fir.logical<4>>
    %9 = fir.load %0 : !fir.ref<!fir.logical<4>>
    return %9 : !fir.logical<4>
  }

// CHECK-LABEL:   func.func @_QPtestAny_2D(
// CHECK-SAME:                                        %[[A:.*]]: !fir.ref<!fir.array<10x0x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
// CHECK:           %[[SIZE10:.*]] = arith.constant 10 : index
// CHECK:           %[[SIZE_0:.*]] = arith.constant 0 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[SIZE10]], %[[SIZE_0]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[A_BOX_LOGICAL:.*]] = fir.embox %[[VAL_0]](%[[SHAPE]]) : (!fir.ref<!fir.array<10x0x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<10x0x!fir.logical<4>>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_LOGICAL]] : (!fir.box<!fir.array<10x0x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranAAnyLogical4x2_simplified(%[[A_BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> i1

// CHECK-LABEL:   func.func private @_FortranAAnyLogical4x2_simplified(
// CHECK-SAME:                                                         %[[ARR:.*]]: !fir.box<none>) -> i1 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[A_BOX_I32:.*]] = fir.convert %[[VAL_0]] : (!fir.box<none>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[A_BOX_I32]], %[[DIMINDEX_0]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[DIMINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMS_1:.*]]:3 = fir.box_dims %[[A_BOX_I32]], %[[DIMINDEX_1]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT1:.*]] = arith.subi %[[DIMS_1]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[RES:.*]]:2 = fir.iterate_while (%[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT1]] step %[[CINDEX_1]]) and (%[[OK:.*]] = %[[TRUE]]) iter_args(%[[INIT:.*]] = %[[FALSE]]) -> (i1) {
// CHECK:             %[[INNER_RES:.*]]:2 = fir.iterate_while (%[[ITER_1:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]]) and (%[[OK1:.*]] = %[[TRUE]]) iter_args(%[[INNER_INIT:.*]] = %[[INIT]]) -> (i1) {
// CHECK:               %[[ITEM:.*]] = fir.coordinate_of %[[A_BOX_I32]], %[[ITER_1]], %[[ITER]] : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[ITEMVAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:               %[[I32_0:.*]] = arith.constant 0 : i32
// CHECK:               %[[NEXT_ANY:.*]] = arith.cmpi ne, %[[ITEMVAL]], %[[I32_0]] : i32
// CHECK:               %[[I1_1:.*]] = arith.constant true
// CHECK:               %[[CONTINUE:.*]] = arith.xori %[[NEXT_ANY]], %[[I1_1]] : i1
// CHECK:               fir.result %[[CONTINUE]], %[[NEXT_ANY]] : i1, i1
// CHECK:             }
// CHECK:             fir.result %[[RETURN_VALS:.*]]#0, %[[RETURN_VALS]]#1 : i1, i1
// CHECK:           }
// CHECK:           return %[[RETURN_VAL:.*]]#1 : i1
// CHECK:         }

// -----
// Ensure All is simplified in correct usage

func.func @_QPtestAll_NoDimArg(%arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.logical<4> {bindc_name = "testAll_NoDimArg", uniq_name = "_QFtestAll_NoDimArgEtestAll_NoDimArg"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
  %c1 = arith.constant 1 : index
  %3 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
  %c3_i32 = arith.constant 3 : i32
  %4 = fir.convert %2 : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
  %5 = fir.convert %3 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %6 = fir.convert %c1 : (index) -> i32
  %7 = fir.call @_FortranAAll(%4, %5, %c3_i32, %6) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
  %8 = fir.convert %7 : (i1) -> !fir.logical<4>
  fir.store %8 to %0 : !fir.ref<!fir.logical<4>>
  %9 = fir.load %0 : !fir.ref<!fir.logical<4>>
  return %9 : !fir.logical<4>
}
func.func private @_FortranAAll(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1 attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPtestAll_NoDimArg(
// CHECK-SAME:                          %[[ARR:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.logical<4> {
// CHECK:           %[[SIZE:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[SIZE]] : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_LOGICAL:.*]] = fir.embox %[[ARR]](%[[SHAPE]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_LOGICAL]] : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranAAllLogical4x1_simplified(%[[A_BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> i1
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAAllLogical4x1_simplified(
// CHECK-SAME:                                                 %[[ARR:.*]]: !fir.box<none>) -> i1 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[INIT_COND:.*]] = arith.constant true
// CHECK:           %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[A_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[A_BOX_I32]], %[[DIM_INDEX0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[RES:.*]]:2 = fir.iterate_while (%[[ITER:.*]] = %[[C_INDEX0]] to %[[EXTENT]] step %[[C_INDEX1]]) and (%[[OK:.*]] = %[[INIT_COND]]) iter_args(%[[INIT:.*]] = %[[TRUE]]) -> (i1) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[A_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:             %[[I32_0:.*]] = arith.constant 0 : i32
// CHECK:             %[[CMP_AND_CONTINUE:.*]] = arith.cmpi ne, %[[ITEM_VAL]], %[[I32_0]] : i32
// CHECK:             fir.result %[[CMP_AND_CONTINUE]], %[[CMP_AND_CONTINUE]] : i1, i1
// CHECK:           }
// CHECK:           return %[[RES:.*]]#1 : i1
// CHECK:         }

// -----
// Ensure All is correctly simplified for different kind logical

func.func @_QPtestAll_NoDimArgLogical1(%arg0: !fir.ref<!fir.array<10x!fir.logical<1>>> {fir.bindc_name = "a"}) -> !fir.logical<1> {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.logical<1> {bindc_name = "testAll_NoDimArgLogical1", uniq_name = "_QFtestAll_NoDimArgLogical1EtestAll_NoDimArgLogical1"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10x!fir.logical<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<1>>>
  %c1 = arith.constant 1 : index
  %3 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
  %c3_i32 = arith.constant 3 : i32
  %4 = fir.convert %2 : (!fir.box<!fir.array<10x!fir.logical<1>>>) -> !fir.box<none>
  %5 = fir.convert %3 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %6 = fir.convert %c1 : (index) -> i32
  %7 = fir.call @_FortranAAll(%4, %5, %c3_i32, %6) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
  %8 = fir.convert %7 : (i1) -> !fir.logical<1>
  fir.store %8 to %0 : !fir.ref<!fir.logical<1>>
  %9 = fir.load %0 : !fir.ref<!fir.logical<1>>
  return %9 : !fir.logical<1>
}
func.func private @_FortranAAll(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1 attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPtestAll_NoDimArgLogical1(
// CHECK-SAME:                          %[[ARR:.*]]: !fir.ref<!fir.array<10x!fir.logical<1>>> {fir.bindc_name = "a"}) -> !fir.logical<1> {
// CHECK:           %[[SIZE:.*]] = arith.constant 10 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[SIZE]] : (index) -> !fir.shape<1>
// CHECK:           %[[A_BOX_LOGICAL:.*]] = fir.embox %[[ARR]](%[[SHAPE]]) : (!fir.ref<!fir.array<10x!fir.logical<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<1>>>
// CHECK:           %[[A_BOX_NONE:.*]] = fir.convert %[[A_BOX_LOGICAL]] : (!fir.box<!fir.array<10x!fir.logical<1>>>) -> !fir.box<none>
// CHECK:           %[[RES:.*]] = fir.call @_FortranAAllLogical1x1_simplified(%[[A_BOX_NONE]]) fastmath<contract> : (!fir.box<none>) -> i1
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAAllLogical1x1_simplified(
// CHECK-SAME:                                                 %[[ARR:.*]]: !fir.box<none>) -> i1 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[INIT_COND:.*]] = arith.constant true
// CHECK:           %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[A_BOX_I8:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi8>>
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[A_BOX_I8]], %[[DIM_INDEX0]] : (!fir.box<!fir.array<?xi8>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[RES:.*]]:2 = fir.iterate_while (%[[ITER:.*]] = %[[C_INDEX0]] to %[[EXTENT]] step %[[C_INDEX1]]) and (%[[OK:.*]] = %[[INIT_COND]]) iter_args(%[[INIT:.*]] = %[[TRUE]]) -> (i1) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[A_BOX_I8]], %[[ITER]] : (!fir.box<!fir.array<?xi8>>, index) -> !fir.ref<i8>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i8>
// CHECK:             %[[I8_0:.*]] = arith.constant 0 : i8
// CHECK:             %[[CMP_AND_CONTINUE:.*]] = arith.cmpi ne, %[[ITEM_VAL]], %[[I8_0]] : i8
// CHECK:             fir.result %[[CMP_AND_CONTINUE]], %[[CMP_AND_CONTINUE]] : i1, i1
// CHECK:           }
// CHECK:           return %[[RES:.*]]#1 : i1
// CHECK:         }


// -----
//  Ensure All is not simplified when call ends in 'Dim'

func.func @_QPtestAll_DimArg(%arg0: !fir.ref<!fir.array<10x10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.array<10x!fir.logical<4>> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  %c10_1 = arith.constant 10 : index
  %1 = fir.alloca !fir.array<10x!fir.logical<4>> {bindc_name = "testAll_DimArg", uniq_name = "_QFtestAll_DimArgEtestAll_DimArg"}
  %2 = fir.shape %c10_1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  %c1_i32 = arith.constant 1 : i32
  %4 = fir.shape %c10, %c10_0 : (index, index) -> !fir.shape<2>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10x10x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10x!fir.logical<4>>>
  %6 = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
  %c0 = arith.constant 0 : index
  %7 = fir.shape %c0 : (index) -> !fir.shape<1>
  %8 = fir.embox %6(%7) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  fir.store %8 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
  %9 = fir.address_of(@_QQcl.04ab56883945fd2c21a3b6d132f0bb37) : !fir.ref<!fir.char<1,48>>
  %c3_i32 = arith.constant 3 : i32
  %10 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
  %11 = fir.convert %5 : (!fir.box<!fir.array<10x10x!fir.logical<4>>>) -> !fir.box<none>
  %12 = fir.convert %9 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %13 = fir.call @_FortranAAllDim(%10, %11, %c1_i32, %12, %c3_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  %14 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
  %c0_2 = arith.constant 0 : index
  %15:3 = fir.box_dims %14, %c0_2 : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
  %16 = fir.box_addr %14 : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
  %17 = fir.shape_shift %15#0, %15#1 : (index, index) -> !fir.shapeshift<1>
  %18 = fir.array_load %16(%17) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<4>>
  %c1 = arith.constant 1 : index
  %c0_3 = arith.constant 0 : index
  %19 = arith.subi %c10_1, %c1 : index
  %20 = fir.do_loop %arg1 = %c0_3 to %19 step %c1 unordered iter_args(%arg2 = %3) -> (!fir.array<10x!fir.logical<4>>) {
    %22 = fir.array_fetch %18, %arg1 : (!fir.array<?x!fir.logical<4>>, index) -> !fir.logical<4>
    %23 = fir.array_update %arg2, %22, %arg1 : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
    fir.result %23 : !fir.array<10x!fir.logical<4>>
  }
  fir.array_merge_store %3, %20 to %1 : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.ref<!fir.array<10x!fir.logical<4>>>
  fir.freemem %16 : !fir.heap<!fir.array<?x!fir.logical<4>>>
  %21 = fir.load %1 : !fir.ref<!fir.array<10x!fir.logical<4>>>
  return %21 : !fir.array<10x!fir.logical<4>>
}
func.func private @_FortranAAllDim(!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none attributes {fir.runtime}

// CHECK-LABEL:   func.func @_QPtestAll_DimArg(
// CHECK-SAME:                          %[[ARR:.*]]: !fir.ref<!fir.array<10x10x!fir.logical<4>>> {fir.bindc_name = "a"}) -> !fir.array<10x!fir.logical<4>> {
// CHECK-NOT        fir.call @_FortranAAllDimLogical4x1_simplified({{.*}})
// CHECK:           fir.call @_FortranAAllDim({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
// CHECK-NOT        fir.call @_FortranAAllDimLogical4x1_simplified({{.*}})

// -----
// Check Minloc simplifies correctly for 1D case with 1D mask, I32 input
func.func @_QPtestminloc_works1d(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "b"}) -> !fir.array<1xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %1 = fir.alloca !fir.array<1xi32> {bindc_name = "testminloc_works1d", uniq_name = "_QFtestminloc_works1dEtestminloc_works1d"}
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %6 = fir.shape %c10_0 : (index) -> !fir.shape<1>
  %7 = fir.embox %arg1(%6) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
  %c4 = arith.constant 4 : index
  %false = arith.constant false
  %8 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %9 = fir.shape %c0 : (index) -> !fir.shape<1>
  %10 = fir.embox %8(%9) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %10 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %11 = fir.address_of(@_QQcl.ea5bcf7f706678e1796661f8916f3379) : !fir.ref<!fir.char<1,55>>
  %c5_i32 = arith.constant 5 : i32
  %12 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %13 = fir.convert %5 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %14 = fir.convert %c4 : (index) -> i32
  %15 = fir.convert %11 : (!fir.ref<!fir.char<1,55>>) -> !fir.ref<i8>
  %16 = fir.convert %7 : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
  %17 = fir.call @_FortranAMinlocInteger4(%12, %13, %14, %15, %c5_i32, %16, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %18 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_1 = arith.constant 0 : index
  %19:3 = fir.box_dims %18, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %20 = fir.box_addr %18 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %21 = fir.shape_shift %19#0, %19#1 : (index, index) -> !fir.shapeshift<1>
  %22 = fir.array_load %20(%21) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  %c1_2 = arith.constant 1 : index
  %c0_3 = arith.constant 0 : index
  %23 = arith.subi %c1, %c1_2 : index
  %24 = fir.do_loop %arg2 = %c0_3 to %23 step %c1_2 unordered iter_args(%arg3 = %3) -> (!fir.array<1xi32>) {
    %26 = fir.array_fetch %22, %arg2 : (!fir.array<?xi32>, index) -> i32
    %27 = fir.array_update %arg3, %26, %arg2 : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
    fir.result %27 : !fir.array<1xi32>
  }
  fir.array_merge_store %3, %24 to %1 : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
  fir.freemem %20 : !fir.heap<!fir.array<?xi32>>
  %25 = fir.load %1 : !fir.ref<!fir.array<1xi32>>
  return %25 : !fir.array<1xi32>
}

// CHECK-LABEL:   func.func @_QPtestminloc_works1d(
// CHECK-SAME:                                     %[[INARR:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                                     %[[MASK:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>> {fir.bindc_name = "b"}) -> !fir.array<1xi32> {
// CHECK:           %[[OUTARR:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK:           %[[SIZE10_0:.*]] = arith.constant 10 : index
// CHECK:           %[[SIZE10_1:.*]] = arith.constant 10 : index
// CHECK:           %[[INARR_SHAPE:.*]] = fir.shape %[[SIZE10_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[BOX_INARR:.*]] = fir.embox %[[INARR]](%[[INARR_SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[MASK_SHAPE:.*]] = fir.shape %[[SIZE10_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[BOX_MASK:.*]] = fir.embox %[[MASK]](%[[MASK_SHAPE]]) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
// CHECK:           %[[REF_BOX_OUTARR_NONE:.*]] = fir.convert %[[OUTARR]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[BOX_INARR_NONE:.*]] = fir.convert %[[BOX_INARR]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
// CHECK:           %[[BOX_MASK_NONE:.*]] = fir.convert %[[BOX_MASK]] : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranAMinlocInteger4x1_Logical4x1_i32_contract_simplified(%[[REF_BOX_OUTARR_NONE]], %[[BOX_INARR_NONE]], %[[BOX_MASK_NONE]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()

// CHECK-LABEL:   func.func private @_FortranAMinlocInteger4x1_Logical4x1_i32_contract_simplified(
// CHECK-SAME:             %[[REF_BOX_OUTARR_NONE:.*]]: !fir.ref<!fir.box<none>>,
// CHECK-SAME:             %[[BOX_INARR_NONE:.*]]: !fir.box<none>,
// CHECK-SAME:             %[[BOX_MASK_NONE:.*]]: !fir.box<none>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[FLAG_ALLOC:.*]] = fir.alloca i32
// CHECK:           %[[INIT_OUT_IDX:.*]] = arith.constant 0 : i32
// CHECK:           %[[OUTARR_SIZE:.*]] = arith.constant 1 : index
// CHECK:           %[[OUTARR:.*]] = fir.allocmem !fir.array<1xi32>
// CHECK:           %[[OUTARR_SHAPE:.*]] = fir.shape %[[OUTARR_SIZE]] : (index) -> !fir.shape<1>
// CHECK:           %[[BOX_OUTARR:.*]] = fir.embox %[[OUTARR]](%[[OUTARR_SHAPE]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<1xi32>>>
// CHECK:           %[[OUTARR_IDX0:.*]] = arith.constant 0 : index
// CHECK:           %[[OUTARR_ITEM0:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX0]] : (!fir.box<!fir.heap<!fir.array<1xi32>>>, index) -> !fir.ref<i32>
// CHECK:           fir.store %[[INIT_OUT_IDX]] to %[[OUTARR_ITEM0]] : !fir.ref<i32>
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[BOX_INARR:.*]] = fir.convert %[[BOX_INARR_NONE]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[FLAG_SET:.*]] = arith.constant 1 : i32
// CHECK:           %[[FLAG_EMPTY:.*]] = arith.constant 0 : i32
// CHECK:           fir.store %[[FLAG_EMPTY]] to %[[FLAG_ALLOC]] : !fir.ref<i32>
// CHECK:           %[[BOX_MASK:.*]] = fir.convert %[[BOX_MASK_NONE]] : (!fir.box<none>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK:           %[[MAX:.*]] = arith.constant 2147483647 : i32
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[BOX_INARR]], %[[DIM_INDEX0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[DOLOOP:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[MIN:.*]] = %[[MAX]]) -> (i32) {
// CHECK:             %[[MASK_ITEM:.*]] = fir.coordinate_of %[[BOX_MASK]], %[[ITER]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[MASK_ITEMVAL:.*]] = fir.load %[[MASK_ITEM]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[MASK_IF_ITEM:.*]] = fir.convert %[[MASK_ITEMVAL]] : (!fir.logical<4>) -> i1
// CHECK:             %[[IF_MASK:.*]] = fir.if %[[MASK_IF_ITEM]] -> (i32) {
// CHECK:               fir.store %[[FLAG_SET]] to %[[FLAG_ALLOC]] : !fir.ref<i32>
// CHECK:               %[[INARR_ITEM:.*]] = fir.coordinate_of %[[BOX_INARR]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[INARR_ITEMVAL:.*]] = fir.load %[[INARR_ITEM]] : !fir.ref<i32>
// CHECK:               %[[NEW_MIN:.*]] = arith.cmpi slt, %[[INARR_ITEMVAL]], %[[MIN]] : i32
// CHECK:               %[[IF_NEW_MIN:.*]] = fir.if %[[NEW_MIN]] -> (i32) {
// CHECK:                 %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:                 %[[OUTARR_IDX:.*]] = arith.constant 0 : index
// CHECK:                 %[[OUTARR_ITEM:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX]] : (!fir.box<!fir.heap<!fir.array<1xi32>>>, index) -> !fir.ref<i32>
// CHECK:                 %[[ITER_I32:.*]] = fir.convert %[[ITER]] : (index) -> i32
// CHECK:                 %[[FORTRAN_IDX:.*]] = arith.addi %[[ITER_I32]], %[[ONE]] : i32
// CHECK:                 fir.store %[[FORTRAN_IDX]] to %[[OUTARR_ITEM]] : !fir.ref<i32>
// CHECK:                 fir.result %[[INARR_ITEMVAL]] : i32
// CHECK:               } else {
// CHECK:                 fir.result %[[MIN]] : i32
// CHECK:               }
// CHECK:               fir.result %[[IF_NEW_MIN:.*]] : i32
// CHECK:             } else {
// CHECK:               fir.result %[[MIN]] : i32
// CHECK:             }
// CHECK:             fir.result %[[IF_MASK:.*]] : i32
// CHECK:           }
// CHECK:           %[[FLAG_VAL:.*]] = fir.load %[[FLAG_ALLOC]] : !fir.ref<i32>
// CHECK:           %[[FLAG_WAS_SET:.*]] = arith.cmpi eq, %[[FLAG_VAL]], %[[FLAG_SET]] : i32
// CHECK:           fir.if %[[FLAG_WAS_SET]] {
// CHECK:             %[[TEST_MAX:.*]] = arith.constant 2147483647 : i32
// CHECK:             %[[INIT_NOT_CHANGED:.*]] = arith.cmpi eq, %[[TEST_MAX]], %[[DO_LOOP:.*]] : i32
// CHECK:             fir.if %[[INIT_NOT_CHANGED]] {
// CHECK:               %[[FLAG_OUTARR_IDX:.*]] = arith.constant 0 : index
// CHECK:               %[[FLAG_OUTARR_ITEM:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[FLAG_OUTARR_IDX]] : (!fir.box<!fir.heap<!fir.array<1xi32>>>, index) -> !fir.ref<i32>
// CHECK:               fir.store %[[FLAG_SET]] to %[[FLAG_OUTARR_ITEM]] : !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[REF_BOX_OUTARR:.*]] = fir.convert %[[REF_BOX_OUTARR_NONE]] : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<1xi32>>>>
// CHECK:           fir.store %[[BOX_OUTARR]] to %[[REF_BOX_OUTARR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<1xi32>>>>
// CHECK:           return
// CHECK:         }

// -----
// Check Minloc simplifies correctly for 2D case with no mask and I64 Int as result

func.func @_QPtestminloc_works2d_nomask(%arg0: !fir.ref<!fir.array<10x10xi32>> {fir.bindc_name = "a"}) -> !fir.array<2xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi64>>>
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  %1 = fir.alloca !fir.array<2xi32> {bindc_name = "testminloc_works2d_nomask", uniq_name = "_QFtestminloc_works2d_nomaskEtestminloc_works2d_nomask"}
  %2 = fir.shape %c2 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.array<2xi32>
  %4 = fir.shape %c10, %c10_0 : (index, index) -> !fir.shape<2>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10xi32>>
  %c8_i32 = arith.constant 8 : i32
  %6 = fir.absent !fir.box<i1>
  %false = arith.constant false
  %7 = fir.zero_bits !fir.heap<!fir.array<?xi64>>
  %c0 = arith.constant 0 : index
  %8 = fir.shape %c0 : (index) -> !fir.shape<1>
  %9 = fir.embox %7(%8) : (!fir.heap<!fir.array<?xi64>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi64>>>
  fir.store %9 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>
  %10 = fir.address_of(@_QQcl.cba8b79c45ccae77d79d66a39ac99823) : !fir.ref<!fir.char<1,62>>
  %c4_i32 = arith.constant 4 : i32
  %11 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %5 : (!fir.box<!fir.array<10x10xi32>>) -> !fir.box<none>
  %13 = fir.convert %10 : (!fir.ref<!fir.char<1,62>>) -> !fir.ref<i8>
  %14 = fir.convert %6 : (!fir.box<i1>) -> !fir.box<none>
  %15 = fir.call @_FortranAMinlocInteger4(%11, %12, %c8_i32, %13, %c4_i32, %14, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %16 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi64>>>>
  %c0_1 = arith.constant 0 : index
  %17:3 = fir.box_dims %16, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi64>>>, index) -> (index, index, index)
  %18 = fir.box_addr %16 : (!fir.box<!fir.heap<!fir.array<?xi64>>>) -> !fir.heap<!fir.array<?xi64>>
  %19 = fir.shape_shift %17#0, %17#1 : (index, index) -> !fir.shapeshift<1>
  %20 = fir.array_load %18(%19) : (!fir.heap<!fir.array<?xi64>>, !fir.shapeshift<1>) -> !fir.array<?xi64>
  %c1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %21 = arith.subi %c2, %c1 : index
  %22 = fir.do_loop %arg1 = %c0_2 to %21 step %c1 unordered iter_args(%arg2 = %3) -> (!fir.array<2xi32>) {
    %24 = fir.array_fetch %20, %arg1 : (!fir.array<?xi64>, index) -> i64
    %25 = fir.convert %24 : (i64) -> i32
    %26 = fir.array_update %arg2, %25, %arg1 : (!fir.array<2xi32>, i32, index) -> !fir.array<2xi32>
    fir.result %26 : !fir.array<2xi32>
  }
  fir.array_merge_store %3, %22 to %1 : !fir.array<2xi32>, !fir.array<2xi32>, !fir.ref<!fir.array<2xi32>>
  fir.freemem %18 : !fir.heap<!fir.array<?xi64>>
  %23 = fir.load %1 : !fir.ref<!fir.array<2xi32>>
  return %23 : !fir.array<2xi32>
}

// CHECK-LABEL:   func.func @_QPtestminloc_works2d_nomask(
// CHECK-SAME:                                     %[[INARR:.*]]: !fir.ref<!fir.array<10x10xi32>> {fir.bindc_name = "a"}) -> !fir.array<2xi32> {
// CHECK:           %[[ABSENT_MASK:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[ABSENT_MASK_NONE:.*]] = fir.convert %[[ABSENT_MASK]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           fir.call @_FortranAMinlocInteger4x2_i64_contract_simplified(%{{.*}}, %{{.*}}, %[[ABSENT_MASK_NONE]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()

// CHECK-LABEL:   func.func private @_FortranAMinlocInteger4x2_i64_contract_simplified(
// CHECK-SAME:              %[[REF_BOX_OUTARR_NONE:.*]]: !fir.ref<!fir.box<none>>,
// CHECK-SAME:              %[[BOX_INARR_NONE:.*]]: !fir.box<none>,
// CHECK-SAME:              %[[BOX_MASK_NONE:.*]]: !fir.box<none>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[FLAG_ALLOC:.*]] = fir.alloca i64
// CHECK:           %[[INIT_OUT_IDX:.*]] = arith.constant 0 : i64
// CHECK:           %[[OUTARR_SIZE:.*]] = arith.constant 2 : index
// CHECK:           %[[OUTARR:.*]] = fir.allocmem !fir.array<2xi64>
// CHECK:           %[[OUTARR_SHAPE:.*]] = fir.shape %[[OUTARR_SIZE]] : (index) -> !fir.shape<1>
// CHECK:           %[[BOX_OUTARR:.*]] = fir.embox %[[OUTARR]](%[[OUTARR_SHAPE]]) : (!fir.heap<!fir.array<2xi64>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<2xi64>>>
// CHECK:           %[[OUTARR_IDX0:.*]] = arith.constant 0 : index
// CHECK:           %[[OUTARR_ITEM0:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX0]] : (!fir.box<!fir.heap<!fir.array<2xi64>>>, index) -> !fir.ref<i64>
// CHECK:           fir.store %[[INIT_OUT_IDX]] to %[[OUTARR_ITEM0]] : !fir.ref<i64>
// CHECK:           %[[OUTARR_IDX1:.*]] = arith.constant 1 : index
// CHECK:           %[[OUTARR_ITEM1:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX1]] : (!fir.box<!fir.heap<!fir.array<2xi64>>>, index) -> !fir.ref<i64>
// CHECK:           fir.store %[[INIT_OUT_IDX]] to %[[OUTARR_ITEM1]] : !fir.ref<i64>
// CHECK:           %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[BOX_INARR:.*]] = fir.convert %[[BOX_INARR_NONE]] : (!fir.box<none>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK:           %[[FLAG_SET:.*]] = arith.constant 1 : i64
// CHECK:           %[[FLAG_EMPTY:.*]] = arith.constant 0 : i64
// CHECK:           fir.store %[[FLAG_EMPTY]] to %[[FLAG_ALLOC]] : !fir.ref<i64>
// CHECK:           %[[MAX:.*]] = arith.constant 2147483647 : i32
// CHECK:           %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_INDEX0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX_INARR]], %[[DIM_INDEX0]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT0:.*]] = arith.subi %[[DIMS0]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[DIM_INDEX1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX_INARR]], %[[DIM_INDEX1]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[EXTENT1:.*]] = arith.subi %[[DIMS1]]#1, %[[C_INDEX1]] : index
// CHECK:           %[[DOLOOP0:.*]] = fir.do_loop %[[ITER0:.*]] = %[[C_INDEX0]] to %[[EXTENT1]] step %[[C_INDEX1]] iter_args(%[[MIN0:.*]] = %[[MAX]]) -> (i32) {
// CHECK:             %[[DOLOOP1:.*]] = fir.do_loop %[[ITER1:.*]] = %[[C_INDEX0]] to %[[EXTENT0]] step %[[C_INDEX1]] iter_args(%[[MIN1:.*]] = %[[MIN0]]) -> (i32) {
// CHECK:               %[[INARR_ITEM:.*]] = fir.coordinate_of %[[BOX_INARR]], %[[ITER1]], %[[ITER0]] : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[INARR_ITEMVAL:.*]] = fir.load %[[INARR_ITEM]] : !fir.ref<i32>
// CHECK:               %[[NEW_MIN:.*]] = arith.cmpi slt, %[[INARR_ITEMVAL]], %[[MIN1]] : i32
// CHECK:               %[[IF_NEW_MIN:.*]] = fir.if %[[NEW_MIN]] -> (i32) {
// CHECK:                 %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:                 %[[OUTARR_IDX0:.*]] = arith.constant 0 : index
// CHECK:                 %[[OUTARR_ITEM0:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX0]] : (!fir.box<!fir.heap<!fir.array<2xi64>>>, index) -> !fir.ref<i64>
// CHECK:                 %[[ITER1_I64:.*]] = fir.convert %[[ITER1]] : (index) -> i64
// CHECK:                 %[[FORTRAN_IDX1:.*]] = arith.addi %[[ITER1_I64]], %[[ONE]] : i64
// CHECK:                 fir.store %[[FORTRAN_IDX1]] to %[[OUTARR_ITEM0]] : !fir.ref<i64>
// CHECK:                 %[[OUTARR_IDX1:.*]] = arith.constant 1 : index
// CHECK:                 %[[OUTARR_ITEM1:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX1]] : (!fir.box<!fir.heap<!fir.array<2xi64>>>, index) -> !fir.ref<i64>
// CHECK:                 %[[ITER0_I64:.*]] = fir.convert %[[ITER0]] : (index) -> i64
// CHECK:                 %[[FORTRAN_IDX0:.*]] = arith.addi %[[ITER0_I64]], %[[ONE]] : i64
// CHECK:                 fir.store %[[FORTRAN_IDX0]] to %[[OUTARR_ITEM1]] : !fir.ref<i64>
// CHECK:                 fir.result %[[INARR_ITEMVAL]] : i32
// CHECK:               } else {
// CHECK:                 fir.result %[[MIN1]] : i32
// CHECK:               }
// CHECK:               fir.result %[[IF_NEW_MIN:.*]] : i32
// CHECK:             }
// CHECK:             fir.result %[[DOLOOP1:.*]] : i32
// CHECK:           }
// CHECK:           %[[FLAG_VAL:.*]] = fir.load %[[FLAG_ALLOC]] : !fir.ref<i64>
// CHECK:           %[[FLAG_WAS_SET:.*]] = arith.cmpi eq, %[[FLAG_VAL]], %[[FLAG_SET]] : i64
// CHECK:           fir.if %[[FLAG_WAS_SET]] {
// CHECK:             %[[TEST_MAX:.*]] = arith.constant 2147483647 : i32
// CHECK:             %[[INIT_NOT_CHANGED:.*]] = arith.cmpi eq, %[[TEST_MAX]], %[[DO_LOOP:.*]] : i32
// CHECK:             fir.if %[[INIT_NOT_CHANGED]] {
// CHECK:               %[[FLAG_OUTARR_IDX0:.*]] = arith.constant 0 : index
// CHECK:               %[[FLAG_OUTARR_ITEM0:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[FLAG_OUTARR_IDX0]] : (!fir.box<!fir.heap<!fir.array<2xi64>>>, index) -> !fir.ref<i64>
// CHECK:               fir.store %[[FLAG_SET]] to %[[FLAG_OUTARR_ITEM0]] : !fir.ref<i64>
// CHECK:               %[[FLAG_OUTARR_IDX1:.*]] = arith.constant 1 : index
// CHECK:               %[[FLAG_OUTARR_ITEM1:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[FLAG_OUTARR_IDX1]] : (!fir.box<!fir.heap<!fir.array<2xi64>>>, index) -> !fir.ref<i64>
// CHECK:               fir.store %[[FLAG_SET]] to %[[FLAG_OUTARR_ITEM1]] : !fir.ref<i64
// CHECK:             }
// CHECK:           }
// CHECK:           %[[REF_BOX_OUTARR:.*]] = fir.convert %[[REF_BOX_OUTARR_NONE]] : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<2xi64>>>>
// CHECK:           fir.store %[[BOX_OUTARR]] to %[[REF_BOX_OUTARR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<2xi64>>>>
// CHECK:           return
// CHECK:         }

// -----
// Check Minloc simplifies correctly for 1D case with scalar mask and f64 input

func.func @_QPtestminloc_works1d_scalarmask_f64(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) -> !fir.array<1xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %1 = fir.alloca !fir.array<1xi32> {bindc_name = "testminloc_works1d_scalarmask_f64", uniq_name = "_QFtestminloc_works1d_scalarmask_f64Etestminloc_works1d_scalarmask_f64"}
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf64>>
  %6 = fir.embox %arg1 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %c4 = arith.constant 4 : index
  %false = arith.constant false
  %7 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %8 = fir.shape %c0 : (index) -> !fir.shape<1>
  %9 = fir.embox %7(%8) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %9 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %10 = fir.address_of(@_QQcl.66951c28c5b8bab5cdb25c1ac762b978) : !fir.ref<!fir.char<1,65>>
  %c6_i32 = arith.constant 6 : i32
  %11 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %5 : (!fir.box<!fir.array<10xf64>>) -> !fir.box<none>
  %13 = fir.convert %c4 : (index) -> i32
  %14 = fir.convert %10 : (!fir.ref<!fir.char<1,65>>) -> !fir.ref<i8>
  %15 = fir.convert %6 : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  %16 = fir.call @_FortranAMinlocReal8(%11, %12, %13, %14, %c6_i32, %15, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %17 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_0 = arith.constant 0 : index
  %18:3 = fir.box_dims %17, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %19 = fir.box_addr %17 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %20 = fir.shape_shift %18#0, %18#1 : (index, index) -> !fir.shapeshift<1>
  %21 = fir.array_load %19(%20) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  %c1_1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %22 = arith.subi %c1, %c1_1 : index
  %23 = fir.do_loop %arg2 = %c0_2 to %22 step %c1_1 unordered iter_args(%arg3 = %3) -> (!fir.array<1xi32>) {
    %25 = fir.array_fetch %21, %arg2 : (!fir.array<?xi32>, index) -> i32
    %26 = fir.array_update %arg3, %25, %arg2 : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
    fir.result %26 : !fir.array<1xi32>
  }
  fir.array_merge_store %3, %23 to %1 : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
  fir.freemem %19 : !fir.heap<!fir.array<?xi32>>
  %24 = fir.load %1 : !fir.ref<!fir.array<1xi32>>
  return %24 : !fir.array<1xi32>
}

// CHECK-LABEL:   func.func @_QPtestminloc_works1d_scalarmask_f64(
// CHECK-SAME:                                     %[[INARR:.*]]: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"},
// CHECK-SAME:                                     %[[MASK:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}) -> !fir.array<1xi32> {
// CHECK:           fir.call @_FortranAMinlocReal8x1_Logical4x0_i32_contract_simplified({{.*}}, {{.*}}, {{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()

// CHECK-LABEL:   func.func private @_FortranAMinlocReal8x1_Logical4x0_i32_contract_simplified(
// CHECK-SAME:             %[[REF_BOX_OUTARR_NONE:.*]]: !fir.ref<!fir.box<none>>,
// CHECK-SAME:             %[[BOX_INARR_NONE:.*]]: !fir.box<none>,
// CHECK-SAME:             %[[BOX_MASK_NONE:.*]]: !fir.box<none>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[FLAG_ALLOC:.*]] = fir.alloca i32
// CHECK:           %[[INIT_OUT_IDX:.*]] = arith.constant 0 : i32
// CHECK:           %[[OUTARR_SIZE:.*]] = arith.constant 1 : index
// CHECK:           %[[OUTARR:.*]] = fir.allocmem !fir.array<1xi32>
// CHECK:           %[[OUTARR_SHAPE:.*]] = fir.shape %[[OUTARR_SIZE]] : (index) -> !fir.shape<1>
// CHECK:           %[[BOX_OUTARR:.*]] = fir.embox %[[OUTARR]](%[[OUTARR_SHAPE]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<1xi32>>>
// CHECK:           %[[OUTARR_IDX0:.*]] = arith.constant 0 : index
// CHECK:           %[[OUTARR_ITEM0:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX0]] : (!fir.box<!fir.heap<!fir.array<1xi32>>>, index) -> !fir.ref<i32>
// CHECK:           fir.store %[[INIT_OUT_IDX]] to %[[OUTARR_ITEM0]] : !fir.ref<i32>
// CHECK:           %[[BOX_MASK:.*]] = fir.convert %[[BOX_MASK_NONE]] : (!fir.box<none>) -> !fir.box<!fir.array<1xi1>>
// CHECK:           %[[MASK_IDX0:.*]] = arith.constant 0 : index
// CHECK:           %[[MASK_ITEM:.*]] = fir.coordinate_of %[[BOX_MASK]], %[[MASK_IDX0]] : (!fir.box<!fir.array<1xi1>>, index) -> !fir.ref<i1>
// CHECK:           %[[MASK:.*]] = fir.load %[[MASK_ITEM]] : !fir.ref<i1>
// CHECK:           %[[INIT_RES:.*]] = fir.if %[[MASK]] -> (f64) {
// CHECK:             %[[C_INDEX0:.*]] = arith.constant 0 : index
// CHECK:             %[[BOX_INARR:.*]] = fir.convert %[[BOX_INARR_NONE]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf64>>
// CHECK:             %[[FLAG_SET:.*]] = arith.constant 1 : i32
// CHECK:             %[[FLAG_EMPTY:.*]] = arith.constant 0 : i32
// CHECK:             fir.store %[[FLAG_EMPTY]] to %[[FLAG_ALLOC]] : !fir.ref<i32>
// CHECK:             %[[MAX:.*]] = arith.constant 1.7976931348623157E+308 : f64
// CHECK:             %[[C_INDEX1:.*]] = arith.constant 1 : index
// CHECK:             %[[DIM_INDEX:.*]] = arith.constant 0 : index
// CHECK:             %[[DIMS:.*]]:3 = fir.box_dims %[[BOX_INARR]], %[[DIM_INDEX]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:             %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[C_INDEX1]] : index
// CHECK:             %[[DOLOOP:.*]] = fir.do_loop %[[ITER:.*]] = %[[C_INDEX0]] to %[[EXTENT]] step %[[C_INDEX1]] iter_args(%[[MIN:.*]] = %[[MAX]]) -> (f64) {
// CHECK:               %[[INARR_ITEM:.*]] = fir.coordinate_of %[[BOX_INARR]], %[[ITER]] : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:               %[[INARR_ITEMVAL:.*]] = fir.load %[[INARR_ITEM]] : !fir.ref<f64>
// CHECK:               %[[NEW_MIN:.*]] = arith.cmpf olt, %[[INARR_ITEMVAL]], %[[MIN]] : f64
// CHECK:               %[[IF_NEW_MIN:.*]] = fir.if %[[NEW_MIN]] -> (f64) {
// CHECK:                 %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:                 %[[OUTARR_IDX:.*]] = arith.constant 0 : index
// CHECK:                 %[[OUTARR_ITEM:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[OUTARR_IDX]] : (!fir.box<!fir.heap<!fir.array<1xi32>>>, index) -> !fir.ref<i32>
// CHECK:                 %[[ITER_I32:.*]] = fir.convert %[[ITER]] : (index) -> i32
// CHECK:                 %[[FORTRAN_IDX:.*]] = arith.addi %[[ITER_I32]], %[[ONE]] : i32
// CHECK:                 fir.store %[[FORTRAN_IDX]] to %[[OUTARR_ITEM]] : !fir.ref<i32>
// CHECK:                 fir.result %[[INARR_ITEMVAL]] : f64
// CHECK:               } else {
// CHECK:                 fir.result %[[MIN]] : f64
// CHECK:               }
// CHECK:               fir.result %[[IF_NEW_MIN:.*]] : f64
// CHECK:             }
// CHECK:           }
// CHECK:           %[[FLAG_CHECK:.*]] = arith.constant 1 : i32
// CHECK:           %[[FLAG_VAL:.*]] = fir.load %[[FLAG_ALLOC]] : !fir.ref<i32>
// CHECK:           %[[FLAG_WAS_SET:.*]] = arith.cmpi eq, %[[FLAG_VAL]], %[[FLAG_CHECK]] : i32
// CHECK:           fir.if %[[FLAG_WAS_SET]] {
// CHECK:             %[[TEST_MAX:.*]] = arith.constant 1.7976931348623157E+308 : f64
// CHECK:             %[[INIT_NOT_CHANGED:.*]] = arith.cmpf oeq, %[[TEST_MAX]], %[[INIT_RES:.*]] : f64
// CHECK:             fir.if %[[INIT_NOT_CHANGED]] {
// CHECK:               %[[FLAG_OUTARR_IDX:.*]] = arith.constant 0 : index
// CHECK:               %[[FLAG_OUTARR_ITEM:.*]] = fir.coordinate_of %[[BOX_OUTARR]], %[[FLAG_OUTARR_IDX]] : (!fir.box<!fir.heap<!fir.array<1xi32>>>, index) -> !fir.ref<i32>
// CHECK:               fir.store %[[FLAG_CHECK]] to %[[FLAG_OUTARR_ITEM]] : !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[REF_BOX_OUTARR:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<1xi32>>>>
// CHECK:           fir.store %[[BOX_OUTARR]] to %[[REF_BOX_OUTARR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<1xi32>>>>
// CHECK:           return
// CHECK:         }

// -----
// Check Minloc is not simplified when BACK arg is set

func.func @_QPtestminloc_doesntwork1d_back(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %1 = fir.alloca !fir.array<1xi32> {bindc_name = "testminloc_doesntwork1d_back", uniq_name = "_QFtestminloc_doesntwork1d_backEtestminloc_doesntwork1d_back"}
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %true = arith.constant true
  %6 = fir.absent !fir.box<i1>
  %c4 = arith.constant 4 : index
  %7 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %8 = fir.shape %c0 : (index) -> !fir.shape<1>
  %9 = fir.embox %7(%8) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %9 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %10 = fir.address_of(@_QQcl.3791f01d699716ba5914ae524c6a8dee) : !fir.ref<!fir.char<1,62>>
  %c4_i32 = arith.constant 4 : i32
  %11 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %5 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %13 = fir.convert %c4 : (index) -> i32
  %14 = fir.convert %10 : (!fir.ref<!fir.char<1,62>>) -> !fir.ref<i8>
  %15 = fir.convert %6 : (!fir.box<i1>) -> !fir.box<none>
  %16 = fir.call @_FortranAMinlocInteger4(%11, %12, %13, %14, %c4_i32, %15, %true) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %17 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_0 = arith.constant 0 : index
  %18:3 = fir.box_dims %17, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %19 = fir.box_addr %17 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %20 = fir.shape_shift %18#0, %18#1 : (index, index) -> !fir.shapeshift<1>
  %21 = fir.array_load %19(%20) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  %c1_1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %22 = arith.subi %c1, %c1_1 : index
  %23 = fir.do_loop %arg1 = %c0_2 to %22 step %c1_1 unordered iter_args(%arg2 = %3) -> (!fir.array<1xi32>) {
    %25 = fir.array_fetch %21, %arg1 : (!fir.array<?xi32>, index) -> i32
    %26 = fir.array_update %arg2, %25, %arg1 : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
    fir.result %26 : !fir.array<1xi32>
  }
  fir.array_merge_store %3, %23 to %1 : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
  fir.freemem %19 : !fir.heap<!fir.array<?xi32>>
  %24 = fir.load %1 : !fir.ref<!fir.array<1xi32>>
  return %24 : !fir.array<1xi32>
}

// CHECK-LABEL:   func.func @_QPtestminloc_doesntwork1d_back(
// CHECK-SAME:                                               %[[ARR:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
// CHECK-NOT:         fir.call @_FortranAMinlocInteger4x1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()
// CHECK:             fir.call @_FortranAMinlocInteger4({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NOT:         fir.call @_FortranAMinlocInteger4x1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()

// -----
// Check Minloc is not simplified when DIM arg is set

func.func @_QPtestminloc_doesntwork1d_dim(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<i32>>
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %1 = fir.alloca !fir.array<1xi32> {bindc_name = "testminloc_doesntwork1d_dim", uniq_name = "_QFtestminloc_doesntwork1d_dimEtestminloc_doesntwork1d_dim"}
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %c1_i32 = arith.constant 1 : i32
  %6 = fir.absent !fir.box<i1>
  %c4 = arith.constant 4 : index
  %false = arith.constant false
  %7 = fir.zero_bits !fir.heap<i32>
  %8 = fir.embox %7 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %8 to %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = fir.address_of(@_QQcl.cfcf4329f25d06a4b02a0c8f532ee9df) : !fir.ref<!fir.char<1,61>>
  %c4_i32 = arith.constant 4 : i32
  %10 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
  %11 = fir.convert %5 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %12 = fir.convert %c4 : (index) -> i32
  %13 = fir.convert %9 : (!fir.ref<!fir.char<1,61>>) -> !fir.ref<i8>
  %14 = fir.convert %6 : (!fir.box<i1>) -> !fir.box<none>
  %15 = fir.call @_FortranAMinlocDim(%10, %11, %12, %c1_i32, %13, %c4_i32, %14, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %16 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %17 = fir.box_addr %16 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  %18 = fir.load %17 : !fir.heap<i32>
  fir.freemem %17 : !fir.heap<i32>
  %c1_0 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %19 = arith.subi %c1, %c1_0 : index
  %20 = fir.do_loop %arg1 = %c0 to %19 step %c1_0 unordered iter_args(%arg2 = %3) -> (!fir.array<1xi32>) {
    %22 = fir.array_update %arg2, %18, %arg1 : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
    fir.result %22 : !fir.array<1xi32>
  }
  fir.array_merge_store %3, %20 to %1 : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
  %21 = fir.load %1 : !fir.ref<!fir.array<1xi32>>
  return %21 : !fir.array<1xi32>
}
// CHECK-LABEL:   func.func @_QPtestminloc_doesntwork1d_dim(
// CHECK-SAME:                                             %[[ARR:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
// CHECK-NOT:         fir.call @_FortranAMinlocDimx1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()
// CHECK:             fir.call @_FortranAMinlocDim({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NOT:         fir.call @_FortranAMinlocDimx1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()

// -----
// Check Minloc is not simplified when dimension of inputArr is unknown

func.func @_QPtestminloc_doesntwork1d_unknownsize(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %c1 = arith.constant 1 : index
  %1 = fir.alloca !fir.array<1xi32> {bindc_name = "testminloc_doesntwork1d_unknownsize", uniq_name = "_QFtestminloc_doesntwork1d_unknownsizeEtestminloc_doesntwork1d_unknownsize"}
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3 = fir.array_load %1(%2) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
  %4 = fir.absent !fir.box<i1>
  %c4 = arith.constant 4 : index
  %false = arith.constant false
  %5 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %6 = fir.shape %c0 : (index) -> !fir.shape<1>
  %7 = fir.embox %5(%6) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %7 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %8 = fir.address_of(@_QQcl.2064f5e9298c2127417d52b69eac898e) : !fir.ref<!fir.char<1,69>>
  %c4_i32 = arith.constant 4 : i32
  %9 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %10 = fir.convert %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  %11 = fir.convert %c4 : (index) -> i32
  %12 = fir.convert %8 : (!fir.ref<!fir.char<1,69>>) -> !fir.ref<i8>
  %13 = fir.convert %4 : (!fir.box<i1>) -> !fir.box<none>
  %14 = fir.call @_FortranAMinlocInteger4(%9, %10, %11, %12, %c4_i32, %13, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %15 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_0 = arith.constant 0 : index
  %16:3 = fir.box_dims %15, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %17 = fir.box_addr %15 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %18 = fir.shape_shift %16#0, %16#1 : (index, index) -> !fir.shapeshift<1>
  %19 = fir.array_load %17(%18) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  %c1_1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %20 = arith.subi %c1, %c1_1 : index
  %21 = fir.do_loop %arg1 = %c0_2 to %20 step %c1_1 unordered iter_args(%arg2 = %3) -> (!fir.array<1xi32>) {
    %23 = fir.array_fetch %19, %arg1 : (!fir.array<?xi32>, index) -> i32
    %24 = fir.array_update %arg2, %23, %arg1 : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
    fir.result %24 : !fir.array<1xi32>
  }
  fir.array_merge_store %3, %21 to %1 : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
  fir.freemem %17 : !fir.heap<!fir.array<?xi32>>
  %22 = fir.load %1 : !fir.ref<!fir.array<1xi32>>
  return %22 : !fir.array<1xi32>
}
// CHECK-LABEL:   func.func @_QPtestminloc_doesntwork1d_unknownsize(
// CHECK-SAME:                                                      %[[ARR:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
// CHECK-NOT:         fir.call @_FortranAMinlocInteger4x1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()
// CHECK:             fir.call @_FortranAMinlocInteger4({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NOT:         fir.call @_FortranAMinlocInteger4x1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()

// -----
// Check Minloc is not simplified when inputArr is characterType

func.func @_QPtestminloc_doesntwork1d_chars(%arg0: !fir.boxchar<1> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2 = fir.convert %1#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %3 = fir.alloca !fir.array<1xi32> {bindc_name = "testminloc_doesntwork1d_chars", uniq_name = "_QFtestminloc_doesntwork1d_charsEtestminloc_doesntwork1d_chars"}
  %4 = fir.shape %c1 : (index) -> !fir.shape<1>
  %5 = fir.array_load %3(%4) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
  %6 = fir.shape %c10 : (index) -> !fir.shape<1>
  %7 = fir.embox %2(%6) : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.char<1>>>
  %8 = fir.absent !fir.box<i1>
  %c4 = arith.constant 4 : index
  %false = arith.constant false
  %9 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %10 = fir.shape %c0 : (index) -> !fir.shape<1>
  %11 = fir.embox %9(%10) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %11 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %12 = fir.address_of(@_QQcl.74460ff3ef22ea53671c22344e1556b9) : !fir.ref<!fir.char<1,41>>
  %c4_i32 = arith.constant 4 : i32
  %13 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %14 = fir.convert %7 : (!fir.box<!fir.array<10x!fir.char<1>>>) -> !fir.box<none>
  %15 = fir.convert %c4 : (index) -> i32
  %16 = fir.convert %12 : (!fir.ref<!fir.char<1,41>>) -> !fir.ref<i8>
  %17 = fir.convert %8 : (!fir.box<i1>) -> !fir.box<none>
  %18 = fir.call @_FortranAMinlocCharacter(%13, %14, %15, %16, %c4_i32, %17, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %19 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_0 = arith.constant 0 : index
  %20:3 = fir.box_dims %19, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %21 = fir.box_addr %19 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %22 = fir.shape_shift %20#0, %20#1 : (index, index) -> !fir.shapeshift<1>
  %23 = fir.array_load %21(%22) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  %c1_1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %24 = arith.subi %c1, %c1_1 : index
  %25 = fir.do_loop %arg1 = %c0_2 to %24 step %c1_1 unordered iter_args(%arg2 = %5) -> (!fir.array<1xi32>) {
    %27 = fir.array_fetch %23, %arg1 : (!fir.array<?xi32>, index) -> i32
    %28 = fir.array_update %arg2, %27, %arg1 : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
    fir.result %28 : !fir.array<1xi32>
  }
  fir.array_merge_store %5, %25 to %3 : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
  fir.freemem %21 : !fir.heap<!fir.array<?xi32>>
  %26 = fir.load %3 : !fir.ref<!fir.array<1xi32>>
  return %26 : !fir.array<1xi32>
}

// CHECK-LABEL:   func.func @_QPtestminloc_doesntwork1d_chars(
// CHECK-SAME:                                                %[[ARR:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
// CHECK-NOT:         fir.call @_FortranAMinlocCharacterx1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()
// CHECK:             fir.call @_FortranAMinlocCharacter({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NOT:         fir.call @_FortranAMinlocCharacterx1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()

// -----
// Check Minloc is not simplified when mask is unknown rank

func.func @_QPtestminloc_doesntwork1d_unknownmask(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %c10 = arith.constant 10 : index
  %1 = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFtestminloc_doesntwork1d_unknownmaskEb"}
  %2 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>> {bindc_name = "mask", uniq_name = "_QFtestminloc_doesntwork1d_unknownmaskEmask"}
  %3 = fir.alloca !fir.heap<!fir.array<?x!fir.logical<4>>> {uniq_name = "_QFtestminloc_doesntwork1d_unknownmaskEmask.addr"}
  %4 = fir.alloca index {uniq_name = "_QFtestminloc_doesntwork1d_unknownmaskEmask.lb0"}
  %5 = fir.alloca index {uniq_name = "_QFtestminloc_doesntwork1d_unknownmaskEmask.ext0"}
  %6 = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
  fir.store %6 to %3 : !fir.ref<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  %c1 = arith.constant 1 : index
  %7 = fir.alloca !fir.array<1xi32> {bindc_name = "testminloc_doesntwork1d_unknownmask", uniq_name = "_QFtestminloc_doesntwork1d_unknownmaskEtestminloc_doesntwork1d_unknownmask"}
  %8 = fir.load %1 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %c0 = arith.constant 0 : index
  %10 = arith.cmpi sgt, %9, %c0 : index
  %11 = arith.select %10, %9, %c0 : index
  %12 = fir.allocmem !fir.array<?x!fir.logical<4>>, %11 {fir.must_be_heap = true, uniq_name = "_QFtestminloc_doesntwork1d_unknownmaskEmask.alloc"}
  fir.store %12 to %3 : !fir.ref<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  fir.store %11 to %5 : !fir.ref<index>
  %c1_0 = arith.constant 1 : index
  fir.store %c1_0 to %4 : !fir.ref<index>
  %13 = fir.shape %c1 : (index) -> !fir.shape<1>
  %14 = fir.array_load %7(%13) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
  %15 = fir.shape %c10 : (index) -> !fir.shape<1>
  %16 = fir.embox %arg0(%15) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %17 = fir.load %3 : !fir.ref<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  %18 = fir.convert %17 : (!fir.heap<!fir.array<?x!fir.logical<4>>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %19 = arith.cmpi ne, %18, %c0_i64 : i64
  %20 = fir.load %4 : !fir.ref<index>
  %21 = fir.load %5 : !fir.ref<index>
  %22 = fir.load %3 : !fir.ref<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  %23 = fir.shape_shift %20, %21 : (index, index) -> !fir.shapeshift<1>
  %24 = fir.embox %22(%23) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
  %25 = fir.absent !fir.box<!fir.array<?x!fir.logical<4>>>
  %26 = arith.select %19, %24, %25 : !fir.box<!fir.array<?x!fir.logical<4>>>
  %c4 = arith.constant 4 : index
  %false = arith.constant false
  %27 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0_1 = arith.constant 0 : index
  %28 = fir.shape %c0_1 : (index) -> !fir.shape<1>
  %29 = fir.embox %27(%28) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %29 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %30 = fir.address_of(@_QQcl.74460ff3ef22ea53671c22344e1556b9) : !fir.ref<!fir.char<1,41>>
  %c7_i32 = arith.constant 7 : i32
  %31 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %32 = fir.convert %16 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %33 = fir.convert %c4 : (index) -> i32
  %34 = fir.convert %30 : (!fir.ref<!fir.char<1,41>>) -> !fir.ref<i8>
  %35 = fir.convert %26 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  %36 = fir.call @_FortranAMinlocInteger4(%31, %32, %33, %34, %c7_i32, %35, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
  %37 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_2 = arith.constant 0 : index
  %38:3 = fir.box_dims %37, %c0_2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %39 = fir.box_addr %37 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %40 = fir.shape_shift %38#0, %38#1 : (index, index) -> !fir.shapeshift<1>
  %41 = fir.array_load %39(%40) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  %c1_3 = arith.constant 1 : index
  %c0_4 = arith.constant 0 : index
  %42 = arith.subi %c1, %c1_3 : index
  %43 = fir.do_loop %arg1 = %c0_4 to %42 step %c1_3 unordered iter_args(%arg2 = %14) -> (!fir.array<1xi32>) {
    %45 = fir.array_fetch %41, %arg1 : (!fir.array<?xi32>, index) -> i32
    %46 = fir.array_update %arg2, %45, %arg1 : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
    fir.result %46 : !fir.array<1xi32>
  }
  fir.array_merge_store %14, %43 to %7 : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
  fir.freemem %39 : !fir.heap<!fir.array<?xi32>>
  %44 = fir.load %7 : !fir.ref<!fir.array<1xi32>>
  return %44 : !fir.array<1xi32>
}

// CHECK-LABEL:   func.func @_QPtestminloc_doesntwork1d_unknownmask(
// CHECK-SAME:                                                      %[[ARR:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) -> !fir.array<1xi32> {
// CHECK-NOT:         fir.call @_FortranAMinlocInteger4x1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()
// CHECK:             fir.call @_FortranAMinlocInteger4({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NOT:         fir.call @_FortranAMinlocInteger4x1_i32_contract_simplified({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>) -> ()
