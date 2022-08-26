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
// CHECK:           %[[RES:.*]] = fir.call @_FortranASumInteger4_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> i32
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           return %{{.*}} : i32
// CHECK:         }
// CHECK:         func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}

// CHECK-LABEL:   func.func private @_FortranASumInteger4_simplified(
// CHECK-SAME:                                                       %[[ARR:.*]]: !fir.box<none>) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[CINDEX_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[CI32_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[SUM:.*]] = %[[CI32_0]]) -> (i32) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:             %[[NEW_SUM:.*]] = arith.addi %[[ITEM_VAL]], %[[SUM]] : i32
// CHECK:             fir.result %[[NEW_SUM]] : i32
// CHECK:           }
// CHECK:           return %[[RES]] : i32
// CHECK:         }

// -----

// Call to SUM with 2D I32 arrays is not replaced.
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
// CHECK-NOT:       fir.call @_FortranASumInteger4_simplified({{.*}})
// CHECK:           fir.call @_FortranASumInteger4({{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
// CHECK-NOT:       fir.call @_FortranASumInteger4_simplified({{.*}})

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
// CHECK:           %[[RES:.*]] = fir.call @_FortranASumReal8_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> f64
// CHECK-NOT:       fir.call @_FortranASumReal8({{.*}})
// CHECK:           return %{{.*}} : f64
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranASumReal8_simplified(
// CHECK-SAME:                                                    %[[ARR:.*]]: !fir.box<none>) -> f64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_F64:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_F64]], %[[CINDEX_0]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[SUM]] = %[[ZERO]]) -> (f64) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_F64]], %[[ITER]] : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<f64>
// CHECK:             %[[NEW_SUM:.*]] = arith.addf %[[ITEM_VAL]], %[[SUM]] : f64
// CHECK:             fir.result %[[NEW_SUM]] : f64
// CHECK:           }
// CHECK:           return %[[RES]] : f64
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
// CHECK-NOT:       fir.call @_FortranACppSumComplex4_simplified({{.*}})
// CHECK:           fir.call @_FortranACppSumComplex4({{.*}}) : (!fir.ref<complex<f32>>, !fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> none
// CHECK-NOT:       fir.call @_FortranACppSumComplex4_simplified({{.*}})

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
// CHECK:           fir.call @_FortranASumInteger4_simplified(%{{.*}})
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:         }

// CHECK-LABEL:   func.func @sum_1d_callb(%{{.*}}) -> i32 {
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:           fir.call @_FortranASumInteger4_simplified(%{{.*}})
// CHECK-NOT:       fir.call @_FortranASumInteger4({{.*}})
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranASumInteger4_simplified({{.*}}) -> i32 {{.*}} {
// CHECK:           return %{{.*}} : i32
// CHECK:         }
// CHECK-NOT:   func.func private @_FortranASumInteger4_simplified({{.*}})

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
// CHECK:           %{{.*}} = fir.call @_FortranASumInteger4_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> i32
// CHECK:           return %{{.*}} : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranASumInteger4_simplified(%{{.*}}) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %{{.*}} : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
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
// CHECK:           %[[RES:.*]] = fir.call @_FortranAMaxvalInteger4_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> i32
// CHECK:           return %{{.*}} : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAMaxvalInteger4_simplified(
// CHECK-SAME:                                                       %[[ARR:.*]]: !fir.box<none>) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[CINDEX_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[CI32_MININT:.*]] = arith.constant -2147483648 : i32
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
// CHECK:           %[[RES:.*]] = fir.call @_FortranAMaxvalReal8_simplified(%[[A_BOX_NONE]]) : (!fir.box<none>) -> f64
// CHECK:           return %{{.*}} : f64
// CHECK:         }

// CHECK-LABEL:   func.func private @_FortranAMaxvalReal8_simplified(
// CHECK-SAME:                                                    %[[ARR:.*]]: !fir.box<none>) -> f64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_F64:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_F64]], %[[CINDEX_0]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           %[[NEG_DBL_MAX:.*]] = arith.constant -1.7976931348623157E+308 : f64
// CHECK:           %[[RES:.*]] = fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] iter_args(%[[MAX]] = %[[NEG_DBL_MAX]]) -> (f64) {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_F64]], %[[ITER]] : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<f64>
// CHECK:             %[[NEW_MAX:.*]] = arith.maxf %[[ITEM_VAL]], %[[MAX]] : f64
// CHECK:             fir.result %[[NEW_MAX]] : f64
// CHECK:           }
// CHECK:           return %[[RES]] : f64
// CHECK:         }
