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
// CHECK:           %[[CI32_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[SUM:.*]] = fir.alloca i32
// CHECK:           fir.store %[[CI32_0]] to %[[SUM]] : !fir.ref<i32>
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_I32:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_I32]], %[[CINDEX_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<i32>
// CHECK:             %[[SUM_VAL:.*]] = fir.load %[[SUM]] : !fir.ref<i32>
// CHECK:             %[[NEW_SUM:.*]] = arith.addi %[[ITEM_VAL]], %[[SUM_VAL]] : i32
// CHECK:             fir.store %[[NEW_SUM]] to %[[SUM]] : !fir.ref<i32>
// CHECK:           }
// CHECK:           %[[RET:.*]] = fir.load %[[SUM]] : !fir.ref<i32>
// CHECK:           return %[[RET]] : i32
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
// CHECK:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[SUM:.*]] = fir.alloca f64
// CHECK:           fir.store %[[ZERO]] to %[[SUM]] : !fir.ref<f64>
// CHECK:           %[[CINDEX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARR_BOX_F64:.*]] = fir.convert %[[ARR]] : (!fir.box<none>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR_BOX_F64]], %[[CINDEX_0]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[CINDEX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[EXTENT:.*]] = arith.subi %[[DIMS]]#1, %[[CINDEX_1]] : index
// CHECK:           fir.do_loop %[[ITER:.*]] = %[[CINDEX_0]] to %[[EXTENT]] step %[[CINDEX_1]] {
// CHECK:             %[[ITEM:.*]] = fir.coordinate_of %[[ARR_BOX_F64]], %[[ITER]] : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:             %[[ITEM_VAL:.*]] = fir.load %[[ITEM]] : !fir.ref<f64>
// CHECK:             %[[SUM_VAL:.*]] = fir.load %[[SUM]] : !fir.ref<f64>
// CHECK:             %[[NEW_SUM:.*]] = arith.addf %[[ITEM_VAL]], %[[SUM_VAL]] : f64
// CHECK:             fir.store %[[NEW_SUM]] to %[[SUM]] : !fir.ref<f64>
// CHECK:           }
// CHECK:           %[[RES:.*]] = fir.load %[[SUM]] : !fir.ref<f64>
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
// CHECK:           fir.do_loop %[[ITER:.*]] = %{{.*}} to %[[EXTENT]] step %[[CINDEX_1]] {
// CHECK:             %{{.*}} = fir.coordinate_of %[[ARR_BOX_I32]], %[[ITER]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           }
// CHECK:           return %{{.*}} : i32
// CHECK:         }
