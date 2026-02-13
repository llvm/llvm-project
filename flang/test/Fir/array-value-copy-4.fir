// Test conversion of array_load with array of derived-type.
// Conversion was previously crashing as reported in
// https://github.com/llvm/llvm-project/issues/59342.

// RUN: fir-opt --array-value-copy %s | FileCheck %s
// RUN: fir-opt --array-value-copy="optimize-conflicts=true" %s | FileCheck %s

func.func @_QMmodPsub1(%arg0: !fir.box<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>> {fir.bindc_name = "x"}) {
  %0 = fir.alloca !fir.box<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  %c3_i64 = arith.constant 3 : i64
  %1 = fir.convert %c3_i64 : (i64) -> index
  %c1_i64 = arith.constant 1 : i64
  %2 = fir.convert %c1_i64 : (i64) -> index
  %c4_i64 = arith.constant 4 : i64
  %3 = fir.convert %c4_i64 : (i64) -> index
  %c0 = arith.constant 0 : index
  %4 = arith.subi %3, %1 : index
  %5 = arith.addi %4, %2 : index
  %6 = arith.divsi %5, %2 : index
  %7 = arith.cmpi sgt, %6, %c0 : index
  %8 = arith.select %7, %6, %c0 : index
  %9 = fir.slice %1, %3, %2 : (index, index, index) -> !fir.slice<1>
  %10 = fir.array_load %arg0 [%9] : (!fir.box<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, !fir.slice<1>) -> !fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  %c1_i64_0 = arith.constant 1 : i64
  %11 = fir.convert %c1_i64_0 : (i64) -> index
  %c1_i64_1 = arith.constant 1 : i64
  %12 = fir.convert %c1_i64_1 : (i64) -> index
  %c2_i64 = arith.constant 2 : i64
  %13 = fir.convert %c2_i64 : (i64) -> index
  %14 = fir.slice %11, %13, %12 : (index, index, index) -> !fir.slice<1>
  %15 = fir.array_load %arg0 [%14] : (!fir.box<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, !fir.slice<1>) -> !fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  %c1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %16 = arith.subi %8, %c1 : index
  %17 = fir.do_loop %arg1 = %c0_2 to %16 step %c1 unordered iter_args(%arg2 = %10) -> (!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) {
    %18 = fir.array_access %15, %arg1 : (!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, index) -> !fir.ref<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
    %19 = fir.array_access %arg2, %arg1 : (!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, index) -> !fir.ref<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
    %20 = fir.embox %19 : (!fir.ref<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
    %21 = fir.embox %18 : (!fir.ref<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
    fir.store %20 to %0 : !fir.ref<!fir.box<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
    %22 = fir.address_of(@_QQclX2E2F64756D6D792E66393000) : !fir.ref<!fir.char<1,12>>
    %c9_i32 = arith.constant 9 : i32
    %23 = fir.convert %0 : (!fir.ref<!fir.box<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>) -> !fir.ref<!fir.box<none>>
    %24 = fir.convert %21 : (!fir.box<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<none>
    %25 = fir.convert %22 : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
    fir.call @_FortranAAssign(%23, %24, %25, %c9_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
    %27 = fir.array_amend %arg2, %19 : (!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
    fir.result %27 : !fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  }
  fir.array_merge_store %10, %17 to %arg0[%9] : !fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.box<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, !fir.slice<1>
  return
}

// CHECK-LABEL: func.func @_QMmodPsub1(
// CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>> {fir.bindc_name = "x"}) {
// CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %c0{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, index) -> (index, index, index)
// CHECK: %[[ALLOCMEM:.*]] = fir.allocmem !fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, %[[BOX_DIMS]]#1
// CHECK: %{{.*}} = fir.embox %[[ALLOCMEM]](%{{.*}}) : (!fir.heap<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMmodTrec1{dat:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>
