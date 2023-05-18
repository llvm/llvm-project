// RUN: fir-opt --split-input-file --fir-memref-dataflow-opt %s | FileCheck %s

// Test that all load-store chains are removed

func.func @load_store_chain_removal(%arg0: !fir.ref<!fir.array<60xi32>>, %arg1: !fir.ref<!fir.array<60xi32>>, %arg2: !fir.ref<!fir.array<60xi32>>) {
  %c1_i64 = arith.constant 1 : i64
  %c60 = arith.constant 60 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFf1dcEi"}
  %1 = fir.alloca !fir.array<60xi32> {bindc_name = "t1", uniq_name = "_QFf1dcEt1"}
  cf.br ^bb1(%c1, %c60 : index, index)
^bb1(%2: index, %3: index):  // 2 preds: ^bb0, ^bb2
  %4 = arith.cmpi sgt, %3, %c0 : index
  cf.cond_br %4, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %5 = fir.convert %2 : (index) -> i32
  fir.store %5 to %0 : !fir.ref<i32>
  %6 = fir.load %0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> i64
  %8 = arith.subi %7, %c1_i64 : i64
  %9 = fir.coordinate_of %arg0, %8 : (!fir.ref<!fir.array<60xi32>>, i64) -> !fir.ref<i32>
  %10 = fir.load %9 : !fir.ref<i32>
  %11 = arith.addi %10, %10 : i32
  %12 = fir.coordinate_of %1, %8 : (!fir.ref<!fir.array<60xi32>>, i64) -> !fir.ref<i32>
  fir.store %11 to %12 : !fir.ref<i32>
  %13 = arith.addi %2, %c1 : index
  %14 = arith.subi %3, %c1 : index
  cf.br ^bb1(%13, %14 : index, index)
^bb3:  // pred: ^bb1
  %15 = fir.convert %2 : (index) -> i32
  fir.store %15 to %0 : !fir.ref<i32>
  cf.br ^bb4(%c1, %c60 : index, index)
^bb4(%16: index, %17: index):  // 2 preds: ^bb3, ^bb5
  %18 = arith.cmpi sgt, %17, %c0 : index
  cf.cond_br %18, ^bb5, ^bb6
^bb5:  // pred: ^bb4
  %19 = fir.convert %16 : (index) -> i32
  fir.store %19 to %0 : !fir.ref<i32>
  %20 = fir.load %0 : !fir.ref<i32>
  %21 = fir.convert %20 : (i32) -> i64
  %22 = arith.subi %21, %c1_i64 : i64
  %23 = fir.coordinate_of %1, %22 : (!fir.ref<!fir.array<60xi32>>, i64) -> !fir.ref<i32>
  %24 = fir.load %23 : !fir.ref<i32>
  %25 = fir.coordinate_of %arg1, %22 : (!fir.ref<!fir.array<60xi32>>, i64) -> !fir.ref<i32>
  %26 = fir.load %25 : !fir.ref<i32>
  %27 = arith.muli %24, %26 : i32
  %28 = fir.coordinate_of %arg2, %22 : (!fir.ref<!fir.array<60xi32>>, i64) -> !fir.ref<i32>
  fir.store %27 to %28 : !fir.ref<i32>
  %29 = arith.addi %16, %c1 : index
  %30 = arith.subi %17, %c1 : index
  cf.br ^bb4(%29, %30 : index, index)
^bb6:  // pred: ^bb4
  %31 = fir.convert %16 : (index) -> i32
  fir.store %31 to %0 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func @load_store_chain_removal
// CHECK-LABEL: ^bb1
// CHECK-LABEL: ^bb2:
// Make sure the previous fir.store/fir.load pair have been elimated and we 
// preserve the last pair of fir.load/fir.store.
// CHECK-COUNT-1: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
// CHECK-COUNT-1: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK-LABEL: ^bb3:
// Make sure the fir.store has been removed.
// CHECK-NOT:     fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK-LABEL: ^bb5:
// CHECK:         %{{.*}} = fir.convert %{{.*}} : (index) -> i32
// Check that the fir.store/fir.load pair has been removed between the convert.
// CHECK-NOT:     fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK-NOT:     %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
// CHECK:         %{{.*}} = fir.convert %{{.*}} : (i32) -> i64
// CHECK:         %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
// CHECK:         %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
// CHECK:         fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
// CHECK-LABEL: ^bb6:
// CHECK-NOT:     fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
