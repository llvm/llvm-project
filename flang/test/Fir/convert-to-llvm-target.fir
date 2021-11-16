// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix INT64
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s --check-prefixes INT64
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=i386-unknown-linux-gnu" %s | FileCheck %s --check-prefixes INT32
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=powerpc64le-unknown-linux-gn" %s | FileCheck %s --check-prefixes INT64

//=============================================================================
// SUMMARY: Tests for FIR --> LLVM MLIR conversion that *depend* on the target
//=============================================================================

// Test fir.emboxchar

func @test_embox(%char_array : !fir.ref<!fir.char<1,?>>) -> () {
  %c10 = arith.constant 10 : i64
  %box_char = fir.emboxchar %char_array, %c10 : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  return
}

// INT64-LABEL: test_embox
// INT64-SAME: (%[[char_array:.*]]: !llvm.ptr<i8>)
// INT64:    %[[c10:.*]] = llvm.mlir.constant(10 : i64) : i64
// INT64:    %[[empty_struct:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i{{.*}})>
// INT64:    %[[struct_with_buffer:.*]] = llvm.insertvalue %[[char_array]], %[[empty_struct]][0 : i32] : !llvm.struct<(ptr<i8>, i{{.*}})>
// INT64:    %{{.*}} = llvm.insertvalue %[[c10]], %[[struct_with_buffer]][1 : i32] : !llvm.struct<(ptr<i8>, i{{.*}})>
// INT64-NEXT:    llvm.return

// INT32-LABEL:  llvm.func @test_embox
// INT32-SAME: %[[char_array:.*]]: !llvm.ptr<i8>)
// INT32:    %[[c10:.*]] = llvm.mlir.constant(10 : i64) : i64
// INT32:    %[[empty_struct:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i32)>
// INT32:    %[[c10_truncated:.*]] = llvm.trunc %[[c10]] : i64 to i32
// INT32:    %[[struct_with_buffer:.*]] = llvm.insertvalue %[[char_array]], %[[empty_struct]][0 : i32] : !llvm.struct<(ptr<i8>, i32)>
// INT32:    %{{.*}} = llvm.insertvalue %[[c10_truncated:.*]], %[[struct_with_buffer]][1 : i32] : !llvm.struct<(ptr<i8>, i32)>
// INT32-NEXT:    llvm.return

// -----

// Test fir.unboxchar

func @unboxchar_i8(%arg0 : !fir.boxchar<1>) -> () {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  return
}

// INT64-LABEL: llvm.func @unboxchar_i8
// INT64-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i8>, i64)>
// INT64:  %{{.*}} = llvm.extractvalue %[[box_char]][0 : i32] : !llvm.struct<(ptr<i8>, i64)>
// INT64:  %{{.*}} = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i8>, i64)>
// INT64-NEXT:  llvm.return

// INT32-LABEL: llvm.func @unboxchar_i8
// INT32-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i8>, i32)>
// INT32:  %{{.*}} = llvm.extractvalue %[[box_char]][0 : i32] : !llvm.struct<(ptr<i8>, i32)>
// INT32:  %[[len_unextended:.*]] = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i8>, i32)>
// INT32:  %{{.*}} = llvm.sext %[[len_unextended]] : i32 to i64
// INT32-NEXT:  llvm.return

func @unboxchar_i32(%arg0 : !fir.boxchar<4>) -> () {
  fir.unboxchar %arg0 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4>>, i64)
  return
}

// INT64-LABEL: llvm.func @unboxchar_i32
// INT64-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i32>, i64)>
// INT64:  %{{.*}} = llvm.extractvalue %[[box_char]][0 : i32] : !llvm.struct<(ptr<i32>, i64)>
// INT64:  %{{.*}} = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i32>, i64)>
// INT64-NEXT:  llvm.return

// INT32-LABEL: llvm.func @unboxchar_i32
// INT32-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i32>, i32)>
// INT32:  %{{.*}} = llvm.extractvalue %[[box_char]][0 : i32] : !llvm.struct<(ptr<i32>, i32)>
// INT32:  %[[len_unextended:.*]] = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i32>, i32)>
// INT32:  %{{.*}} = llvm.sext %[[len_unextended]] : i32 to i64
// INT32-NEXT:  llvm.return

// -----

// Test fir.boxchar_len

func @boxchar_len_i8_i32(%arg0 : !fir.boxchar<1>) -> () {
  fir.boxchar_len %arg0 : (!fir.boxchar<1>) -> i32
  return
}

// INT64-LABEL: llvm.func @boxchar_len_i8_i32
// INT64-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i8>, i64)>
// INT64:  %[[len:.*]] = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i8>, i64)>
// INT64: %{{.*}} = llvm.trunc %[[len]] : i64 to i32
// INT64-NEXT:  llvm.return

// INT32-LABEL: llvm.func @boxchar_len_i8_i32
// INT32-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i8>, i32)>
// INT32:  %{{.*}} = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i8>, i32)>
// INT32-NOT:  llvm.trunc
// INT32-NOT:  llvm.sext
// INT32-NEXT:  llvm.return

func @boxchar_len_i8_i64(%arg0 : !fir.boxchar<1>) -> () {
  fir.boxchar_len %arg0 : (!fir.boxchar<1>) -> i64
  return
}

// INT64-LABEL: llvm.func @boxchar_len_i8_i64
// INT64-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i8>, i64)>
// INT64:  %{{.*}} = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i8>, i64)>
// INT64-NOT:  llvm.trunc
// INT64-NOT:  llvm.sext
// INT64-NEXT:  llvm.return

// INT32-LABEL: llvm.func @boxchar_len_i8_i64
// INT32-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i8>, i32)>
// INT32:  %[[len:.*]] = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i8>, i32)>
// INT32:  %{{.*}} = llvm.sext %0 : i32 to i64
// INT32-NEXT:  llvm.return

func @boxchar_len_i32_i32(%arg0 : !fir.boxchar<4>) -> () {
  fir.boxchar_len %arg0 : (!fir.boxchar<4>) -> i32
  return
}

// INT64-LABEL: llvm.func @boxchar_len_i32_i32
// INT64-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i32>, i64)>
// INT64:  %[[len:.*]] = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i32>, i64)>
// INT64: %{{.*}} = llvm.trunc %[[len]] : i64 to i32
// INT64-NEXT:  llvm.return

// INT32-LABEL: llvm.func @boxchar_len_i32_i32
// INT32-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i32>, i32)>
// INT32:  %{{.*}} = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i32>, i32)>
// INT32-NOT:  llvm.trunc
// INT32-NOT:  llvm.sext
// INT32-NEXT:  llvm.return

func @boxchar_len_i32_i64(%arg0 : !fir.boxchar<4>) -> (i64) {
  %0 = fir.boxchar_len %arg0 : (!fir.boxchar<4>) -> i64
  return %0 : i64
}

// INT64-LABEL: llvm.func @boxchar_len_i32_i64
// INT64-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i32>, i64)>
// INT64:  %{{.*}} = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i32>, i64)>
// INT64-NOT:  llvm.trunc
// INT64-NOT:  llvm.sext
// INT64-NEXT:  llvm.return

// INT32-LABEL: llvm.func @boxchar_len_i32_i64
// INT32-SAME: %[[box_char:.*]]: !llvm.struct<(ptr<i32>, i32)>
// INT32:  %[[len:.*]] = llvm.extractvalue %[[box_char]][1 : i32] : !llvm.struct<(ptr<i32>, i32)>
// INT32:  %{{.*}} = llvm.sext %0 : i32 to i64
// INT32-NEXT:  llvm.return
