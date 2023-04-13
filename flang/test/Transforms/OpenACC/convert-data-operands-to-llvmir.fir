// RUN: fir-opt -fir-openacc-data-operand-conversion='use-opaque-pointers=1' -split-input-file %s | FileCheck %s
// RUN: fir-opt -fir-openacc-data-operand-conversion='use-opaque-pointers=1' -split-input-file %s | fir-opt -split-input-file --fir-to-llvm-ir | FileCheck %s --check-prefix=LLVMIR

fir.global internal @_QFEa : !fir.array<10xf32> {
  %0 = fir.undefined !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @_QQsub1() attributes {fir.bindc_name = "arr"} {
  %0 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
  acc.data copy(%0 : !fir.ref<!fir.array<10xf32>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @_QQsub1() attributes {fir.bindc_name = "arr"} {
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ADDR]] : !fir.ref<!fir.array<10xf32>> to !llvm.ptr<array<10 x f32>>
// CHECK: acc.data copy(%[[CAST]] : !llvm.ptr<array<10 x f32>>)

// LLVMIR-LABEL: llvm.func @_QQsub1() attributes {fir.bindc_name = "arr"} {
// LLVMIR: %[[ADDR:.*]] = llvm.mlir.addressof @_QFEa : !llvm.ptr<array<10 x f32>>
// LLVMIR: acc.data copy(%[[ADDR]] : !llvm.ptr<array<10 x f32>>) {

// -----

fir.global internal @_QFEa : !fir.array<10xf32> {
  %0 = fir.undefined !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @_QQsub_enter_exit() attributes {fir.bindc_name = "a"} {
  %0 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
  acc.enter_data copyin(%0 : !fir.ref<!fir.array<10xf32>>)
  acc.exit_data copyout(%0 : !fir.ref<!fir.array<10xf32>>)
  return
}

// CHECK-LABEL: func.func @_QQsub_enter_exit() attributes {fir.bindc_name = "a"} {
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
// CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ADDR]] : !fir.ref<!fir.array<10xf32>> to !llvm.ptr<array<10 x f32>>
// CHECK: acc.enter_data copyin(%[[CAST0]] : !llvm.ptr<array<10 x f32>>)
// CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[ADDR]] : !fir.ref<!fir.array<10xf32>> to !llvm.ptr<array<10 x f32>>
// CHECK: acc.exit_data copyout(%[[CAST1]] : !llvm.ptr<array<10 x f32>>)

// LLVMIR-LABEL: llvm.func @_QQsub_enter_exit() attributes {fir.bindc_name = "a"} {
// LLVMIR: %[[ADDR:.*]] = llvm.mlir.addressof @_QFEa : !llvm.ptr<array<10 x f32>>
// LLVMIR: acc.enter_data copyin(%[[ADDR]] : !llvm.ptr<array<10 x f32>>)
// LLVMIR: acc.exit_data copyout(%[[ADDR]] : !llvm.ptr<array<10 x f32>>)

// -----

fir.global internal @_QFEa : !fir.array<10xf32> {
  %0 = fir.undefined !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @_QQsub_update() attributes {fir.bindc_name = "a"} {
  %0 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
  acc.update device(%0 : !fir.ref<!fir.array<10xf32>>)
  return
}

// CHECK-LABEL: func.func @_QQsub_update() attributes {fir.bindc_name = "a"} {
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ADDR]] : !fir.ref<!fir.array<10xf32>> to !llvm.ptr<array<10 x f32>>
// CHECK: acc.update device(%[[CAST]] : !llvm.ptr<array<10 x f32>>)

// LLVMIR-LABEL: llvm.func @_QQsub_update() attributes {fir.bindc_name = "a"} {
// LLVMIR: %[[ADDR:.*]] = llvm.mlir.addressof @_QFEa : !llvm.ptr<array<10 x f32>>
// LLVMIR: acc.update device(%[[ADDR]] : !llvm.ptr<array<10 x f32>>)

// -----

fir.global internal @_QFEa : !fir.array<10xf32> {
  %0 = fir.undefined !fir.array<10xf32>
  fir.has_value %0 : !fir.array<10xf32>
}

func.func @_QQsub_parallel() attributes {fir.bindc_name = "test"} {
  %0 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  acc.parallel copyin(%0: !fir.ref<!fir.array<10xf32>>) {
    acc.loop {
      acc.yield
    }
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @_QQsub_parallel() attributes {fir.bindc_name = "test"} {
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ADDR]] : !fir.ref<!fir.array<10xf32>> to !llvm.ptr<array<10 x f32>>
// CHECK: acc.parallel copyin(%[[CAST]] : !llvm.ptr<array<10 x f32>>) {
// CHECK:   acc.loop

// LLVMIR-LABEL: llvm.func @_QQsub_parallel() attributes {fir.bindc_name = "test"} {
// LLVMIR: %[[ADDR:.*]] = llvm.mlir.addressof @_QFEa : !llvm.ptr<array<10 x f32>>
// LLVMIR: acc.parallel copyin(%[[ADDR]] : !llvm.ptr<array<10 x f32>>) {
