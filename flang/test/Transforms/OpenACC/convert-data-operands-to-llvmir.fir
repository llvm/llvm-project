// RUN: fir-opt -fir-openacc-data-operand-conversion='use-opaque-pointers=1' -split-input-file %s | FileCheck %s
// RUN: fir-opt -fir-openacc-data-operand-conversion='use-opaque-pointers=1' -split-input-file %s | fir-opt -split-input-file --fir-to-llvm-ir | FileCheck %s --check-prefix=LLVMIR

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
