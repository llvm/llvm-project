// Test FIR types conversion.

// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck %s


// Test sequence types `!fir.array`

func private @foo0(%arg0: !fir.array<10x10xi64>)
// CHECK-LABEL: foo0 
// CHECK-SAME: !llvm.array<10 x array<10 x i64>>
func private @foo1(%arg0: !fir.array<?xf32>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<f32>
func private @foo2(%arg0: !fir.array<?x?xf128>)
// CHECK-LABEL: foo2 
// CHECK-SAME: !llvm.ptr<f128>
func private @foo3(%arg0: !fir.array<*:i32>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.ptr<i32>

// -----

// Test reference types `!fir.ref`

func private @foo0(%arg0: !fir.ref<i32>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.ptr<i32>
func private @foo1(%arg0: !fir.ref<!fir.array<10xf32>>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<array<10 x f32>>

