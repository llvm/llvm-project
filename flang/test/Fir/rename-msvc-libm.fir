// RUN: fir-opt --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s -DHYPOTF=hypotf
// RUN: fir-opt --fir-to-llvm-ir="target=aarch64-pc-windows-msvc" %s | FileCheck %s -DHYPOTF=_hypotf

// Test hypotf renaming

func.func private @hypotf(f32, f32) -> f32

// CHECK: llvm.func @[[HYPOTF]](f32, f32) -> f32

func.func @call_hypotf(%arg0 : f32, %arg1 : f32) -> f32 {
  %0 = fir.call @hypotf(%arg0, %arg1) : (f32, f32) -> f32
  return %0 : f32
}

// CHECK-LABEL: llvm.func @call_hypotf
// CHECK-SAME: %[[arg0:.*]]: f32, %[[arg1:.*]]: f32
// CHECK: llvm.call @[[HYPOTF]](%[[arg0]], %[[arg1]]) : (f32, f32) -> f32
