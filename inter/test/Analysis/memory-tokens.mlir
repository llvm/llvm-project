// RUN: inter-opt %s --inter-infer-memory-tokens | FileCheck %s

func.func @aliasing(%input: !xw.ptr<#xw.global>,
                    %output: !xw.ptr<#xw.global>) attributes {
    xw.simd_width = 16 : i32} {
  %value, %read = xw.load %input
      : (!xw.ptr<#xw.global>) -> (!xw.simd<i32, 16>, !xw.mem.token)
  %written = xw.store %value -> %output
      : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> !xw.mem.token
  return
}

// CHECK-LABEL: func.func @aliasing
// CHECK: %[[VALUE:.*]], %[[READ:.*]] = xw.load
// CHECK: xw.store %[[VALUE]] -> {{.*}} after %[[READ]]
// CHECK-SAME: xw.tokens_inferred
// CHECK-NOT: llvm.
// CHECK-NOT: cf.
