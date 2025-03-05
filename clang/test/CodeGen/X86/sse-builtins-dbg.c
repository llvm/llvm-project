// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-apple-macosx10.8.0 -target-feature +sse4.1 -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-apple-macosx10.8.0 -target-feature +sse4.1 -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s

// Test that intrinsic calls inlined from _mm_* wrappers have debug metadata.

#include <xmmintrin.h>

__m128 test_rsqrt_ss(__m128 x) {
  // CHECK-LABEL: test_rsqrt_ss
  // CHECK: call {{.*}}<4 x float> @llvm.x86.sse.rsqrt.ss({{.*}}, !dbg !{{.*}}
  // CHECK: ret <4 x float>
  return _mm_rsqrt_ss(x);
}
